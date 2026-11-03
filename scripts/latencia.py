"""Latência e custo estimado por 1 milhão de reclamações: LLM × BERTimbau × TF-IDF.

Uso: uv run --group train python scripts/latencia.py
- TF-IDF e BERTimbau em CPU (cenário barato de produção), uma reclamação por vez.
- LLM pela API, uma reclamação por requisição (cenário online) e em lotes de 20 (offline).
Premissas de preço explícitas em PRECOS; o resultado vai para reports/latencia_custo.json.
"""

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from project_nov26_p4_destilacao.llm import MODELO_PADRAO, chamar_llm
from project_nov26_p4_destilacao.rotular import INSTRUCOES

RAIZ = Path(__file__).resolve().parents[1]
N_LOCAL, N_LLM = 100, 20
PRECOS = {
    "data_consulta": "2026-11-03",
    "llm_entrada_usd_por_1m_tokens": 0.60,  # versão paga do mesmo modelo no OpenRouter
    "llm_saida_usd_por_1m_tokens": 2.40,
    "cpu_usd_por_hora": 0.10,  # premissa: instância de 2 vCPU sob demanda
}


def cronometrar(funcao, textos: list[str]) -> list[float]:
    tempos = []
    for t in textos:
        inicio = time.perf_counter()
        funcao(t)
        tempos.append(time.perf_counter() - inicio)
    return tempos


def main() -> None:
    torch.set_num_threads(2)
    textos = pd.read_csv(RAIZ / "data/referencia.csv")["texto"].tolist()
    cpu = next(
        (
            linha.split(":", 1)[1].strip()
            for linha in Path("/proc/cpuinfo").read_text().splitlines()
            if linha.startswith("model name")
        ),
        platform.machine(),
    )
    resultado = {"hardware": f"{cpu}, 2 threads (torch.set_num_threads)", "premissas": PRECOS}

    tfidf = joblib.load(RAIZ / "models/tfidf.joblib")
    t_tfidf = cronometrar(lambda t: tfidf.predict([t]), textos[:N_LOCAL])

    pasta = RAIZ / "models/bertimbau-triagem"
    tok = AutoTokenizer.from_pretrained(pasta)
    bert = AutoModelForSequenceClassification.from_pretrained(pasta).eval()

    def prever_bert(texto: str) -> None:
        with torch.inference_mode():
            bert(**tok(texto, truncation=True, max_length=128, return_tensors="pt"))

    t_bert = cronometrar(prever_bert, textos[:N_LOCAL])

    # Vazão em lote na CPU (cenário offline) para a conta de custo
    inicio = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, N_LOCAL, 32):
            lote = textos[i : i + 32]
            bert(**tok(lote, truncation=True, max_length=128, padding=True, return_tensors="pt"))
    por_hora_bert = N_LOCAL / (time.perf_counter() - inicio) * 3600

    t_llm, tokens_in, tokens_out = [], [], []
    for t in textos[:N_LLM]:
        msgs = [{"role": "system", "content": INSTRUCOES}, {"role": "user", "content": f"1. {t}"}]
        r = chamar_llm(msgs, MODELO_PADRAO)
        t_llm.append(r["latencia_s"])
        tokens_in.append(r["tokens_entrada"])
        tokens_out.append(r["tokens_saida"])

    lotes = json.loads((RAIZ / "reports/tokens_lotes.json").read_text())

    def custo_llm(entrada: float, saida: float) -> float:
        return round(
            (
                entrada * PRECOS["llm_entrada_usd_por_1m_tokens"]
                + saida * PRECOS["llm_saida_usd_por_1m_tokens"]
            ),
            2,
        )  # por 1M reclamações

    resultado["latencia_ms_p50"] = {
        "tfidf_cpu": round(float(np.median(t_tfidf)) * 1000, 2),
        "bertimbau_cpu": round(float(np.median(t_bert)) * 1000, 1),
        "llm_api": round(float(np.median(t_llm)) * 1000, 0),
    }
    resultado["custo_usd_por_1m_reclamacoes"] = {
        "llm_uma_por_requisicao": custo_llm(np.mean(tokens_in), np.mean(tokens_out)),
        "llm_em_lotes_de_20": custo_llm(
            lotes["entrada_por_reclamacao"], lotes["saida_por_reclamacao"]
        ),
        "bertimbau_cpu": round(1e6 / por_hora_bert * PRECOS["cpu_usd_por_hora"], 2),
    }
    resultado["detalhes"] = {
        "n_local": N_LOCAL,
        "n_llm": N_LLM,
        "tokens_por_reclamacao_llm_individual": [
            round(float(np.mean(tokens_in))),
            round(float(np.mean(tokens_out))),
        ],
        "bertimbau_reclamacoes_por_hora_cpu": round(por_hora_bert),
    }
    (RAIZ / "reports/latencia_custo.json").write_text(json.dumps(resultado, indent=2))
    print(json.dumps(resultado, indent=2))


if __name__ == "__main__":
    main()
