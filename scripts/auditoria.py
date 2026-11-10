"""Auditoria da referência: previsões por item, IC95 por bootstrap e quase duplicatas.

Uso: uv run --group train python scripts/auditoria.py   (CPU, fp32, sem custo de API)
Grava:
- reports/previsoes_referencia.csv: verdade e previsão de professor, aluno e TF-IDF por item
- reports/auditoria.json: métricas em CPU fp32, IC95 (bootstrap pareado, seed 42) e a
  sensibilidade à remoção dos itens da referência que são quase duplicatas do treino
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from project_nov26_p4_destilacao.normalizacao import normalizar
from project_nov26_p4_destilacao.rotular import CLASSES

RAIZ = Path(__file__).resolve().parents[1]
SEED, REAMOSTRAGENS = 42, 2000


def f1(y: np.ndarray, p: np.ndarray) -> float:
    return float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0))


def prever_aluno(textos: list[str]) -> list[str]:
    pasta = RAIZ / "models/bertimbau-triagem"
    tok = AutoTokenizer.from_pretrained(pasta)
    modelo = AutoModelForSequenceClassification.from_pretrained(pasta).eval()
    saida: list[str] = []
    with torch.inference_mode():
        for i in range(0, len(textos), 32):
            lote = tok(
                textos[i : i + 32],
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
            saida += [modelo.config.id2label[int(k)] for k in modelo(**lote).logits.argmax(-1)]
    return saida


def ic(valores: list[float]) -> list[float]:
    return np.percentile(valores, [2.5, 97.5]).round(3).tolist()


def main() -> None:
    ref = pd.read_csv(RAIZ / "data/referencia.csv")
    treino = pd.read_csv(RAIZ / "data/rotulos_llm.csv")
    prof = pd.read_csv(RAIZ / "reports/professor_na_referencia.csv")
    assert prof["id"].tolist() == ref["id"].tolist(), "ordem da referência mudou"

    previsoes = pd.DataFrame(
        {
            "id": ref["id"],
            "rotulo": ref["rotulo"],
            "professor": prof["rotulo_llm"],
            "aluno": prever_aluno(ref["texto"].tolist()),
            "tfidf": joblib.load(RAIZ / "models/tfidf.joblib").predict(ref["texto"]),
        }
    )
    # Itens da referência cujo texto normalizado também está no treino
    quase_dup = normalizar(ref["texto"]).isin(set(normalizar(treino["texto"])))
    previsoes["quase_duplicata_do_treino"] = quase_dup.to_numpy()
    previsoes.to_csv(RAIZ / "reports/previsoes_referencia.csv", index=False)

    colunas = ("rotulo", "aluno", "tfidf", "professor")
    y, aluno, tfidf, prof_ = (previsoes[c].to_numpy() for c in colunas)
    rng = np.random.default_rng(SEED)
    chaves = ("aluno", "aluno_menos_tfidf", "prof_menos_aluno", "razao")
    boot: dict[str, list[float]] = {k: [] for k in chaves}
    for _ in range(REAMOSTRAGENS):
        i = rng.integers(0, len(y), len(y))
        fa, ft, fp = f1(y[i], aluno[i]), f1(y[i], tfidf[i]), f1(y[i], prof_[i])
        boot["aluno"].append(fa)
        boot["aluno_menos_tfidf"].append(fa - ft)
        boot["prof_menos_aluno"].append(fp - fa)
        boot["razao"].append(fa / fp)

    sem_dup = ~quase_dup.to_numpy()
    resumo = {
        "ambiente": "CPU, fp32 (o relatório do treino, bertimbau.json, usa GPU fp16)",
        "n_referencia": len(y),
        "f1_macro": {
            "professor": round(f1(y, prof_), 3),
            "aluno": round(f1(y, aluno), 3),
            "tfidf": round(f1(y, tfidf), 3),
        },
        "ic95_bootstrap": {k: ic(v) for k, v in boot.items()},
        "reamostragens": REAMOSTRAGENS,
        "quase_duplicatas_no_treino": int(quase_dup.sum()),
        "f1_macro_sem_quase_duplicatas": {
            "n": int(sem_dup.sum()),
            "professor": round(f1(y[sem_dup], prof_[sem_dup]), 3),
            "aluno": round(f1(y[sem_dup], aluno[sem_dup]), 3),
            "tfidf": round(f1(y[sem_dup], tfidf[sem_dup]), 3),
        },
        "concordancia_professor": {
            "150_primeiras": round(float((y[:150] == prof_[:150]).mean()), 3),
            "300_novas": round(float((y[150:] == prof_[150:]).mean()), 3),
        },
    }
    (RAIZ / "reports/auditoria.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
