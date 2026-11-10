"""Roda o projeto inteiro: amostra → professor → baseline → aluno → auditoria → figuras → latência.

Uso: uv run --group train python scripts/pipeline.py [opções]
  --de rotular       começa nesta etapa (as anteriores já rodaram)
  --ate finetune     para depois desta etapa
  --epocas 5         repassado ao finetune.py (também --batch)
  --publicar         no fim, envia o modelo ao Hugging Face (precisa de HF_TOKEN)
Respostas do LLM vêm do cache quando o texto e o prompt não mudaram: refazer custa pouco.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
# Nome da etapa → comando (sys.executable: o mesmo Python/ambiente do pipeline)
ETAPAS = {
    "amostrar": ["scripts/amostrar.py"],
    "rotular": ["scripts/rotular_llm.py"],
    "baseline": ["scripts/baseline.py"],
    "finetune": ["scripts/finetune.py"],
    "auditoria": ["scripts/auditoria.py"],
    "figuras": ["scripts/figuras.py"],
    "latencia": ["scripts/latencia.py"],
    "testes": ["-m", "pytest", "-q"],
}


def conferir_antes(etapa: str) -> None:
    # Falha cedo, com mensagem clara, em vez de quebrar no meio da etapa
    if etapa == "rotular":
        ref = pd.read_csv(RAIZ / "data/referencia.csv", keep_default_na=False)
        faltam = int((ref["rotulo"] == "").sum())
        if faltam:
            sys.exit(f"data/referencia.csv tem {faltam} itens sem rótulo: rotule antes de seguir.")
    if etapa == "rotular" and not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY não definida (crie o .env.shared, ver README).")
    if etapa == "publicar" and not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN não definido (crie o .env.shared, ver README).")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--de", choices=list(ETAPAS), default="amostrar")
    p.add_argument("--ate", choices=list(ETAPAS), default="testes")
    p.add_argument("--epocas", type=int, default=5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--publicar", action="store_true")
    args = p.parse_args()
    load_dotenv(find_dotenv(".env.shared", usecwd=True))  # repassado às etapas

    nomes = list(ETAPAS)
    escolhidas = nomes[nomes.index(args.de) : nomes.index(args.ate) + 1]
    if args.publicar:
        escolhidas.append("publicar")
    comandos = {
        **ETAPAS,
        "finetune": [*ETAPAS["finetune"], "--epocas", str(args.epocas), "--batch", str(args.batch)],
        "publicar": ["scripts/publicar.py"],
    }

    inicio_total = time.perf_counter()
    for etapa in escolhidas:
        conferir_antes(etapa)
        print(f"\n=== {etapa} ===", flush=True)
        inicio = time.perf_counter()
        resultado = subprocess.run([sys.executable, *comandos[etapa]], cwd=RAIZ)
        if resultado.returncode:
            sys.exit(f"Etapa '{etapa}' falhou. Corrija e continue com: --de {etapa}")
        print(f"--- {etapa}: {time.perf_counter() - inicio:.0f} s", flush=True)
    print(f"\nPipeline concluído em {(time.perf_counter() - inicio_total) / 60:.1f} min.")


if __name__ == "__main__":
    main()
