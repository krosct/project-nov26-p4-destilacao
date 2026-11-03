"""Tokens por reclamação na rotulação em lotes, a partir do cache de respostas do LLM.

Uso: uv run python scripts/tokens_lotes.py   (rodar logo após scripts/rotular_llm.py)
Inclui as novas tentativas: é o custo real, não o ideal.
"""

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
N_RECLAMACOES = 150 + 3000


def main() -> None:
    respostas = [json.loads(p.read_text()) for p in (RAIZ / ".cache/llm").glob("*.json")]
    entrada = sum(r["tokens_entrada"] or 0 for r in respostas)
    saida = sum(r["tokens_saida"] or 0 for r in respostas)
    resultado = {
        "requisicoes": len(respostas),
        "reclamacoes": N_RECLAMACOES,
        "entrada_por_reclamacao": round(entrada / N_RECLAMACOES, 1),
        "saida_por_reclamacao": round(saida / N_RECLAMACOES, 1),
    }
    (RAIZ / "reports/tokens_lotes.json").write_text(json.dumps(resultado, indent=2))
    print(resultado)


if __name__ == "__main__":
    main()
