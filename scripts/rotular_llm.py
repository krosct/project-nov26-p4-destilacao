"""Rotula a referência (para medir o professor) e a amostra de treino com o LLM.

Uso: uv run python scripts/rotular_llm.py
Também grava reports/tokens_lotes.json (tokens por reclamação em lotes), lido por latencia.py.
"""

import json
from pathlib import Path

import pandas as pd

from project_nov26_p4_destilacao.avaliar import matriz, metricas
from project_nov26_p4_destilacao.llm import MODELO_PADRAO
from project_nov26_p4_destilacao.rotular import Uso, rotular

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    uso = Uso()  # soma requisições e tokens das duas rotulações

    # 1. Professor na referência: mede a qualidade do LLM contra os rótulos de referência
    ref = pd.read_csv(RAIZ / "data/referencia.csv")
    ref["rotulo_llm"] = rotular(ref["texto"].tolist(), uso=uso)
    ref[["id", "rotulo", "rotulo_llm"]].to_csv(
        RAIZ / "reports/professor_na_referencia.csv", index=False
    )
    validos = ref.dropna(subset=["rotulo_llm"])  # None = o LLM não conseguiu rotular
    resumo = {
        "modelo": MODELO_PADRAO,
        "sem_rotulo": int(ref["rotulo_llm"].isna().sum()),
        **metricas(validos["rotulo"].tolist(), validos["rotulo_llm"].tolist()),
        "matriz_confusao": matriz(validos["rotulo"].tolist(), validos["rotulo_llm"].tolist()),
    }
    (RAIZ / "reports/professor.json").write_text(json.dumps(resumo, indent=2))
    print("professor na referência:", {k: v for k, v in resumo.items() if k != "matriz_confusao"})

    # 2. Professor no treino: estes rótulos são o que o aluno vai imitar
    treino = pd.read_csv(RAIZ / "data/amostra_treino.csv")
    treino["rotulo"] = rotular(treino["texto"].tolist(), uso=uso)
    print("sem rótulo no treino:", int(treino["rotulo"].isna().sum()))
    treino.dropna(subset=["rotulo"])[["id", "texto", "rotulo"]].to_csv(
        RAIZ / "data/rotulos_llm.csv", index=False
    )
    print(treino["rotulo"].value_counts().to_dict())  # distribuição das classes

    # 3. Custo real da rotulação em lotes (inclui novas tentativas e plano B)
    n = len(ref) + len(treino)
    tokens = {
        "requisicoes": uso.requisicoes,
        "reclamacoes": n,
        "entrada_por_reclamacao": round(uso.tokens_entrada / n, 1),
        "saida_por_reclamacao": round(uso.tokens_saida / n, 1),
    }
    (RAIZ / "reports/tokens_lotes.json").write_text(json.dumps(tokens, indent=2))
    print("tokens em lotes:", tokens)


if __name__ == "__main__":
    main()
