"""Amostra reclamações do Olist: 150 para referência + 3.000 para treino, sem sobreposição.

Uso: uv run python scripts/amostrar.py
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SEED = 42
N_REFERENCIA, N_TREINO = 150, 3000


def main() -> None:
    df = pd.read_csv(RAIZ / "data/raw/olist_order_reviews_dataset.csv")
    df = df[(df["review_score"] <= 3) & df["review_comment_message"].notna()].copy()
    df["texto"] = df["review_comment_message"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["texto"].str.len() >= 20].drop_duplicates("texto")
    print(f"reclamações elegíveis: {len(df)}")

    amostra = df.sample(N_REFERENCIA + N_TREINO, random_state=SEED)
    colunas = {"review_id": "id", "texto": "texto", "review_score": "nota"}
    amostra = amostra[list(colunas)].rename(columns=colunas)
    referencia, treino = amostra.iloc[:N_REFERENCIA], amostra.iloc[N_REFERENCIA:]
    assert not set(referencia["id"]) & set(treino["id"])

    referencia.assign(rotulo="").to_csv(RAIZ / "data/referencia.csv", index=False)
    treino.to_csv(RAIZ / "data/amostra_treino.csv", index=False)
    print(f"referência: {len(referencia)} · treino: {len(treino)}")


if __name__ == "__main__":
    main()
