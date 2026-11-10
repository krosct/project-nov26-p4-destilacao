"""Amostra reclamações do Olist: 450 para referência + 6.000 para treino, sem sobreposição.

Uso: uv run python scripts/amostrar.py
Rótulos já existentes em data/referencia.csv são preservados (pelo id); itens novos saem
com `rotulo` vazio, para rotular seguindo data/taxonomia.md.
Com a mesma SEED, aumentar N_REFERENCIA ou N_TREINO só acrescenta itens ao fim do sorteio:
os primeiros continuam os mesmos.
"""

from pathlib import Path

import pandas as pd

from project_nov26_p4_destilacao.normalizacao import sem_quase_duplicatas

RAIZ = Path(__file__).resolve().parents[1]
SEED = 42
N_REFERENCIA, N_TREINO = 450, 6000


def main() -> None:
    df = pd.read_csv(RAIZ / "data/raw/olist_order_reviews_dataset.csv")
    # Reclamação = nota até 3 e com comentário
    df = df[(df["review_score"] <= 3) & df["review_comment_message"].notna()].copy()
    # Quebras de linha e espaços repetidos viram um espaço só
    df["texto"] = df["review_comment_message"].str.replace(r"\s+", " ", regex=True).str.strip()
    # Descarta textos curtos demais para classificar e duplicados
    df = df[df["texto"].str.len() >= 20].drop_duplicates("texto")
    print(f"reclamações elegíveis: {len(df)}")

    # Um único sorteio, cortado em duas partes: referência e treino não se sobrepõem
    amostra = df.sample(N_REFERENCIA + N_TREINO, random_state=SEED)
    colunas = {"review_id": "id", "texto": "texto", "review_score": "nota"}
    amostra = amostra[list(colunas)].rename(columns=colunas)
    referencia, treino = amostra.iloc[:N_REFERENCIA], amostra.iloc[N_REFERENCIA:]
    assert not set(referencia["id"]) & set(treino["id"])  # sem vazamento para o teste
    # Textos que só diferem em caixa ou pontuação ("NÃO RECEBI O PRODUTO") também vazam: saem do
    # treino. A referência não muda, então os rótulos e as métricas continuam comparáveis.
    antes = len(treino)
    treino = sem_quase_duplicatas(treino, referencia)
    print(f"quase duplicatas removidas do treino: {antes - len(treino)}")

    # Reaproveita os rótulos que já existem; itens novos ficam com rotulo vazio
    arquivo_ref = RAIZ / "data/referencia.csv"
    rotulados = {}
    if arquivo_ref.exists():
        antiga = pd.read_csv(arquivo_ref, keep_default_na=False)
        rotulados = dict(zip(antiga["id"], antiga["rotulo"], strict=True))
    referencia = referencia.assign(rotulo=referencia["id"].map(rotulados).fillna(""))
    referencia.to_csv(arquivo_ref, index=False)
    treino.to_csv(RAIZ / "data/amostra_treino.csv", index=False)
    faltam = int((referencia["rotulo"] == "").sum())
    print(f"referência: {len(referencia)} ({faltam} sem rótulo) · treino: {len(treino)}")


if __name__ == "__main__":
    main()
