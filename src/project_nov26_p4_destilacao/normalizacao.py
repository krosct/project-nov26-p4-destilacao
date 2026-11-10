"""Normalização de texto para achar quase duplicatas (fonte única, usada em vários scripts)."""

import pandas as pd


def normalizar(s: pd.Series) -> pd.Series:
    """Minúsculas, sem pontuação e com espaços colapsados ("NÃO RECEBI!" vira "não recebi")."""
    return (
        s.str.lower()
        .str.replace(r"[^a-zà-ú0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def sem_quase_duplicatas(treino: pd.DataFrame, referencia: pd.DataFrame) -> pd.DataFrame:
    """Tira do treino os textos que, normalizados, também estão na referência (vazamento)."""
    return treino[~normalizar(treino["texto"]).isin(set(normalizar(referencia["texto"])))]
