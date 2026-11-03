"""Baseline: TF-IDF + regressão logística treinada nos rótulos do LLM, avaliada na referência.

Uso: uv run python scripts/baseline.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from project_nov26_p4_destilacao.avaliar import matriz, metricas

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    treino = pd.read_csv(RAIZ / "data/rotulos_llm.csv")
    ref = pd.read_csv(RAIZ / "data/referencia.csv")
    modelo = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, strip_accents="unicode", sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ).fit(treino["texto"], treino["rotulo"])
    previsto = modelo.predict(ref["texto"]).tolist()
    resumo = {
        **metricas(ref["rotulo"].tolist(), previsto),
        "matriz_confusao": matriz(ref["rotulo"].tolist(), previsto),
    }
    (RAIZ / "models").mkdir(exist_ok=True)
    joblib.dump(modelo, RAIZ / "models/tfidf.joblib")
    (RAIZ / "reports/baseline_tfidf.json").write_text(json.dumps(resumo, indent=2))
    print({k: v for k, v in resumo.items() if k != "matriz_confusao"})


if __name__ == "__main__":
    main()
