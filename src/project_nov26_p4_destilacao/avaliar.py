"""Métricas de classificação contra o conjunto de referência."""

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from project_nov26_p4_destilacao.rotular import CLASSES


def metricas(verdade: list[str], previsto: list[str]) -> dict:
    return {
        "acuracia": round(accuracy_score(verdade, previsto), 3),
        "f1_macro": round(
            f1_score(verdade, previsto, labels=CLASSES, average="macro", zero_division=0), 3
        ),
        "kappa": round(cohen_kappa_score(verdade, previsto, labels=CLASSES), 3),
    }


def matriz(verdade: list[str], previsto: list[str]) -> list[list[int]]:
    return confusion_matrix(verdade, previsto, labels=CLASSES).tolist()
