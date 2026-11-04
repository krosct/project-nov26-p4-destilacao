"""Métricas de classificação contra o conjunto de referência."""

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from project_nov26_p4_destilacao.rotular import CLASSES


def metricas(verdade: list[str], previsto: list[str]) -> dict:
    # labels=CLASSES: as 7 classes entram na conta, mesmo as que não forem previstas
    return {
        "acuracia": round(accuracy_score(verdade, previsto), 3),
        # macro: cada classe pesa igual, então as classes pequenas contam tanto quanto as grandes
        "f1_macro": round(
            float(f1_score(verdade, previsto, labels=CLASSES, average="macro", zero_division=0)), 3
        ),
        # kappa: concordância descontando o acerto esperado por acaso
        "kappa": round(cohen_kappa_score(verdade, previsto, labels=CLASSES), 3),
    }


def matriz(verdade: list[str], previsto: list[str]) -> list[list[int]]:
    # Linhas = verdade, colunas = previsto, na ordem de CLASSES; tolist() para salvar em JSON
    return confusion_matrix(verdade, previsto, labels=CLASSES).tolist()
