"""Matriz de confusão do BERTimbau na referência → reports/matriz_confusao.png

Uso: uv run --group train python scripts/figuras.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from project_nov26_p4_destilacao.rotular import CLASSES

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    m = np.array(json.loads((RAIZ / "reports/bertimbau.json").read_text())["matriz_confusao"])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.imshow(m, cmap="Blues")
    rotulos = [c.replace("_", " ") for c in CLASSES]
    ax.set_xticks(range(len(CLASSES)), rotulos, rotation=40, ha="right")
    ax.set_yticks(range(len(CLASSES)), rotulos)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if m[i, j]:
                cor = "white" if m[i, j] > m.max() / 2 else "black"
                ax.text(j, i, m[i, j], ha="center", va="center", color=cor)
    ax.set(
        xlabel="Previsto (BERTimbau)",
        ylabel="Referência",
        title="Matriz de confusão — 150 reclamações de referência",
    )
    fig.tight_layout()
    fig.savefig(RAIZ / "reports/matriz_confusao.png", dpi=120)


if __name__ == "__main__":
    main()
