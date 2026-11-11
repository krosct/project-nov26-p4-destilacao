"""Figuras dos relatórios.

Uso: uv run --group train python scripts/figuras.py
- reports/matriz_confusao.png: BERTimbau (modelo em uso) na referência.
- reports/epocas.png: treino curto (5 épocas) × longo (15), se os dois relatórios existirem.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from project_nov26_p4_destilacao.rotular import CLASSES

RAIZ = Path(__file__).resolve().parents[1]


def virgula(texto: str) -> str:
    return texto.replace(".", ",")  # decimal em pt-BR


# Paleta: duas séries validadas para daltonismo; textos em tons neutros, nunca na cor da série
AZUL, LARANJA = "#2a78d6", "#eb6834"
TEXTO, TEXTO_2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def matriz_confusao() -> None:
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
        title=f"Matriz de confusão — {m.sum()} reclamações de referência",
    )
    fig.tight_layout()
    fig.savefig(RAIZ / "reports/matriz_confusao.png", dpi=120)
    plt.close(fig)


def _estilo(ax) -> None:
    # Eixos e grade discretos: o destaque fica com os dados
    ax.set_facecolor(FUNDO)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TEXTO_2, labelsize=9)
    ax.xaxis.label.set_color(TEXTO_2)
    ax.yaxis.label.set_color(TEXTO_2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: virgula(f"{v:.2f}")))


def _titulo(ax, titulo: str, subtitulo: str) -> None:
    # Título + subtítulo explicativo, alinhados à esquerda
    ax.set_title(titulo, loc="left", fontsize=11, color=TEXTO, pad=22)
    ax.text(0, 1.02, subtitulo, transform=ax.transAxes, fontsize=9, color=TEXTO_2)


def _rotulo_fim(ax, x, y, texto) -> None:
    # Rótulo direto no fim da linha, em tom de texto (a cor da linha já identifica a série)
    ax.annotate(
        texto,
        (x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=TEXTO,
    )


def epocas() -> None:
    curto = RAIZ / "reports/bertimbau_5ep.json"
    longo = RAIZ / "reports/bertimbau_15ep.json"
    if not (curto.exists() and longo.exists()):
        return
    c, lg = json.loads(curto.read_text()), json.loads(longo.read_text())
    if "historico" not in c or "historico" not in lg:
        return
    hl = lg["historico"]
    ep = [h["epoca"] for h in hl]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=FUNDO)

    # 1. Overfitting: treino continua caindo, validação volta a subir
    treino = [h["loss_treino"] for h in hl]
    valid = [h["loss_validacao"] for h in hl]
    ax1.plot(ep, treino, color=AZUL, linewidth=2, label="treino")
    ax1.plot(ep, valid, color=LARANJA, linewidth=2, label="validação")
    melhor = ep[int(np.argmin(valid))]
    ax1.axvline(melhor, color=TEXTO_2, linewidth=1, linestyle=":")
    ax1.annotate(
        f"menor loss de validação\n(época {melhor})",
        (melhor, max(valid) * 0.92),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        color=TEXTO_2,
    )
    _rotulo_fim(ax1, ep[-1], treino[-1], "treino")
    _rotulo_fim(ax1, ep[-1], valid[-1], "validação")
    ax1.set(xlabel="época", ylabel="loss (entropia cruzada)", xlim=(0.5, ep[-1] + 4))
    _titulo(
        ax1,
        "O treino decora e a loss de validação sobe",
        "Loss por época no treino de 15 épocas",
    )
    ax1.legend(frameon=False, fontsize=9, labelcolor=TEXTO_2, loc="upper center")
    _estilo(ax1)

    # 2. Sem ganho: F1 de validação estabiliza cedo; na referência os dois empatam
    for dados, cor, nome in ((lg, AZUL, "15 épocas"), (c, LARANJA, "5 épocas")):
        h = dados["historico"]
        x, y = [p["epoca"] for p in h], [p["f1_validacao"] for p in h]
        ax2.plot(x, y, color=cor, linewidth=2, marker="o", markersize=4, label=nome)
        esc = dados["melhor_epoca"]
        ax2.plot(
            esc,
            y[x.index(esc)],
            marker="o",
            markersize=9,
            color=cor,
            markeredgecolor=FUNDO,
            markeredgewidth=2,
        )
    ax2.set(
        xlabel="época", ylabel="F1 macro na validação", xlim=(0.5, ep[-1] + 4), ylim=(0.78, 0.87)
    )
    _titulo(
        ax2,
        "O F1 de validação oscila; o ganho no teste é pequeno",
        # O resultado que importa é o do teste final, na referência rotulada à parte
        virgula(
            f"Teste final (450 reclamações de referência): "
            f"5 épocas = {c['f1_macro']:.3f} · 15 épocas = {lg['f1_macro']:.3f}"
        ),
    )
    # Legenda com as duas séries e o significado do ponto maior
    escolhida = Line2D(
        [],
        [],
        marker="o",
        markersize=9,
        color=TEXTO_2,
        linestyle="none",
        markeredgecolor=FUNDO,
        markeredgewidth=2,
        label="época escolhida",
    )
    ax2.legend(
        handles=[*ax2.get_lines()[::2], escolhida],
        frameon=False,
        fontsize=9,
        labelcolor=TEXTO_2,
        loc="lower right",
    )
    _estilo(ax2)

    fig.tight_layout()
    fig.savefig(RAIZ / "reports/epocas.png", dpi=120, facecolor=FUNDO)
    plt.close(fig)


def main() -> None:
    matriz_confusao()
    epocas()


if __name__ == "__main__":
    main()
