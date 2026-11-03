"""Rotulação de reclamações com o LLM professor, em lotes.

Cada requisição leva até `TAMANHO_LOTE` reclamações numeradas de 1 a N e devolve
uma lista JSON. A resposta é validada: rótulos dentro da taxonomia e exatamente
os mesmos números enviados (nem faltando, nem sobrando). Lotes que falham após
as novas tentativas são rotulados um a um.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, TypeAdapter, ValidationError

from project_nov26_p4_destilacao.llm import MODELO_PADRAO, chamar_llm

CLASSES = [
    "nao_recebido",
    "pedido_incompleto",
    "produto_errado",
    "defeito_ou_qualidade",
    "atraso_ou_entrega",
    "atendimento_ou_cobranca",
    "outro",
]
Rotulo = Literal[
    "nao_recebido",
    "pedido_incompleto",
    "produto_errado",
    "defeito_ou_qualidade",
    "atraso_ou_entrega",
    "atendimento_ou_cobranca",
    "outro",
]
TAMANHO_LOTE = 20
TAXONOMIA = (Path(__file__).resolve().parents[2] / "data/taxonomia.md").read_text()


class Classificacao(BaseModel):
    n: int
    rotulo: Rotulo


LISTA = TypeAdapter(list[Classificacao])

INSTRUCOES = f"""Você classifica reclamações de clientes de e-commerce segundo a taxonomia abaixo.

{TAXONOMIA}

Responda SOMENTE com uma lista JSON, um objeto por reclamação, no formato
[{{"n": 1, "rotulo": "nao_recebido"}}, ...], usando exatamente os números recebidos."""


def _extrair_json(texto: str) -> str:
    m = re.search(r"\[.*\]", texto, flags=re.DOTALL)
    return m.group(0) if m else texto


def _classificar(textos: list[str], modelo: str, max_tentativas: int = 3) -> list[str] | None:
    numerados = "\n".join(f"{i}. {t}" for i, t in enumerate(textos, start=1))
    mensagens = [
        {"role": "system", "content": INSTRUCOES},
        {"role": "user", "content": numerados},
    ]
    esperado = set(range(1, len(textos) + 1))
    for _ in range(max_tentativas):
        r = chamar_llm(mensagens, modelo)
        try:
            itens = LISTA.validate_json(_extrair_json(r["texto"]))
            recebidos = [i.n for i in itens]
            if set(recebidos) != esperado or len(recebidos) != len(esperado):
                raise ValueError(f"números esperados {sorted(esperado)}, recebidos {recebidos}")
            por_n = {i.n: i.rotulo for i in itens}
            return [por_n[i] for i in sorted(esperado)]
        except (ValidationError, ValueError, json.JSONDecodeError) as erro:
            mensagens = mensagens + [
                {"role": "assistant", "content": r["texto"]},
                {
                    "role": "user",
                    "content": f"Resposta inválida: {erro}\nDevolva só a lista corrigida.",
                },
            ]
    return None


def _rotular_lote(lote: list[str], modelo: str) -> list[str | None]:
    resultado = _classificar(lote, modelo)
    if resultado is None:  # plano B: um a um
        resultado = [(_classificar([t], modelo) or [None])[0] for t in lote]
    return resultado


def rotular(
    textos: list[str],
    modelo: str = MODELO_PADRAO,
    tamanho_lote: int = TAMANHO_LOTE,
    paralelo: int = 5,
) -> list[str | None]:
    """Rotula em lotes; `paralelo` lotes simultâneos (a ordem da saída é preservada)."""
    lotes = [textos[i : i + tamanho_lote] for i in range(0, len(textos), tamanho_lote)]
    with ThreadPoolExecutor(max_workers=paralelo) as executor:
        resultados = executor.map(lambda lote: _rotular_lote(lote, modelo), lotes)
    return [rotulo for lote in resultados for rotulo in lote]
