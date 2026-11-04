"""Rotulação de reclamações com o LLM professor, em lotes.

Cada requisição leva até `TAMANHO_LOTE` reclamações numeradas de 1 a N e devolve
uma lista JSON. A resposta é validada: rótulos dentro da taxonomia e exatamente
os mesmos números enviados (nem faltando, nem sobrando). Lotes que falham após
as novas tentativas são rotulados um a um.
"""

import json
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Literal, get_args

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, TypeAdapter, ValidationError

from project_nov26_p4_destilacao.llm import MODELO_PADRAO, chamar_llm

# Fonte única das classes: o tipo valida a resposta do LLM e a lista é usada nas
# métricas e no fine-tuning. A ordem define os ids do modelo publicado.
Rotulo = Literal[
    "nao_recebido",
    "pedido_incompleto",
    "produto_errado",
    "defeito_ou_qualidade",
    "atraso_ou_entrega",
    "atendimento_ou_cobranca",
    "outro",
]
CLASSES: list[str] = list(get_args(Rotulo))
TAMANHO_LOTE = 20
# A mesma taxonomia guia a rotulação da referência e o prompt do LLM
TAXONOMIA = (Path(__file__).resolve().parents[2] / "data/taxonomia.md").read_text()


class Classificacao(BaseModel):
    n: int
    rotulo: Rotulo  # fora das 7 classes → ValidationError


# Valida a lista inteira devolvida pelo LLM (criado uma vez, reaproveitado em toda chamada)
LISTA = TypeAdapter(list[Classificacao])

INSTRUCOES = f"""Você classifica reclamações de clientes de e-commerce segundo a taxonomia abaixo.

{TAXONOMIA}

Responda SOMENTE com uma lista JSON, um objeto por reclamação, no formato
[{{"n": 1, "rotulo": "nao_recebido"}}, ...], usando exatamente os números recebidos."""


@dataclass
class Uso:
    """Requisições e tokens gastos, incluindo novas tentativas e o plano B (custo real)."""

    requisicoes: int = 0
    tokens_entrada: int = 0
    tokens_saida: int = 0
    _trava: Lock = field(default_factory=Lock, repr=False)

    def somar(self, resposta: dict) -> None:
        with self._trava:  # vários lotes somam ao mesmo tempo (threads)
            self.requisicoes += 1
            self.tokens_entrada += resposta["tokens_entrada"] or 0
            self.tokens_saida += resposta["tokens_saida"] or 0


def _extrair_json(texto: str) -> str:
    # O LLM às vezes envolve a lista em ```json ... ``` ou em texto; fica só o [...]
    m = re.search(r"\[.*\]", texto, flags=re.DOTALL)
    return m.group(0) if m else texto


def _classificar(
    textos: list[str], modelo: str, uso: Uso | None, max_tentativas: int = 3
) -> list[str] | None:
    # Numeração 1..N: é o formato de lista que o LLM conhece melhor
    numerados = "\n".join(f"{i}. {t}" for i, t in enumerate(textos, start=1))
    mensagens: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": INSTRUCOES},
        {"role": "user", "content": numerados},
    ]
    esperado = set(range(1, len(textos) + 1))
    for _ in range(max_tentativas):
        r = chamar_llm(mensagens, modelo)
        if uso is not None:
            uso.somar(r)
        try:
            itens = LISTA.validate_json(_extrair_json(r["texto"]))
            recebidos = [i.n for i in itens]
            # set: nenhum número faltando ou inventado · len: nenhum repetido
            if set(recebidos) != esperado or len(recebidos) != len(esperado):
                raise ValueError(f"números esperados {sorted(esperado)}, recebidos {recebidos}")
            por_n = {i.n: i.rotulo for i in itens}
            return [por_n[i] for i in sorted(esperado)]  # volta à ordem dos textos
        except (ValidationError, ValueError, json.JSONDecodeError) as erro:
            # Devolve o erro ao LLM e pede a lista corrigida na próxima tentativa
            mensagens = mensagens + [
                {"role": "assistant", "content": r["texto"]},
                {
                    "role": "user",
                    "content": f"Resposta inválida: {erro}\nDevolva só a lista corrigida.",
                },
            ]
    return None


def _rotular_lote(lote: list[str], modelo: str, uso: Uso | None) -> Sequence[str | None]:
    resultado = _classificar(lote, modelo, uso)
    if resultado is None:  # plano B: um a um; o que ainda falhar fica None
        resultado = [(_classificar([t], modelo, uso) or [None])[0] for t in lote]
    return resultado


def rotular(
    textos: list[str],
    modelo: str = MODELO_PADRAO,
    tamanho_lote: int = TAMANHO_LOTE,
    paralelo: int = 5,
    uso: Uso | None = None,
) -> list[str | None]:
    """Rotula em lotes; `paralelo` lotes simultâneos (a ordem da saída é preservada).

    Se `uso` for passado, acumula nele as requisições e os tokens gastos.
    """
    lotes = [textos[i : i + tamanho_lote] for i in range(0, len(textos), tamanho_lote)]
    # map dispara os lotes já; o with só termina quando todos acabam
    with ThreadPoolExecutor(max_workers=paralelo) as executor:
        resultados = executor.map(lambda lote: _rotular_lote(lote, modelo, uso), lotes)
    # Achata [[lote 1], [lote 2], ...] em um rótulo por texto, na ordem original
    return [rotulo for lote in resultados for rotulo in lote]
