"""Cliente de LLM independente de provedor (API compatível com a OpenAI).

Respostas ficam em cache em disco: reexecutar a avaliação não gasta requisições
e o resultado é reproduzível.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

load_dotenv()

BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
MODELO_PADRAO = os.environ.get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
CACHE = Path(os.environ.get("LLM_CACHE_DIR", ".cache/llm"))


def _client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])


def chamar_llm(
    mensagens: list[dict],
    modelo: str = MODELO_PADRAO,
    max_tentativas: int = 6,
    **kwargs,
) -> dict:
    """Retorna {texto, latencia_s, tokens_entrada, tokens_saida, cache}."""
    chave = hashlib.sha256(json.dumps([mensagens, modelo, kwargs], sort_keys=True).encode())
    arquivo = CACHE / f"{chave.hexdigest()}.json"
    if arquivo.exists():
        return {**json.loads(arquivo.read_text()), "cache": True}

    for tentativa in range(max_tentativas):
        try:
            inicio = time.perf_counter()
            resp = _client().chat.completions.create(
                model=modelo, messages=mensagens, temperature=0, **kwargs
            )
            texto = resp.choices[0].message.content
            if not texto:
                raise ValueError("resposta vazia")
            resultado = {
                "texto": texto,
                "latencia_s": round(time.perf_counter() - inicio, 3),
                "tokens_entrada": resp.usage.prompt_tokens if resp.usage else None,
                "tokens_saida": resp.usage.completion_tokens if resp.usage else None,
            }
            break
        except (RateLimitError, APIConnectionError, APIStatusError, ValueError):
            if tentativa == max_tentativas - 1:
                raise
            time.sleep(min(2 ** (tentativa + 1), 60))  # 2, 4, 8, 16, 32 s

    CACHE.mkdir(parents=True, exist_ok=True)
    arquivo.write_text(json.dumps(resultado, ensure_ascii=False))
    return {**resultado, "cache": False}
