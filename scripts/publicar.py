"""Publica o modelo no Hugging Face Hub com model card gerado a partir dos relatórios.

Uso: HF_TOKEN=... uv run --group train python scripts/publicar.py
"""

import json
import os
from pathlib import Path

from huggingface_hub import HfApi

RAIZ = Path(__file__).resolve().parents[1]
REPO = "trebuchetly/bertimbau-triagem-reclamacoes"


def model_card() -> str:
    bert = json.loads((RAIZ / "reports/bertimbau.json").read_text())
    tfidf = json.loads((RAIZ / "reports/baseline_tfidf.json").read_text())
    llm = json.loads((RAIZ / "reports/professor.json").read_text())
    lat = json.loads((RAIZ / "reports/latencia_custo.json").read_text())
    custo = lat["custo_usd_por_1m_reclamacoes"]
    ms = lat["latencia_ms_p50"]
    return f"""---
language: pt
license: cc-by-nc-sa-4.0
base_model: neuralmind/bert-base-portuguese-cased
pipeline_tag: text-classification
tags:
- text-classification
- portuguese
- customer-complaints
- distillation
datasets:
- olistbr/brazilian-ecommerce
metrics:
- f1
- accuracy
widget:
- text: "Comprei 2 unidades e só veio uma na caixa."
- text: "Até hoje não recebi o produto, já passou do prazo."
- text: "O produto chegou quebrado e não funciona."
---

# BERTimbau — triagem de reclamações (PT-BR)

Classifica reclamações de clientes de e-commerce em 7 categorias de problema. É um modelo
**aluno**: foi treinado com rótulos gerados por um LLM grande (`{llm["modelo"]}`, o professor),
para substituí-lo em produção com uma fração do custo e da latência.

Código, dados e avaliação: [github.com/krosct/project-nov26-p4-destilacao](https://github.com/krosct/project-nov26-p4-destilacao)

## Classes

`nao_recebido` · `pedido_incompleto` · `produto_errado` · `defeito_ou_qualidade` ·
`atraso_ou_entrega` · `atendimento_ou_cobranca` · `outro`
(definições e regras de desempate em [`data/taxonomia.md`](https://github.com/krosct/project-nov26-p4-destilacao/blob/main/data/taxonomia.md))

## Resultados (conjunto de referência, 150 reclamações nunca usadas no treino)

| Modelo | F1 macro | Acurácia | Kappa | Latência p50 | Custo / 1M reclamações |
|---|---|---|---|---|---|
| LLM professor | {llm["f1_macro"]} | {llm["acuracia"]} | {llm["kappa"]} | {ms["llm_api"]:.0f} ms (API) | US$ {custo["llm_uma_por_requisicao"]} |
| **Este modelo** | **{bert["f1_macro"]}** | **{bert["acuracia"]}** | **{bert["kappa"]}** | **{ms["bertimbau_cpu"]} ms (CPU)** | **US$ {custo["bertimbau_cpu"]}** |
| TF-IDF + regressão logística | {tfidf["f1_macro"]} | {tfidf["acuracia"]} | {tfidf["kappa"]} | {ms["tfidf_cpu"]} ms (CPU) | — |

Premissas de custo (preço da API e da máquina) no repositório, em `reports/latencia_custo.json`.

## Uso

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{REPO}")
clf("Comprei 2 unidades e só veio uma na caixa.")
```

## Dados de treino

- {bert["n_treino"] + bert["n_validacao"]} reclamações (nota ≤ 3) do
  [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
  (CC BY-NC-SA 4.0), rotuladas pelo LLM professor; 10% separados para escolher a melhor época.
- Fine-tuning de `neuralmind/bert-base-portuguese-cased`: até 5 épocas (melhor checkpoint pela validação: época 4), lr 2e-5, lote 16,
  textos truncados em 128 tokens, fp16.

## Uso pretendido e limitações

- **Pretendido:** triagem/roteamento de reclamações em português para filas de atendimento,
  com revisão humana dos casos de baixa confiança.
- **Não use** para decisões automáticas sobre clientes (reembolso, bloqueio) sem revisão.
- Treinado em **e-commerce**: em outros domínios (ex.: delivery de comida) as classes de
  entrega e qualidade podem se comportar diferente; reavalie antes de usar.
- Herda erros e vieses do LLM professor (os rótulos de treino têm ruído).
- Uma classe por texto: reclamações com vários problemas recebem só o principal.
- Licença **não comercial** (herdada dos dados).
"""


def main() -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    pasta = RAIZ / "models/bertimbau-triagem"
    (pasta / "README.md").write_text(model_card())
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(pasta),
        repo_id=REPO,
        repo_type="model",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "README.md"],
        commit_message="Publica BERTimbau de triagem de reclamações",
    )
    print(f"https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
