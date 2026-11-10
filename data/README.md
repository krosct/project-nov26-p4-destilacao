# Dados

## Fonte e licença

Avaliações do [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (arquivo `olist_order_reviews_dataset.csv`), licença **CC BY-NC-SA 4.0**. Os arquivos derivados deste diretório (trechos de comentários e rótulos) são distribuídos sob a mesma licença, com atribuição à Olist, para uso **não comercial**.

O CSV bruto não é versionado. Para baixar:

```bash
mkdir -p data/raw
curl -L -o data/raw/olist.zip https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce
unzip -o data/raw/olist.zip olist_order_reviews_dataset.csv -d data/raw && rm data/raw/olist.zip
```

## Arquivos

| Arquivo | Conteúdo | Gerado por |
|---|---|---|
| `taxonomia.md` | 7 classes, definições, exemplos e regras de desempate | — |
| `referencia.csv` | 450 reclamações rotuladas segundo a taxonomia (**nunca usadas no treino**) | `scripts/amostrar.py` + rotulação |
| `amostra_treino.csv` | 5.971 reclamações para treino, sem rótulo (6.000 sorteadas menos 29 quase duplicatas da referência) | `scripts/amostrar.py` |
| `rotulos_llm.csv` | as 5.971 reclamações rotuladas pelo LLM professor | `scripts/rotular_llm.py` |

## Amostragem

Avaliações com nota ≤ 3 e comentário de pelo menos 20 caracteres, sem textos duplicados: 13.142 elegíveis. Sorteio com `random_state=42` de 6.450, divididas em 450 de referência e 6.000 de treino, sem sobreposição. Depois, `amostrar.py` retira do treino os textos que, normalizados (caixa, pontuação e espaços), também estão na referência (29 linhas, 14 textos distintos, como "NÃO RECEBI O PRODUTO"): o treino fica com 5.971 e a referência continua com 450. Com a mesma semente, aumentar a amostra só acrescenta itens ao fim do sorteio; `amostrar.py` preserva os rótulos de referência já existentes.

Distribuição das classes:

| Classe | Referência | Treino (rótulos do LLM) |
|---|---|---|
| `nao_recebido` | 115 | 1.542 |
| `produto_errado` | 81 | 1.117 |
| `defeito_ou_qualidade` | 81 | 887 |
| `pedido_incompleto` | 67 | 1.011 |
| `atraso_ou_entrega` | 41 | 495 |
| `atendimento_ou_cobranca` | 35 | 408 |
| `outro` | 30 | 511 |

Os comentários vêm como foram escritos pelos clientes. A Olist substituiu nomes de lojas por nomes de casas de *Game of Thrones* ("stark", "lannister", "targaryen"), o que aparece nos textos.
