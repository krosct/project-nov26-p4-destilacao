# Taxonomia de reclamações

Uma classe por reclamação: **o problema principal**, isto é, o que motivou a reclamação.

| Classe | Definição | Exemplo |
|---|---|---|
| `nao_recebido` | O pedido (inteiro) ainda não chegou ou consta como entregue sem ter chegado. Inclui atraso em que o cliente **ainda não recebeu**. | "Não recebi o produto até o momento." |
| `pedido_incompleto` | Chegou só parte do pedido: faltam unidades, itens, peças, acessórios ou manual. | "Comprei 2 e veio apenas 1." |
| `produto_errado` | Chegou algo diferente do pedido ou do anunciado: modelo, cor, tamanho, quantidade por embalagem, réplica/não original, ficha técnica que não condiz. | "Pedi gotas e veio comprimido." |
| `defeito_ou_qualidade` | O produto chegou, é o certo, mas está quebrado, com defeito, não funciona ou tem qualidade/material abaixo do esperado. | "Veio quebrado." / "Material muito fino." |
| `atraso_ou_entrega` | O produto foi recebido, mas houve problema no processo de entrega: atraso, retirada obrigatória nos Correios, frete, rastreio, extravio já resolvido. | "Tive que buscar nos Correios." |
| `atendimento_ou_cobranca` | Atendimento, falta de resposta, cobrança indevida, estorno, nota fiscal, venda sem estoque ou cancelamento pela loja, segurança do pagamento. | "Venderam e depois disseram que não tinha estoque." |
| `outro` | Sem problema identificável, elogio, dúvida ou texto genérico demais. | "Ainda não usei, mas recomendo." |

## Regras de desempate

Quando a reclamação cita mais de um problema, vale o principal. Se não houver um principal claro, use a primeira classe desta ordem:

`nao_recebido` → `pedido_incompleto` → `produto_errado` → `defeito_ou_qualidade` → `atraso_ou_entrega` → `atendimento_ou_cobranca` → `outro`

Casos específicos:
- Não recebeu porque a loja cancelou ou não tinha estoque → `atendimento_ou_cobranca`.
- Recebeu parte do pedido e o resto não chegou → `pedido_incompleto` (não `nao_recebido`).
- Recebeu com atraso **e** com defeito → `defeito_ou_qualidade`.
- "Não condiz com a descrição/foto" → `produto_errado`.
