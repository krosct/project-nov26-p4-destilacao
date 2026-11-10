import csv
from pathlib import Path

import pytest

from project_nov26_p4_destilacao import rotular
from project_nov26_p4_destilacao.avaliar import matriz, metricas
from project_nov26_p4_destilacao.rotular import CLASSES

RAIZ = Path(__file__).resolve().parents[1]


def resposta(texto):
    return {"texto": texto, "latencia_s": 0.1, "tokens_entrada": 10, "tokens_saida": 5}


def test_lote_valido(monkeypatch):
    monkeypatch.setattr(
        rotular,
        "chamar_llm",
        lambda m, modelo: resposta(
            '[{"n": 2, "rotulo": "outro"}, {"n": 1, "rotulo": "nao_recebido"}]'
        ),
    )
    assert rotular.rotular(["não chegou", "ok"], tamanho_lote=2) == ["nao_recebido", "outro"]


def test_lote_com_numero_faltando_tenta_de_novo(monkeypatch):
    respostas = iter(
        [
            '[{"n": 1, "rotulo": "nao_recebido"}]',  # faltou o 2
            '```json\n[{"n": 1, "rotulo": "nao_recebido"}, '
            '{"n": 2, "rotulo": "produto_errado"}]\n```',
        ]
    )
    recebidas = []

    def llm_falso(mensagens, modelo):
        recebidas.append(mensagens)
        return resposta(next(respostas))

    monkeypatch.setattr(rotular, "chamar_llm", llm_falso)
    assert rotular.rotular(["a", "b"], tamanho_lote=2) == ["nao_recebido", "produto_errado"]
    assert "inválida" in recebidas[1][-1]["content"]


def test_uso_soma_tokens_de_todas_as_tentativas(monkeypatch):
    respostas = iter(
        [
            '[{"n": 1, "rotulo": "nao_recebido"}]',  # inválida: faltou o 2
            '[{"n": 1, "rotulo": "nao_recebido"}, {"n": 2, "rotulo": "outro"}]',
        ]
    )
    monkeypatch.setattr(rotular, "chamar_llm", lambda m, modelo: resposta(next(respostas)))
    uso = rotular.Uso()
    rotular.rotular(["a", "b"], tamanho_lote=2, uso=uso)
    assert (uso.requisicoes, uso.tokens_entrada, uso.tokens_saida) == (2, 20, 10)


def test_rotulo_fora_da_taxonomia_e_rejeitado(monkeypatch):
    monkeypatch.setattr(
        rotular, "chamar_llm", lambda m, modelo: resposta('[{"n": 1, "rotulo": "frete"}]')
    )
    assert rotular.rotular(["a"], tamanho_lote=1) == [None]


def test_metricas_caso_conhecido():
    verdade = ["nao_recebido", "nao_recebido", "outro", "produto_errado"]
    previsto = ["nao_recebido", "outro", "outro", "produto_errado"]
    m = metricas(verdade, previsto)
    assert m["acuracia"] == 0.75
    assert 0 < m["kappa"] < 1
    assert len(matriz(verdade, previsto)) == len(CLASSES)


def test_referencia_rotulada_e_separada_do_treino():
    ref = list(csv.DictReader(open(RAIZ / "data/referencia.csv")))
    assert len(ref) == 450 and all(r["rotulo"] in CLASSES for r in ref)
    caminho = RAIZ / "data/rotulos_llm.csv"
    if not caminho.exists():
        pytest.skip("rótulos do LLM ainda não gerados")
    treino = {r["id"] for r in csv.DictReader(open(caminho))}
    assert not treino & {r["id"] for r in ref}


def test_treino_nao_tem_quase_duplicatas_da_referencia():
    import pandas as pd

    from project_nov26_p4_destilacao.normalizacao import normalizar, sem_quase_duplicatas

    ref = pd.read_csv(RAIZ / "data/referencia.csv")
    treino = pd.read_csv(RAIZ / "data/rotulos_llm.csv")
    assert not set(normalizar(treino["texto"])) & set(normalizar(ref["texto"]))
    # a função remove "NÃO RECEBI O PRODUTO!" quando "Não recebi o produto" está na referência
    fake_ref = pd.DataFrame({"texto": ["Não recebi o produto"]})
    fake_tr = pd.DataFrame({"texto": ["NÃO RECEBI O PRODUTO!", "Chegou quebrado"]})
    assert sem_quase_duplicatas(fake_tr, fake_ref)["texto"].tolist() == ["Chegou quebrado"]
