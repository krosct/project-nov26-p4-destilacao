"""Fine-tuning do BERTimbau nos rótulos do LLM (aluno), avaliado na referência.

Uso: uv run --group train python scripts/finetune.py [--epocas N] [--max-steps N] [--amostra N]
Treino: 90% dos rótulos do LLM · validação (escolha da época): 10% · teste: referência.
Grava reports/bertimbau_{N}ep.json (com a curva por época). Se salvar na pasta padrão,
grava também reports/bertimbau.json, o relatório do modelo em uso (lido por figuras.py e
publicar.py); experimentos com --saida diferente não mexem nele.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from project_nov26_p4_destilacao.avaliar import matriz, metricas
from project_nov26_p4_destilacao.rotular import CLASSES

RAIZ = Path(__file__).resolve().parents[1]
BASE = "neuralmind/bert-base-portuguese-cased"
SAIDA = RAIZ / "models/bertimbau-triagem"
SEED = 42
# Tradução classe ↔ número; fica salva no modelo publicado (a ordem de CLASSES é contrato)
id2label = dict(enumerate(CLASSES))
label2id = {c: i for i, c in id2label.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=-1, help="para testes rápidos")
    p.add_argument("--amostra", type=int, default=None, help="usa só N exemplos (teste)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epocas", type=int, default=5)
    p.add_argument("--saida", default=str(SAIDA))
    args = p.parse_args()

    # Destilação: o aluno treina nos rótulos do professor (LLM), não em rótulos humanos
    dados = pd.read_csv(RAIZ / "data/rotulos_llm.csv")
    if args.amostra:
        dados = dados.sample(args.amostra, random_state=SEED)
    # stratify: mesma proporção de cada classe no treino e na validação
    treino, val = train_test_split(
        dados, test_size=0.1, stratify=dados["rotulo"], random_state=SEED
    )
    ref = pd.read_csv(RAIZ / "data/referencia.csv")  # teste: nunca usado no treino

    tok = AutoTokenizer.from_pretrained(BASE)  # texto → ids do vocabulário do BERTimbau

    def preparar(df: pd.DataFrame) -> Dataset:
        # Tabela text/label → acrescenta input_ids, attention_mask e token_type_ids
        ds = Dataset.from_dict(
            {"text": df["texto"].tolist(), "label": [label2id[r] for r in df["rotulo"]]}
        )
        # Sem padding aqui: o DataCollator completa lote a lote
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

    # Corpo pré-treinado + cabeça nova de 7 saídas (pesos aleatórios, aprendidos no treino)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(CLASSES), id2label=id2label, label2id=label2id
    )

    def compute_metrics(pred):
        return {"f1_macro": f1_score(pred.label_ids, pred.predictions.argmax(-1), average="macro")}

    gpu = torch.cuda.is_available()
    passos = (
        args.max_steps if args.max_steps > 0 else math.ceil(len(treino) / args.batch) * args.epocas
    )
    completo = args.max_steps < 0  # treino de verdade (não teste rápido com --max-steps)
    argumentos = TrainingArguments(
        output_dir=str(RAIZ / "checkpoints"),
        learning_rate=2e-5,  # passos pequenos para não apagar o que o BERTimbau já sabe
        num_train_epochs=args.epocas,
        max_steps=args.max_steps,  # > 0 substitui as épocas
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=32,  # avaliação não guarda gradientes: cabe lote maior
        warmup_steps=int(0.1 * passos),  # learning rate sobe aos poucos nos primeiros 10%
        weight_decay=0.01,  # regularização: puxa os pesos levemente para zero
        # Avalia e salva a cada época para escolher a melhor pelo F1 na validação
        eval_strategy="epoch" if completo else "no",
        save_strategy="epoch" if completo else "no",
        save_total_limit=1,
        load_best_model_at_end=completo,
        metric_for_best_model="f1_macro",
        fp16=gpu,  # meia precisão: metade da memória na GPU
        dataloader_num_workers=0,
        logging_steps=20,
        report_to="none",  # não envia métricas a serviços externos
        seed=SEED,
    )
    trainer = Trainer(
        model=modelo,
        args=argumentos,
        train_dataset=preparar(treino),
        eval_dataset=preparar(val),
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),  # completa com [PAD] até o maior do lote
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # Época do checkpoint escolhido (maior F1 na validação)
    melhor_epoca = next(
        (
            round(log["epoch"])
            for log in trainer.state.log_history
            if trainer.state.best_metric is not None
            and log.get("eval_f1_macro") == trainer.state.best_metric
        ),
        None,
    )
    # Curva por época: loss de treino (média dos registros da época) e métricas da validação
    historico = []
    for log in trainer.state.log_history:
        if "eval_f1_macro" in log:
            epoca = round(log["epoch"])
            perdas = [
                r["loss"]
                for r in trainer.state.log_history
                if "loss" in r and epoca - 1 < r["epoch"] <= epoca
            ]
            historico.append(
                {
                    "epoca": epoca,
                    "loss_treino": round(float(np.mean(perdas)), 4) if perdas else None,
                    "loss_validacao": round(log["eval_loss"], 4),
                    "f1_validacao": round(log["eval_f1_macro"], 4),
                }
            )

    # Avaliação final na referência (os tipos do Trainer.predict não incluem datasets.Dataset)
    saida_ref = trainer.predict(preparar(ref))  # pyright: ignore[reportArgumentType]
    # Maior das 7 pontuações → id → nome da classe
    previsto = [id2label[i] for i in np.argmax(saida_ref.predictions, axis=-1)]
    resumo = {
        "base": BASE,
        "epocas": args.epocas,
        "melhor_epoca": melhor_epoca,
        "n_treino": len(treino),
        "n_validacao": len(val),
        "f1_macro_validacao": round(trainer.evaluate()["eval_f1_macro"], 3) if len(val) else None,
        **metricas(ref["rotulo"].tolist(), previsto),
        "matriz_confusao": matriz(ref["rotulo"].tolist(), previsto),
        "pico_memoria_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if gpu else None,
        "historico": historico,
    }
    if completo:
        # Modelo e tokenizador juntos: quem usa o modelo precisa dos dois
        trainer.save_model(args.saida)
        tok.save_pretrained(args.saida)
        relatorio = json.dumps(resumo, indent=2)
        (RAIZ / f"reports/bertimbau_{args.epocas}ep.json").write_text(relatorio)
        if Path(args.saida).resolve() == SAIDA:  # só o modelo em uso vira o relatório principal
            (RAIZ / "reports/bertimbau.json").write_text(relatorio)
    print({k: v for k, v in resumo.items() if k not in ("matriz_confusao", "historico")})


if __name__ == "__main__":
    main()
