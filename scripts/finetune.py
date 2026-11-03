"""Fine-tuning do BERTimbau nos rótulos do LLM (aluno), avaliado na referência.

Uso: uv run --group train python scripts/finetune.py [--epocas N] [--max-steps N] [--amostra N]
Treino: 90% dos rótulos do LLM · validação (escolha da época): 10% · teste: referência.
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

    dados = pd.read_csv(RAIZ / "data/rotulos_llm.csv")
    if args.amostra:
        dados = dados.sample(args.amostra, random_state=SEED)
    treino, val = train_test_split(
        dados, test_size=0.1, stratify=dados["rotulo"], random_state=SEED
    )
    ref = pd.read_csv(RAIZ / "data/referencia.csv")

    tok = AutoTokenizer.from_pretrained(BASE)

    def preparar(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict(
            {"text": df["texto"].tolist(), "label": [label2id[r] for r in df["rotulo"]]}
        )
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

    modelo = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(CLASSES), id2label=id2label, label2id=label2id
    )

    def compute_metrics(pred):
        return {"f1_macro": f1_score(pred.label_ids, pred.predictions.argmax(-1), average="macro")}

    gpu = torch.cuda.is_available()
    passos = (
        args.max_steps if args.max_steps > 0 else math.ceil(len(treino) / args.batch) * args.epocas
    )
    argumentos = TrainingArguments(
        output_dir=str(RAIZ / "checkpoints"),
        learning_rate=2e-5,
        num_train_epochs=args.epocas,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=32,
        warmup_steps=int(0.1 * passos),  # 10% dos passos
        weight_decay=0.01,
        eval_strategy="epoch" if args.max_steps < 0 else "no",
        save_strategy="epoch" if args.max_steps < 0 else "no",
        save_total_limit=1,
        load_best_model_at_end=args.max_steps < 0,
        metric_for_best_model="f1_macro",
        fp16=gpu,
        dataloader_num_workers=0,
        logging_steps=20,
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=modelo,
        args=argumentos,
        train_dataset=preparar(treino),
        eval_dataset=preparar(val),
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    saida_ref = trainer.predict(preparar(ref.rename(columns={"rotulo": "rotulo"})))
    previsto = [id2label[i] for i in np.argmax(saida_ref.predictions, axis=-1)]
    resumo = {
        "base": BASE,
        "epocas": args.epocas,
        "n_treino": len(treino),
        "n_validacao": len(val),
        "f1_macro_validacao": round(trainer.evaluate()["eval_f1_macro"], 3) if len(val) else None,
        **metricas(ref["rotulo"].tolist(), previsto),
        "matriz_confusao": matriz(ref["rotulo"].tolist(), previsto),
        "pico_memoria_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if gpu else None,
    }
    if args.max_steps < 0:
        trainer.save_model(args.saida)
        tok.save_pretrained(args.saida)
        (RAIZ / f"reports/bertimbau_{args.epocas}ep.json").write_text(json.dumps(resumo, indent=2))
    print({k: v for k, v in resumo.items() if k != "matriz_confusao"})


if __name__ == "__main__":
    main()
