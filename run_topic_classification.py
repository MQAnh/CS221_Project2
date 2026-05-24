#!/usr/bin/env python
# coding=utf-8
"""
Train PhoBERT for Vietnamese topic classification from CSV files.

Expected CSV columns:
    title,link,label

Only "title" is used as input text. "label" must be integer class id.
Example label values: 0,1,2,...,9
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# MELoRA is a custom PEFT class in the MELoRA repo.
# Keep this import optional so normal LoRA can still run.
try:
    from peft import MELoraConfig
except Exception:
    MELoraConfig = None


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    train_file: str = field(metadata={"help": "Path to train CSV file."})
    validation_file: str = field(metadata={"help": "Path to validation CSV file."})
    test_file: Optional[str] = field(default=None, metadata={"help": "Path to test CSV file."})

    text_column: str = field(default="title", metadata={"help": "Input text column."})
    label_column: str = field(default="label", metadata={"help": "Label column."})

    max_seq_length: int = field(default=128)
    pad_to_max_length: bool = field(default=False)
    overwrite_cache: bool = field(default=False)

    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_predict_samples: Optional[int] = field(default=None)

    replace_underscore: bool = field(
        default=False,
        metadata={
            "help": (
                "Set True if your text is NOT intended for PhoBERT-style word segmentation. "
                "For PhoBERT, usually keep False because Vietnamese word-segmented text uses underscores."
            )
        },
    )

    def __post_init__(self):
        for path in [self.train_file, self.validation_file, self.test_file]:
            if path is not None:
                ext = path.split(".")[-1]
                if ext != "csv":
                    raise ValueError("This script currently expects CSV files only.")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="vinai/phobert-base",
        metadata={"help": "PhoBERT checkpoint name or local path."},
    )
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)

    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "PhoBERT usually works safer with the slow tokenizer."},
    )
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=False)

    # PEFT arguments
    use_peft: bool = field(default=True)
    mode: str = field(default="base", metadata={"help": "base = LoRA, me = MELoRA if available."})
    lora_path: Optional[str] = field(default=None)
    rank: int = field(default=8)
    l_num: Optional[int] = field(default=None)
    lora_alpha: int = field(default=16)
    target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["query", "value"],
        metadata={"help": "For PhoBERT/RoBERTa, common choices are query value."},
    )
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_task_type: str = field(default="SEQ_CLS")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    lora_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params

        if param.requires_grad:
            trainable_params += num_params
            if "lora_" in name:
                lora_params += num_params

    print(
        f"lora params: {lora_params:,d} || "
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory {training_args.output_dir} already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )

    set_seed(training_args.seed)

    # 1. Load CSV dataset
    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
    }
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    raw_datasets = load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )

    # 2. Validate columns
    for split_name, dataset in raw_datasets.items():
        cols = dataset.column_names
        if data_args.text_column not in cols:
            raise ValueError(f"Split {split_name} does not contain text column '{data_args.text_column}'. Found: {cols}")
        if split_name != "test" and data_args.label_column not in cols:
            raise ValueError(f"Split {split_name} does not contain label column '{data_args.label_column}'. Found: {cols}")

    # 3. Infer labels from train split
    label_list = raw_datasets["train"].unique(data_args.label_column)
    label_list = [int(x) for x in label_list]
    label_list = sorted(label_list)

    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: str(label) for label, i in label_to_id.items()}
    num_labels = len(label_list)

    print(f"Detected labels: {label_list}")
    print(f"num_labels = {num_labels}")

    # 4. Load PhoBERT
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=num_labels,
        label2id={str(k): v for k, v in label_to_id.items()},
        id2label=id_to_label,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # 5. Add LoRA/MELoRA
    if model_args.use_peft:
        if "me" in model_args.mode:
            if MELoraConfig is None:
                raise ImportError("MELoraConfig is not available in your installed peft package.")
            r = [model_args.rank] * (model_args.l_num or 1)
            alpha = [model_args.lora_alpha] * (model_args.l_num or 1)
            print("*** Using MELoRA ***")
            peft_config = MELoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=model_args.target_modules,
                lora_dropout=model_args.lora_dropout,
                bias=model_args.lora_bias,
                mode=model_args.mode,
                task_type=model_args.lora_task_type,
            )
        elif "base" in model_args.mode:
            print("*** Using LoRA ***")
            peft_config = LoraConfig(
                r=model_args.rank,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.target_modules,
                lora_dropout=model_args.lora_dropout,
                bias=model_args.lora_bias,
                task_type=model_args.lora_task_type,
            )
        else:
            raise ValueError(f"Unknown mode: {model_args.mode}")

        model = get_peft_model(model, peft_config)
        print_trainable_parameters(model)

        if model_args.lora_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_args.lora_path)

    # 6. Tokenize
    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        texts = examples[data_args.text_column]

        if data_args.replace_underscore:
            texts = [str(x).replace("_", " ") for x in texts]
        else:
            texts = [str(x) for x in texts]

        result = tokenizer(
            texts,
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )

        if data_args.label_column in examples:
            result["labels"] = [label_to_id[int(x)] for x in examples[data_args.label_column]]

        return result

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing CSV data",
        )

    train_dataset = None
    eval_dataset = None
    predict_dataset = None

    if training_args.do_train:
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    if training_args.do_eval:
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    if training_args.do_predict:
        if "test" not in tokenized_datasets:
            raise ValueError("--do_predict requires --test_file")
        predict_dataset = tokenized_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(min(len(predict_dataset), data_args.max_predict_samples)))

    # 7. Metrics: accuracy + macro-F1
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)

        acc = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
        macro_f1 = metric_f1.compute(
            predictions=preds,
            references=p.label_ids,
            average="macro",
        )["f1"]

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
        }

    # 8. Data collator
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Save only PEFT adapter weights when using PEFT
    if model_args.use_peft:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    # 9. Train
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 10. Eval
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 11. Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")
        output = trainer.predict(predict_dataset, metric_key_prefix="predict")
        logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
        pred_ids = np.argmax(logits, axis=1)

        # Convert internal class ids back to your original label ids.
        pred_labels = [id_to_label[int(i)] for i in pred_ids]

        os.makedirs(training_args.output_dir, exist_ok=True)
        output_predict_file = os.path.join(training_args.output_dir, "predict_results_topic.txt")

        if trainer.is_world_process_zero():
            with open(output_predict_file, "w", encoding="utf-8") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(pred_labels):
                    writer.write(f"{index}\t{item}\n")

            print(f"Saved predictions to: {output_predict_file}")


if __name__ == "__main__":
    main()
