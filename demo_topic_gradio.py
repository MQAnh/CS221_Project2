#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_PEFT_SRC = PROJECT_ROOT / "peft-0.5.0" / "src"
if LOCAL_PEFT_SRC.exists():
    sys.path.insert(0, str(LOCAL_PEFT_SRC))

gr = None
torch = None
PeftModel = None
AutoConfig = None
AutoModelForSequenceClassification = None
AutoTokenizer = None


# Keep label order and spelling aligned with the existing analysis scripts.
CLASS_NAMES = [
    "TECHNOLOGY",
    "TRAVEL",
    "EDUCATION",
    "ENTERTAINMENT",
    "SCIENCE",
    "BUSINESS",
    "LAW",
    "HEATH",
    "WORLD",
    "SPORT",
    "NEWS",
    "VEHICLE",
    "LIFE",
]

DEFAULT_MAX_LENGTH = 128
DEFAULT_ADAPTER_CANDIDATES = [
    PROJECT_ROOT / "topic_phobert_lora",
    PROJECT_ROOT.parent / "hihi" / "topic_phobert_lora",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio demo for PhoBERT LoRA topic classification.")
    parser.add_argument(
        "--adapter_path",
        default=os.environ.get("TOPIC_MODEL_PATH"),
        help="Path to the LoRA adapter directory. Defaults to ../hihi/topic_phobert_lora if present.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        default=None,
        help="Base model path/name. Defaults to base_model_name_or_path in adapter_config.json.",
    )
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", default=None, help="Defaults to cuda if available, otherwise cpu.")
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_libraries():
    global gr
    global torch
    global PeftModel
    global AutoConfig
    global AutoModelForSequenceClassification
    global AutoTokenizer

    try:
        import gradio as imported_gradio
        import torch as imported_torch
        from peft import PeftModel as imported_peft_model
        from transformers import AutoConfig as imported_auto_config
        from transformers import AutoModelForSequenceClassification as imported_auto_model
        from transformers import AutoTokenizer as imported_auto_tokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install the project requirements first:\n"
            "  pip install -r requirements.txt\n"
            f"\nOriginal error: {exc}"
        ) from exc

    gr = imported_gradio
    torch = imported_torch
    PeftModel = imported_peft_model
    AutoConfig = imported_auto_config
    AutoModelForSequenceClassification = imported_auto_model
    AutoTokenizer = imported_auto_tokenizer


def resolve_adapter_path(adapter_path):
    if adapter_path:
        path = Path(adapter_path).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path

    for candidate in DEFAULT_ADAPTER_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    candidates = "\n".join(f"  - {candidate}" for candidate in DEFAULT_ADAPTER_CANDIDATES)
    raise FileNotFoundError(
        "Could not find an adapter directory. Pass --adapter_path or set TOPIC_MODEL_PATH.\n"
        f"Tried:\n{candidates}"
    )


def read_adapter_config(adapter_path):
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json at: {config_path}")

    with config_path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def validate_adapter_weights(adapter_path):
    safetensors_path = adapter_path / "adapter_model.safetensors"
    bin_path = adapter_path / "adapter_model.bin"

    if safetensors_path.exists() and safetensors_path.stat().st_size > 0:
        return

    if not bin_path.exists():
        raise FileNotFoundError(f"Missing adapter weights: {bin_path}")
    if bin_path.stat().st_size == 0:
        raise RuntimeError(f"Adapter weights file is empty: {bin_path}")

    try:
        torch.load(bin_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            "Could not read adapter_model.bin with torch.load. "
            "The file may be incomplete or corrupted. "
            f"Path: {bin_path}"
        ) from exc


def load_runtime(args):
    adapter_path = resolve_adapter_path(args.adapter_path)
    adapter_config = read_adapter_config(adapter_path)
    validate_adapter_weights(adapter_path)

    base_model_name = (
        args.base_model_name_or_path
        or adapter_config.get("base_model_name_or_path")
        or "vinai/phobert-base"
    )

    id2label = {idx: CLASS_NAMES[idx] for idx in range(len(CLASS_NAMES))}
    label2id = {label: idx for idx, label in id2label.items()}

    config = AutoConfig.from_pretrained(
        base_model_name,
        num_labels=len(CLASS_NAMES),
        id2label=id2label,
        label2id=label2id,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, config=config)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(args.device)
    model.eval()

    return {
        "adapter_path": adapter_path,
        "base_model_name": base_model_name,
        "tokenizer": tokenizer,
        "model": model,
        "device": args.device,
        "max_length": args.max_length,
    }


def make_predict_fn(runtime):
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    device = runtime["device"]
    max_length = runtime["max_length"]

    def predict(title):
        title = (title or "").strip()
        if not title:
            return "EMPTY_INPUT", {}

        inputs = tokenizer(
            title,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        best_id = max(range(len(probabilities)), key=probabilities.__getitem__)
        scores = {
            f"{idx} - {CLASS_NAMES[idx]}": float(probabilities[idx])
            for idx in range(len(CLASS_NAMES))
        }
        return f"{best_id} - {CLASS_NAMES[best_id]} ({probabilities[best_id]:.2%})", scores

    return predict


def build_demo(runtime):
    predict = make_predict_fn(runtime)

    with gr.Blocks(title="Topic classifier") as demo:
        gr.Markdown("# Topic title classifier")
        with gr.Row():
            title_input = gr.Textbox(label="Title", lines=3, placeholder="Enter a news title...")
        with gr.Row():
            predict_button = gr.Button("Predict", variant="primary")
            clear_button = gr.Button("Clear")
        with gr.Row():
            prediction_output = gr.Textbox(label="Predicted class")
            scores_output = gr.Label(label="Top classes", num_top_classes=5)

        predict_button.click(
            fn=predict,
            inputs=title_input,
            outputs=[prediction_output, scores_output],
        )
        title_input.submit(
            fn=predict,
            inputs=title_input,
            outputs=[prediction_output, scores_output],
        )
        clear_button.click(
            fn=lambda: ("", "", {}),
            inputs=[],
            outputs=[title_input, prediction_output, scores_output],
        )

        gr.Examples(
            examples=[
                ["viet_trinh lam vedette"],
                ["diem_chuan truong dh van_hoa tphcm dh hoa_sen"],
                ["vieri ronaldo gioi hon cristiano_ronaldo"],
            ],
            inputs=title_input,
        )

    return demo


def main():
    args = parse_args()
    load_libraries()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    runtime = load_runtime(args)
    print(f"Loaded adapter: {runtime['adapter_path']}")
    print(f"Base model: {runtime['base_model_name']}")
    print(f"Device: {runtime['device']}")

    demo = build_demo(runtime)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
