import argparse
import csv
import json
import os
from pathlib import Path
from typing import List

import mlx.optimizers as opt
from mlx_lm import generate, load
from mlx_lm.tuner import TrainingArgs, linear_to_lora_layers, train

SYSTEM_PROMPT = "너는 충북대학교 학칙 Q&A 어시스턴트다. 학칙에 근거해 정확·간결하게 한국어로 답한다."


def ensure_csv(csv_path: Path) -> None:
    """Create a dummy CSV with three rows if it does not exist."""
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = [
        {
            "Question": "학칙 제1조는 무엇인가?",
            "Answer": "학칙 제1조는 대학의 목적을 규정한다.",
        },
        {"Question": "수강신청 학점 상한은?", "Answer": "학기당 최대 18학점이다."},
        {"Question": "졸업 요건은?", "Answer": "전공과 교양 학점을 충족해야 한다."},
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Question", "Answer"])
        writer.writeheader()
        writer.writerows(rows)


def csv_to_jsonl(csv_path: Path, jsonl_path: Path) -> None:
    """Convert the Q/A CSV file into chat-style JSONL."""
    with open(csv_path, newline="", encoding="utf-8") as cf, open(
        jsonl_path, "w", encoding="utf-8"
    ) as jf:
        reader = csv.DictReader(cf)
        for row in reader:
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["Question"]},
                    {"role": "assistant", "content": row["Answer"]},
                ]
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Ko-Llama on CBNU ACR")
    parser.add_argument("--data_dir", default="data", help="Directory containing CSV")
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Optimizer learning rate"
    )
    parser.add_argument(
        "--adapter_file",
        default="adapters.safetensors",
        help="Output path for adapter weights",
    )
    args = parser.parse_args()

    base_model = os.getenv("BASE_MODEL", "beomi/Llama-3-Open-Ko-8B")
    data_dir = Path(args.data_dir)
    csv_path = data_dir / "chungbuk_univ_academic_rules_QA.csv"
    jsonl_path = data_dir / "train.jsonl"

    ensure_csv(csv_path)
    csv_to_jsonl(csv_path, jsonl_path)

    model, tokenizer = load(base_model)

    # Convert all linear layers to LoRA layers
    linear_to_lora_layers(
        model,
        num_layers=len(model.layers),
        config={
            "rank": args.lora_r,
            "scale": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
    )

    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.num_epochs
        * len(open(jsonl_path, "r", encoding="utf-8").readlines()),
        max_seq_length=args.max_seq_length,
        adapter_file=args.adapter_file,
    )

    optimizer = opt.AdamW(learning_rate=args.learning_rate)

    import types
    from mlx_lm.tuner.datasets import load_local_dataset

    config = types.SimpleNamespace(chat_feature="messages")
    train_ds, val_ds, _ = load_local_dataset(data_dir, tokenizer, config)

    if not val_ds:
        val_ds = train_ds

    train(model, optimizer, train_ds, val_ds, training_args)

    # Sample inference
    sample_question = "충북대학교의 졸업 요건은?"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample_question},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    answer = generate(model, tokenizer, prompt, max_tokens=128)
    print("Sample answer:", answer)


if __name__ == "__main__":
    main()
