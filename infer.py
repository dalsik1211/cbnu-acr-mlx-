import os

from mlx_lm import generate, load

SYSTEM_PROMPT = "너는 충북대학교 학칙 Q&A 어시스턴트다. 학칙에 근거해 정확·간결하게 한국어로 답한다."


def main() -> None:
    base_model = os.getenv("BASE_MODEL", "beomi/Llama-3-Open-Ko-8B")
    adapter_path = os.getenv("ADAPTER_PATH", "adapters.safetensors")
    model, tokenizer = load(base_model, adapter_path=adapter_path)

    print("Ctrl-D to exit.")
    while True:
        try:
            question = input("User: ")
        except (EOFError, KeyboardInterrupt):
            break
        if not question.strip():
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        answer = generate(model, tokenizer, prompt, max_tokens=256)
        print("Assistant:", answer)


if __name__ == "__main__":
    main()
