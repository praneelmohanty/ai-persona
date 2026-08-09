import os
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL = os.environ.get(
    "MODEL",
    os.path.expanduser("~/.lmstudio/models/mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"),
)
ADAPTER = BASE_DIR / "training" / "adapters_01"
SYSTEM_PROMPT_PATH = BASE_DIR / "training" / "system_prompt.txt"

with open(SYSTEM_PROMPT_PATH) as f:
    SYSTEM_PROMPT = f.read().strip()


def main():
    user_msg = input("Enter a message: ")

    model, tokenizer = load(MODEL, adapter_path=str(ADAPTER))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
    tokens = mx.array(tokens)
    prompt_len = len(tokens)

    sampler = make_sampler(temp=0.8, top_p=0.95)

    for _ in range(50):
        logits = model(tokens[None, :])
        logits = logits[:, -1, :] / 0.8
        token = sampler(logits)
        tokens = mx.concatenate([tokens, mx.array([token.item()])])
        if token.item() == tokenizer.eos_token_id:
            break
        mx.eval(tokens)

    generated = tokenizer.decode(tokens[prompt_len:].tolist(), clean_up_tokenization_spaces=False)

    generated = (
        generated.replace("â\u0080\u0099", "'")
        .replace("â\u0080\u0093", "-")
        .replace("ð\u009f\u0098\u00ad", "😭")
        .replace("ð\u009f\u0092\u0080", "💀")
        .replace("ð\u009f\u00a4·â\u0080\u008dâ\u0099\u0082ï¸\u008f", "🤷\u200d♂️")
        .replace("ð\u009f\u0098\u0082", "😂")
    )

    print("Praneel: ", end="")
    print(generated.strip()[:-10])


if __name__ == "__main__":
    main()
