import os
from pathlib import Path

from dotenv import load_dotenv
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "paths.env")

MODEL = os.getenv("MODEL_PATH")
ADAPTER = Path(os.getenv("ADAPTER_PATH"))
SYSTEM_PROMPT_PATH = Path(os.getenv("SYSTEM_PROMPT_PATH"))

with open(SYSTEM_PROMPT_PATH) as f:
    SYSTEM_PROMPT = f.read().strip()


def main():
    print('Model loading...')
    model, tokenizer = load(MODEL, adapter_path=str(ADAPTER))

    user_msg = input("Enter a message: ")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    sampler = make_sampler(temp=0.8, top_p=0.95)

    print("Praneel: ", end="", flush=True)

    for response in stream_generate(
        model, tokenizer,
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False),
        max_tokens=50,
        sampler=sampler,
    ):
        text = response.text
        text = (
            text.replace("â\u0080\u0099", "'")
            .replace("â\u0080\u0093", "-")
            .replace("ð\u009f\u0098\u00ad", "😭")
            .replace("ð\u009f\u0092\u0080", "💀")
            .replace("ð\u009f\u00a4·â\u0080\u008dâ\u0099\u0082ï¸\u008f", "🤷\u200d♂️")
            .replace("ð\u009f\u0098\u0082", "😂")
        )
        print(text, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
