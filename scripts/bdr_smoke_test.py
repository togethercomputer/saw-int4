"""
Quick sanity-check for a running BDR / SGLang server.

Start the server in any mode (BF16, INT4, or BDR) as shown in the README,
then run this script.  A coherent, chemistry-correct answer confirms that
the server is up and the BDR installation is working.

Usage:
    python scripts/bdr_smoke_test.py [--port 30000] [--model Qwen/Qwen3-4B-Thinking-2507]
"""

import argparse
from openai import OpenAI

# Sample question from GPQA (graduate-level, Google-proof Q&A benchmark).
# Expected answer: A.
GPQA_SAMPLE = """\
Answer the following multiple choice question.

Think step by step before answering.

The last line of your response should be exactly:
Answer: $LETTER

Where LETTER is one of A, B, C, or D.

You overhear two chemists talking to each other as they leave a synthetic organic chemistry lab. One asks the other, "So, how did it go?" The second chemist replies, "Not well - my compounds are on top of each other."

What is the second chemist most likely referring to?

A) The compounds they are working with have similar polarities.
B) The compounds they are working with are bonding to each other through non-covalent/van der Waals interactions.
C) The compounds they are working with have similar boiling points.
D) The compounds they are working with have similar optical rotations.
"""


def parse_args():
    p = argparse.ArgumentParser(description="BDR server smoke test")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    return p.parse_args()


def main():
    args = parse_args()
    base_url = f"http://0.0.0.0:{args.port}/v1"

    print(f"Server : {base_url}")
    print(f"Model  : {args.model}")
    print(f"\n--- Prompt (GPQA sample) ---\n{GPQA_SAMPLE}")

    client = OpenAI(api_key="EMPTY", base_url=base_url)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": GPQA_SAMPLE}],
        temperature=0.6,
        top_p=0.95,
        max_tokens=32768,
        stream=True,
    )

    print("--- Response ---")
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
