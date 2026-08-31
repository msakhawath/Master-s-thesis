# =============================================================
# test_ollama_connection.py — explicit model health check
# =============================================================
# Examples:
#   python test_ollama_connection.py
#   python test_ollama_connection.py --model gemma3:4b
#   python test_ollama_connection.py --base-url http://localhost:11434/v1
# =============================================================

import argparse
import time
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="Check Ollama and one required model.")
    parser.add_argument(
        "--model",
        default="gemma3:1b",
        help="Exact Ollama model required by the experiment (default: gemma3:1b).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible Ollama endpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Testing Ollama connection...")
    print(f"Endpoint: {args.base_url}")
    print("-" * 60)

    # IMPORTANT:
    # OpenAI(...) is only the PYTHON CLIENT LIBRARY here.
    # Because base_url points to localhost (or your university Ollama server),
    # the request is sent to OLLAMA, not to OpenAI's cloud.
    # Ollama implements a compatible subset of the OpenAI API schema.
    client = OpenAI(base_url=args.base_url.rstrip("/") + "/", api_key="ollama")

    try:
        models = client.models.list()
    except Exception as e:
        print(f"✗ Could not connect to Ollama: {e}")
        print("Start Ollama with: ollama serve")
        raise SystemExit(1)

    model_ids = [m.id for m in models.data]
    print("✓ Ollama is reachable.")
    print("\nInstalled models:")
    if model_ids:
        for model_id in model_ids:
            print(f"   - {model_id}")
    else:
        print("   (none)")

    if args.model not in model_ids:
        print(f"\n✗ Required model '{args.model}' is not installed.")
        print(f"Install it with: ollama pull {args.model}")
        raise SystemExit(2)

    print(f"\nTesting the REQUIRED model: {args.model}")
    try:
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=5,
            temperature=0.0,
            seed=12345,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        reply = (resp.choices[0].message.content or "").strip()
        print(f"✓ Model generated a response in {elapsed_ms:.1f} ms: {reply!r}")
        print("\nHealth check passed.")
    except Exception as e:
        print(f"✗ Model '{args.model}' exists but failed to generate: {e}")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
