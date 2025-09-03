#!/usr/bin/env python3
import argparse
import json
import sys

import numpy as np
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OpenCLIP embedding API")
    parser.add_argument("--url", required=True, help="Base URL, e.g., http://localhost:8000 or public ngrok URL")
    parser.add_argument("--key", default=None, help="Optional Bearer API key")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("texts", nargs="*", default=["a photo of a dog", "a red car"])
    args = parser.parse_args()

    endpoint = args.url.rstrip("/") + "/embed"
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "0"
    }
    if args.key:
        headers["Authorization"] = f"Bearer {args.key}"
    payload = {"texts": args.texts, "normalize": True}

    try:
        resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=args.timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    embs = np.array(data.get("embeddings", []), dtype=np.float32)
    if embs.ndim != 2:
        print("Invalid response shape", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({
        "num_texts": len(args.texts),
        "embeddings_shape": list(embs.shape),
    }))


if __name__ == "__main__":
    main()


