from __future__ import annotations

import argparse
import json
import sys

from llm import check_provider_readiness, provider_names


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate local provider configuration and credentials without "
            "sending an inference request."
        )
    )
    parser.add_argument("--provider", choices=provider_names(), default="gemini")
    parser.add_argument("--model")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        readiness = check_provider_readiness(args.provider, model=args.model)
    except ValueError as error:
        print(f"Provider check failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(readiness, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
