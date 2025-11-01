from __future__ import annotations

import argparse
from .agent_protocol import wants_access_to_screen_and_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--why", type=str, default="KroxAI requires UI access to perceive and act.")
    args = ap.parse_args()
    out = wants_access_to_screen_and_input(rationale=args.why)
    print(out.to_json())


if __name__ == "__main__":
    main()
