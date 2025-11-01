from __future__ import annotations

import argparse
import json
from typing import Optional

from .integrations.invoice_link import send_request, dispatch


def main():
    ap = argparse.ArgumentParser(description="Send a request to Server/invoice_server and optionally dispatch it.")
    ap.add_argument("--app-id", type=str, default=None, help="Target app id registered in invoice_server")
    ap.add_argument("--payload", type=str, default='{}', help="JSON payload string")
    ap.add_argument("--dispatch", action="store_true", help="Dispatch immediately after enqueuing")
    args = ap.parse_args()

    try:
        payload = json.loads(args.payload)
    except Exception as e:
        raise SystemExit(f"Invalid JSON for --payload: {e}")

    sig = send_request(payload, app_id=args.app_id)
    print(f"signature={sig}")
    if args.dispatch:
        ok = dispatch(sig)
        print(f"dispatched={ok}")


if __name__ == "__main__":
    main()
