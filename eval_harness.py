"""Lightweight evaluation harness for KroxAI.

Features:
- Reads prompts from a JSONL file (default: kroxai/prompts.jsonl)
- Calls KroxAI server /chat (default: http://127.0.0.1:5000)
- Optional API key via env KROXAI_API_KEY
- Falls back to local kroxai.torch_chat when server is unavailable
- Writes results to JSONL with per-prompt latency and prints basic stats
"""
import os
import json
import time
import argparse

try:
    import requests  # type: ignore
except Exception:
    requests = None


def _post_chat(server_url: str, payload: dict, api_key: str | None, timeout: float = 20.0) -> tuple[bool, str, float]:
    t0 = time.time()
    if not requests:
        return False, "[requests missing]", time.time() - t0
    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        r = requests.post(server_url.rstrip("/") + "/chat", json=payload, headers=headers, timeout=timeout)
        took = time.time() - t0
        if r.ok:
            data = r.json()
            reply = data.get("reply") or data.get("answer") or str(data)
            return True, reply, took
        return False, f"[HTTP {r.status_code}] {r.text[:200]}", took
    except Exception as e:
        return False, f"[HTTP error] {e}", time.time() - t0


def _local_chat(prompt: str) -> tuple[str, float]:
    t0 = time.time()
    try:
        from .torch_chat import KroxAI
        agent = KroxAI()
        # Minimal call; generator may ignore decoding params
        out = agent.generate(prompt, temperature=0.9, top_p=0.95, max_new_tokens=160)
        return str(out), time.time() - t0
    except Exception as e:
        return f"[local error] {e}", time.time() - t0


def run_eval(server_url: str | None, input_path: str, out_path: str):
    api_key = os.environ.get("KROXAI_API_KEY")
    results = []
    with open(input_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("Skip invalid JSONL line")
                continue
            text = obj.get("text") or obj.get("prompt") or ""
            payload = {
                "text": text,
                "conversation_id": "eval",
                "temperature": 0.9,
                "top_p": 0.95,
                "max_new_tokens": 160,
            }

            if server_url:
                ok, reply, took = _post_chat(server_url, payload, api_key)
                if not ok:
                    # Try local fallback
                    local_reply, local_took = _local_chat(text)
                    reply = f"{reply}\n[fallback] {local_reply}"
                    took = took + local_took
            else:
                reply, took = _local_chat(text)

            row = {"text": text, "reply": reply, "latency_s": round(took, 3)}
            results.append(row)
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.flush()
            print(f"OK: {text[:40]}... -> {row['latency_s']}s")

    # Basic metrics
    latencies = [r.get("latency_s", 0.0) for r in results]
    n = len(latencies)
    if n:
        lat_sorted = sorted(latencies)
        p95_idx = max(0, min(n - 1, int(0.95 * n) - 1))
        p95 = lat_sorted[p95_idx]
        print(f"Evaluated {n} prompts. p95 latency: {p95:.2f}s")
    else:
        print("No prompts.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.environ.get("KROXAI_SERVER", "http://127.0.0.1:5000"), help="KroxAI server base URL or leave empty to use local shim")
    ap.add_argument("--prompts", dest="input_path", default="kroxai/prompts.jsonl", help="Path to prompts JSONL with {text: ...}")
    ap.add_argument("--out", dest="out_path", default="kroxai/eval_results.jsonl", help="Output path for results JSONL")
    args = ap.parse_args()
    server = args.server.strip() if args.server else None
    if server == "none" or server == "local":
        server = None
    run_eval(server, args.input_path, args.out_path)
