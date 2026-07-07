#!/usr/bin/env python3
"""Сабмит воркфлоу в ComfyUI API, ожидание готовности, отчёт по выходному файлу."""
import json, sys, time, urllib.request, urllib.error

HOST = "http://127.0.0.1:10100"

def post_prompt(graph):
    data = json.dumps({"prompt": graph}).encode()
    req = urllib.request.Request(HOST + "/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def get_json(path):
    with urllib.request.urlopen(HOST + path, timeout=30) as r:
        return json.load(r)

def main():
    graph = json.load(open(sys.argv[1]))
    label = sys.argv[2] if len(sys.argv) > 2 else "smoke"
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 900
    t0 = time.time()
    try:
        resp = post_prompt(graph)
    except urllib.error.HTTPError as e:
        print(f"[{label}] POST /prompt FAIL {e.code}: {e.read().decode()[:800]}")
        sys.exit(2)
    pid = resp.get("prompt_id")
    print(f"[{label}] prompt_id={pid} принят, жду выполнения (timeout {timeout}s)...")
    while time.time() - t0 < timeout:
        time.sleep(3)
        hist = get_json(f"/history/{pid}")
        if pid in hist:
            h = hist[pid]
            status = h.get("status", {})
            if status.get("completed") or status.get("status_str") == "success":
                imgs = []
                for node_id, out in h.get("outputs", {}).items():
                    for im in out.get("images", []):
                        imgs.append(im)
                dt = time.time() - t0
                print(f"[{label}] ✅ ГОТОВО за {dt:.1f}s. Изображения: {imgs}")
                return 0
            if status.get("status_str") == "error":
                print(f"[{label}] ❌ ОШИБКА выполнения:")
                for m in status.get("messages", []):
                    print("   ", m)
                return 3
    print(f"[{label}] ⏱ таймаут {timeout}s")
    return 4

if __name__ == "__main__":
    sys.exit(main())
