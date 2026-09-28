"""Test the Laya decision model (convaiinnovations/laya) and measure its speed.

Usage:
    .venv/bin/python test_laya.py                 # benchmark on cpu and mps (if available)
    .venv/bin/python test_laya.py --devices cpu   # pick devices explicitly
    .venv/bin/python test_laya.py --runs 50       # more timed iterations
"""

import argparse
import json
import platform
import statistics
import time

import torch
from laya import Router

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts, upgrades",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}

# Labeled cases: (state, expected answers). Only the answers we are sure of are checked.
CASES = [
    (
        {"subject": "Duplicate charge on invoice #4411",
         "body": "We were billed twice for March. Please refund the duplicate today or we will cancel our plan."},
        {"department": "billing", "churn_risk": True, "refund_requested": True},
    ),
    (
        {"subject": "Production API returning 500s",
         "body": "Since 9am every call to /v1/orders fails with a 500 error. Our checkout is completely down."},
        {"department": "technical", "refund_requested": False},
    ),
    (
        {"subject": "Enterprise pricing",
         "body": "We are a 300-person company and would like a quote for the enterprise plan with SSO."},
        {"department": "sales", "churn_risk": False, "refund_requested": False},
    ),
    (
        {"subject": "Thanks!",
         "body": "Just wanted to say the new dashboard looks great. No action needed."},
        {"churn_risk": False, "refund_requested": False},
    ),
    (
        {"subject": "Doble cargo en mi tarjeta",
         "body": "Me cobraron dos veces este mes. Quiero un reembolso inmediato o cancelaré mi suscripción."},
        {"department": "billing", "churn_risk": True, "refund_requested": True},
    ),
]

ANSWER_KEY = {"choice": "choice", "score": "score", "noul": "noul"}


def answer_value(answer, qtype):
    value = answer[ANSWER_KEY[qtype]]
    # noul answers are a probability of "yes"; threshold at 0.5 for a boolean check
    return value >= 0.5 if qtype == "noul" else value


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device):
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    return out, (time.perf_counter() - t0) * 1000


def stats(ms):
    s = sorted(ms)
    p = lambda q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))]
    return {"mean": statistics.mean(s), "p50": p(0.50), "p95": p(0.95), "min": s[0], "max": s[-1]}


def fmt(st):
    return "  ".join(f"{k}={v:7.1f}ms" for k, v in st.items())


def run_accuracy(router):
    print("\n-- Accuracy check --")
    passed = total = 0
    for state, expected in CASES:
        result = router.predict(state, QUESTIONS)
        answers = result["answers"]
        line = []
        for key, want in expected.items():
            got = answer_value(answers[key], QUESTIONS[key]["type"])
            ok = got == want
            passed += ok
            total += 1
            line.append(f"{key}={got}{'' if ok else f' (expected {want})'}")
        urgency = answers["urgency"]["score"]
        print(f"  [{result['routing']['model']:>12}] {state['subject'][:34]:<34} "
              f"urgency={urgency:.2f}  " + "  ".join(line))
    print(f"  => {passed}/{total} checks passed")
    return passed, total


def run_benchmark(device, runs, warmup):
    print(f"\n==== device: {device} ====")
    router, load_ms = timed(lambda: Router(device=device, preload=True), device)
    print(f"Model load (preload all checkpoints): {load_ms:,.0f} ms")

    state, _ = CASES[0]
    _, first_ms = timed(lambda: router.predict(state, QUESTIONS), device)
    print(f"First predict (cold):                 {first_ms:,.1f} ms")

    for _ in range(warmup):
        router.predict(state, QUESTIONS)

    one_q = {"department": QUESTIONS["department"]}
    results = {}
    for label, qs in (("1 question", one_q), (f"{len(QUESTIONS)} questions", QUESTIONS)):
        ms = [timed(lambda: router.predict(state, qs), device)[1] for _ in range(runs)]
        st = stats(ms)
        results[label] = st
        print(f"Warm latency, {label:<12} ({runs} runs): {fmt(st)}")
        print(f"  throughput: {1000 / st['mean']:.1f} predicts/s, "
              f"{len(qs) * 1000 / st['mean']:.1f} questions/s")

    # Across all test cases (different inputs / languages -> may hit different checkpoints)
    ms = []
    for _ in range(max(1, runs // len(CASES))):
        for s, _ in CASES:
            ms.append(timed(lambda: router.predict(s, QUESTIONS), device)[1])
    results["mixed inputs"] = stats(ms)
    print(f"Warm latency, mixed inputs ({len(ms)} runs): {fmt(results['mixed inputs'])}")

    passed, total = run_accuracy(router)
    return {"device": device, "load_ms": load_ms, "first_predict_ms": first_ms,
            "latency": results, "accuracy": f"{passed}/{total}"}


def main():
    ap = argparse.ArgumentParser()
    default_devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) \
        + (["cuda"] if torch.cuda.is_available() else [])
    ap.add_argument("--devices", nargs="+", default=default_devices)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="laya_benchmark.json")
    args = ap.parse_args()

    import laya
    print(f"laya {laya.__version__} | torch {torch.__version__} | "
          f"{platform.platform()} | {platform.processor() or platform.machine()}")

    report = [run_benchmark(d, args.runs, args.warmup) for d in args.devices]

    print("\n==== Summary ====")
    print(f"{'device':<6} {'load':>9} {'cold':>9} {'1q p50':>9} {'4q p50':>9} {'4q p95':>9}  accuracy")
    for r in report:
        lat = r["latency"]
        q4 = lat[f"{len(QUESTIONS)} questions"]
        print(f"{r['device']:<6} {r['load_ms']:>7.0f}ms {r['first_predict_ms']:>7.1f}ms "
              f"{lat['1 question']['p50']:>7.1f}ms {q4['p50']:>7.1f}ms {q4['p95']:>7.1f}ms  {r['accuracy']}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
