# Laya AI Demo: Accuracy and Speed Test

A small harness that runs [Laya](https://huggingface.co/convaiinnovations/laya), the open-weight decision model from Convai Innovations, on local hardware. It checks answers against labelled support emails and measures latency on CPU and Apple Silicon GPU (MPS).

Laya is a 421M-parameter bidirectional encoder (ModernBERT-based). It does not generate text. You give it a *state* (an email, a ticket, JSON) and typed *questions*, and it returns typed answers in a single forward pass:

| Question type | Returns | Example |
|---|---|---|
| `choice` | one label from `criteria` | department: `billing` |
| `score` | a number on the `criteria` scale (0 = first level) | urgency: `1.94` |
| `noul` | probability of "yes" (0–1) | churn_risk: `0.74` |

## Quick start

Requires Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/pip install laya
.venv/bin/python test_laya.py
```

The first run downloads the checkpoints from Hugging Face, which took about 80 s here. After that, everything runs offline.

Options:

```bash
.venv/bin/python test_laya.py --devices cpu mps   # default: cpu + any available GPU (mps/cuda)
.venv/bin/python test_laya.py --runs 50 --warmup 10
.venv/bin/python test_laya.py --out results.json
```

## What the script does

For each device, [test_laya.py](test_laya.py):

1. **Load time:** builds `laya.Router(preload=True)`, which loads both the English and multilingual checkpoints.
2. **Cold call:** times the first `predict()` after loading.
3. **Warm latency:** after warmup, runs 30 timed calls each for:
   - 1 question (`department`)
   - 4 questions (`department`, `urgency`, `churn_risk`, `refund_requested`)
   - mixed inputs (all 5 test emails, so both checkpoints get used)
4. **Accuracy:** runs 5 labelled emails (4 English, 1 Spanish) and compares 13 expected answers. `noul` probabilities count as "yes" at ≥ 0.5.

The GPU is synchronized before and after each timed call, so the numbers are wall-clock time for the full call. Results are also written to [laya_benchmark.json](laya_benchmark.json).

## Results

**Environment:** Apple M3 · macOS 26.5 · Python 3.12.2 · laya 0.3.21 · torch 2.14.0 · run on 2026-09-28. Checkpoints were already cached.

### Latency

| | CPU | M3 GPU (MPS) |
|---|---:|---:|
| Load both checkpoints | 7,623 ms | 4,744 ms |
| First predict (cold) | 570 ms | 401 ms |
| **1 question: p50** | 135.1 ms | **51.3 ms** |
| 1 question: p95 | 154.2 ms | 52.8 ms |
| **4 questions: p50** | 348.0 ms | **140.0 ms** |
| 4 questions: p95 | 569.1 ms | 143.5 ms |
| Mixed inputs: p50 / p95 | 319.9 / 419.3 ms | 128.9 / 144.7 ms |
| Throughput (4-question calls) | ~10 questions/s | ~29 questions/s |

### Accuracy

**12 / 13 checks passed**, with identical results on CPU and MPS.

| Email | Checkpoint | Department | Urgency | Churn risk | Refund requested |
|---|---|---|---:|---|---|
| Duplicate charge on invoice | english | billing ✅ | 1.49 | yes ✅ (0.74) | yes ✅ (0.81) |
| Production API returning 500s | english | technical ✅ | 1.95 | – | no ✅ (0.07) |
| Enterprise pricing | english | sales ✅ | 1.13 | no ✅ | no ✅ |
| "Thanks!" (no action needed) | english | – | 0.30 | no ✅ (0.07) | no ✅ (0.03) |
| Doble cargo en mi tarjeta (Spanish) | multilingual | billing ✅ | 1.94 | **no ❌ (0.08)** | yes ✅ (0.99) |

Urgency is on a 0–2 scale (not urgent → soon → critical), and the order of the emails matches intuition: the outage and the Spanish billing complaint score highest, the thank-you note lowest.

## Insights

- **Use the GPU on Apple Silicon.** MPS is about 2.6× faster than CPU for a single question (51 vs 135 ms), with much tighter tail latency: the 4-question p95 is 144 ms on MPS against 569 ms on CPU.
- **Slower than the published figure.** Convai quotes about 33 ms per question on an Nvidia T4. An M3 over MPS reaches about 51 ms, which is still fast enough for inline routing and triage.
- **Extra questions cost almost the full price.** Going from 1 to 4 questions raised latency about 2.7×, not the much smaller increase the published "10 questions batched in 72 ms on T4" suggests. Keep the question set tight when latency matters.
- **Warm up before serving.** The first call after loading is 3–8× slower than steady state. Preload and send a dummy request at startup.
- **Language routing works.** The Spanish email was sent to the multilingual checkpoint automatically, and it got the department and refund request right.
- **The multilingual checkpoint missed a clear churn signal.** "…o cancelaré mi suscripción" scored only 0.08 for churn risk, while the English equivalent scored 0.74. Validate non-English `noul` questions on your own data before relying on them.
- **Confidence values are partly uncalibrated.** On load, laya 0.3.21 warns that the checkpoint ships invalid temperatures for some entries (`choice:11+`) and clamps them to 0.5. Don't hardcode confidence thresholds from one run; recheck after upgrading the package or checkpoint.

## Caveats

- The accuracy set is 5 hand-written emails and 13 checks. It is a smoke test, not an evaluation. Use a labelled sample of your real traffic for that.
- Latencies were measured on a laptop with other apps running. Expect some variance between runs, especially on CPU.

## Files

| File | Purpose |
|---|---|
| [test_laya.py](test_laya.py) | Accuracy and latency test harness |
| [laya_benchmark.json](laya_benchmark.json) | Raw results from the last run (per device: load, cold, latency stats, accuracy) |

## References

- Model card: [convaiinnovations/laya on Hugging Face](https://huggingface.co/convaiinnovations/laya)
- [Laya AI Model: How It Works, Run It Locally, and Evaluate It](https://huggingface.co/blog/sora-2/laya-ai-model-how-it-works-run-it-locally-and-eval)
- Project site: [laya-ai.com](https://laya-ai.com/)
