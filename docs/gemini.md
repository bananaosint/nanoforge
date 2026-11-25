# Gemini assistance in Studio

Core training and manual feedback remain local and work without Gemini. The React Studio now has an optional asynchronous pretraining watchdog and a Gemini-assisted teacher workflow in **HITL Testing Studio → Edit & Teach**. No external requests run by default: the provider starts in `mock` mode and the watchdog starts disabled.

## Setup and credentials

On first Studio launch, **Set up Gemini assistance** prompts for a Gemini API key. Choose API, local CLI or mock, then save. **Use local mock instead** completes setup without a key. The header's **Gemini settings** button reopens these controls. If API mode is selected and the backend restarts without a configured key, the prompt appears again.

API keys entered in this password field are held only in backend memory, excluded from settings responses, SQLite, checkpoint payloads and browser storage. Alternatively, set `GEMINI_API_KEY` in the backend environment. **Forget session key** removes the in-memory override; an environment key remains available until removed from the environment. Already-running requests finish or time out with the client configuration they captured. Never commit a key or place it in CLI arguments/config files. Use the authenticated notebook launcher or a private deployment when accessing Studio remotely.

Nonsecret mode/model/command/timeout preferences persist in the existing SQLite settings table. **Test saved provider** sends synthetic metrics using the last saved settings; saving a key does not itself make an API request. Model IDs are editable because account/model availability changes. The initial API model is `gemini-3.8-flash`, checked against Google's model documentation during implementation.

For a separate nonsecret configuration file, copy `configs/gemini.example.json`, select a mode, and start the backend with `NANOFORGE_GEMINI_CONFIG=/absolute/path/gemini.json`. That file takes precedence over SQLite at startup; UI changes apply during the current process. Keys in this file are rejected. A CLI configuration, for example:

```json
{
  "mode": "cli",
  "model": "gemini-3.8-flash",
  "cli_command": ["/absolute/path/to/my-gemini-wrapper"],
  "timeout_seconds": 30,
  "setup_complete": true
}
```

The wrapper receives one JSON object on stdin with `task`, `instructions`, `response_schema` and `data`. It must return the requested JSON object on stdout (optionally inside a single Markdown JSON fence). CLI authentication belongs to that command; the adapter requires no API key in CLI mode. Commands use an argv array with `shell=False`. Timeout kills the process group on POSIX, and stderr is not returned to the UI. API mode uses the fixed Google `generateContent` HTTPS endpoint, an `x-goog-api-key` header and structured JSON output. Both paths enforce response-size and schema limits. Mock returns deterministic local test data, not a capable teacher.

## Pretraining watchdog

In **Train & Monitor**, enable **Checkpoint watchdog · Circuit Breaker** and choose 250, 500 or 1,000 steps (default 500). API/CLI requests run on a separate daemon worker with at most one pending request and no backlog. Training and shutdown never wait for Gemini. A slow request causes later scheduled reviews to be skipped, rather than queued indefinitely. Results are consumed between complete optimizer updates; stale results from another run, changed provider or more than two intervals ago are discarded. Late results after a run finishes are shown as advice without applying actions.

The payload contains structured telemetry only:

- Absolute model step, target step, current optimizer LR, raw `balance_loss`/`z_loss` and their current coefficients.
- Up to 100 recent training losses and pre-clipping, unscaled gradient norms; the last five validation evaluations and best validation loss.
- The last two distinct checkpoint routing snapshots, averaged over recent training tokens for each layer/expert. Routing units are **percent of tokens assigned to each expert**: top-k routing sums to `100 × k` percent per layer. Dense models send an empty routing list.

No prompts, datasets, sample generations or credentials are included in watchdog payloads. Loss coefficients are reported separately from raw MoE losses. Training telemetry is captured before evaluation/generation can overwrite module-level router statistics. Two-GPU pretraining gathers token-weighted statistics from the replicas.

The response must have exactly:

```json
{"action":"CONTINUE","reason":"Concise explanation based on the metrics.","confidence":0.9}
```

Actions are limited to `CONTINUE`, `ROLLBACK_AND_DAMPEN`, `BUMP_BALANCE_LOSS`, and `EARLY_STOP`. Extra fields, malformed/fenced prose, duplicate JSON keys, nonfinite numbers, invalid actions and out-of-range confidence are rejected. Provider errors/timeouts/schema failures become watchdog events and leave training unchanged. Existing local nonfinite-loss/gradient checks still stop invalid training independently.

Gemini chooses an action, never an arbitrary numeric hyperparameter. Local policy applies:

| Action | Numeric guard and effect |
| --- | --- |
| CONTINUE | No intervention. |
| ROLLBACK_AND_DAMPEN | A recovery checkpoint must exist, and recent train loss must rise by more than 25% or latest validation must exceed the best by 10%. Restore model weights, AdamW state, step and AMP scaler; multiply current LR by 0.5 (floor `1e-8`). Preserve the original target step. Limit three rollbacks per training invocation. |
| BUMP_BALANCE_LOSS | MoE only; at least two distinct snapshots must show sustained concentration and starved experts. Double the balance coefficient, capped at 1, with a small positive floor if it was zero. Update live router modules and saved model config together. |
| EARLY_STOP | Five recent evaluations must all exceed the best validation loss by more than 1%. Finish/save the current model as stopped. |

Minimum decision confidence is 0.8. The precise collapse guard compares each layer with its ideal top-k token share: a peak above `min(95%, 1.75 × ideal)` and a minimum below `0.25 × ideal`, sustained in both snapshots. These are conservative local rules, not promises that a Gemini confidence score is calibrated or that interventions improve a model.

Headless Studio/CLI training accepts these fields in its existing training JSON:

```json
{
  "watchdog_enabled": true,
  "watchdog_interval_steps": 500,
  "watchdog_min_confidence": 0.8,
  "watchdog_lr_factor": 0.5,
  "watchdog_balance_multiplier": 2.0
}
```

The UI exposes enable/interval; the remaining policy values are configurable in training JSON/API. If you change them there, those values govern the run. SFT/DPO do not invoke the watchdog. TPU/XLA workers and short ZeroGPU leases explicitly reject an enabled asynchronous watchdog; use the persistent CPU/CUDA/MPS/XPU Studio/CLI trainer for it.

Checkpoints include optional numeric watchdog history/events while keeping the ordinary Studio `.pt` format. Best validation recovery weights live under `checkpoints/.watchdog/`, protected from ordinary checkpoint deletion. Local filesystems use hard links; Drive/FUSE falls back to a copy. A successful new best replaces the previous recovery link for that run. Rollback constructs and validates replacement model/optimizer/scaler state before swapping the live objects; restore failure retains the current model. Normal checkpoint/rollback disk I/O still takes the training lock; only provider work is asynchronous. Larger recovery models may need additional temporary GPU memory.

When moving only a `.pt` to another machine, its hidden best-weight file is not bundled. Missing recovery references are discarded and the next pretraining evaluation establishes a local recovery baseline. Older checkpoints without watchdog metadata still load. The monitor displays metrics, expert loads and the latest 30 events.

## Gemini-assisted Edit & Teach

1. Generate a student completion in **HITL Testing Studio → Edit & Teach**, then click **Ask Gemini to edit this response**. You can also request a teacher target for the current prompt before generating a student completion.
2. The provider receives that prompt and student completion, not the model weights or dataset. A background job creates a draft containing `corrected_output`, `reason`, and `confidence`. Nothing is added to the training library yet.
3. Review/edit **Teacher target** and click **Approve & add to SFT**. Approval saves one ordinary SFT feedback row with the original student output, approved target, checkpoint and teacher provenance. Retrying/double-clicking approval cannot insert a duplicate row.
4. Use the existing **Train on feedback → Start SFT** controls to distill approved targets. This uses the same `format_prompt` delimiters, completion-only loss mask and portable checkpoint path as manual corrections. Teacher explanations/confidence are metadata, not training targets. Manual editing and DPO remain available.

Jobs and drafts persist in SQLite and can be selected from the lab history. A restart marks unfinished jobs as failed instead of resending prompts automatically. An API/CLI error creates no feedback and does not affect ongoing training. Long targets may be truncated to the model context by the existing SFT batching logic; character tokenizers still map unseen characters to their unknown token.

## Verification and limits

`backend/tests/test_gemini.py` covers strict JSON, deterministic mocks, fenced CLI responses, CLI timeout/failed exits, mocked HTTP/API contracts, key exclusion, file configuration, slow/failing providers during training, transactional checkpoint recovery, confidence/staleness/numeric guards, MoE routing, replica telemetry, teacher failure/restart, idempotent approval and SFT training. The UI was exercised locally from first-launch setup through mock watchdog pretraining, editing/approving a teacher draft and completed SFT.

Actual Google API calls and GPU-specific recovery/scaler paths were not exercised on the CPU development host. The adapter uses existing `httpx`/Pydantic dependencies; no Google SDK or credentials are needed for tests.

Google references: [Models](https://aistudio.google.com/docs/models), [Structured outputs](https://aistudio.google.com/docs/structured-output), [GenerateContent and responseFormat](https://aistudio.google.com/api-reference/generate-content).
