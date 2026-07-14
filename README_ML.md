# Ikkat ML — Fault-Tolerant Checkpoint System on the Ikkat DFS

This is the ML checkpoint extension built on top of the Ikkat distributed
file system (Project 1). It turns the same primary-backup, write-ahead-log,
prime-deduplication storage layer into a **content-addressed checkpoint
store** for distributed model training, with no changes to the replication
protocol itself.

If you haven't read it yet, `ML_EXTENSION_EXPLAINED.md` covers the
motivation and architecture in depth. This README covers: what changed in
this revision, how the pieces fit together, how to run everything, and the
baseline experiment results.

## The Problem This Solves

Training a model across multiple workers means each worker periodically
saves a *checkpoint* — a snapshot of every tensor in the model. Naively,
every checkpoint stores every tensor, even ones that haven't changed since
the last checkpoint. In real training, early layers and embedding tables
converge quickly and stop changing — so most of each checkpoint after the
first one is redundant.

The fix is the same one Git uses for file blobs: **content-addressed
storage**. Hash each tensor (SHA-256). If a tensor with that hash already
exists on the server, store nothing — just record the hash in the
checkpoint's metadata. If it's new, write it once. Identical tensors —
whether from the same worker's previous checkpoint or a different worker
entirely — are stored exactly once.

## Architecture Mapping

| Project 1 (prime dedup) | Project 2 (ML checkpoints) |
|---|---|
| Prime number (`uint64`) | Tensor blob (`[]byte`) |
| `IsPrime(n)` | `ml.HashTensor(t)` (SHA-256) |
| `FilterUniquePrimes` | Dedup check in `ServerHandle.SaveTensors` |
| `Close()` commits a write | `SaveCheckpointMeta()` commits a checkpoint |
| WAL `LogEntry{Op: "WRITE"}` | Same — tensors are written via the normal `Create`/`Open`/`Write`/`Close` RPCs and `"WRITE"` log entries |
| Chandy-Lamport snapshot | Extended to report per-worker training step (`WorkerSteps`) and tensor count (`TensorCount`), derived from disk at snapshot time |

## What Changed in This Revision

The previous drop of this extension had its files scattered in the repo
root with mismatched package names, which doesn't compile. This revision
fixes that and removes a chunk of dead code along the way. If you're
comparing against an earlier version, here's the diff:

**Files moved into their correct package directories** (Go requires every
`.go` file in a directory to declare the same package):

- `tensor` (no extension, `package ml_common`) → `ml_common/tensor.go`
- `checkpoint.go` (`package ml_common`) → `ml_common/checkpoint.go`
- `coordinator.go` (`package main`) → `ml_coordinator/coordinator.go`
- `worker.go` (`package main`) → `ml_worker/worker.go`

These three files were all sitting in the repo root alongside the
`package ikkat` files (`server_basics.go`, `client_basics.go`, etc.), which
is a hard compile error: *"found packages ikkat and ml_common in same
directory"*.

**`server_ml.go` was deleted.** It implemented a second, parallel path for
saving tensors — native methods on `*server` (`SaveTensors`,
`SaveCheckpointMeta`, etc.) using new `TENSOR_WRITE`/`TENSOR_META` write-ahead
log operation types. This path was **never actually used**: it was never
registered as a gRPC service, so nothing could call it over the network, and
`ml_worker`/`ml_coordinator` both go through `ServerHandle`
(`server_handle.go`) instead, which reuses the *existing*
`Create`/`Open`/`Write`/`Close`/`TestAuth` RPCs and the existing `"WRITE"`
log op. Keeping both was a source of duplicate type definitions
(`SaveTensorsRequest`/`SaveTensorsResponse` existed in two places) and dead
fields (`tensorStore`, `workerSteps` on the server struct, never read by
the path that's actually used).

**Removed along with `server_ml.go`:**
- `tensorStore map[string]string` and `workerSteps map[string]int` fields
  on the `server` struct (`server_basics.go`) and their initialization in
  `NewServer` (`connection.go`)
- The `TENSOR_WRITE` / `TENSOR_META` cases in `apply()`
  (`server_replication.go`) — replaced with a no-op case that just logs and
  skips, so old log files containing these entries from before this cleanup
  don't trigger "unknown op" warnings on replay
- The two `rebuildTensorStore()` calls (on recovery and on failover) —
  `rebuildTensorStore` lived only in `server_ml.go`

**`ml_common/types.go` (new file)** — `SaveTensorsRequest` and
`SaveTensorsResponse` now live here as the single definition, importable by
both `server_handle.go` and `ml_worker/worker.go`. `SaveTensorsResponse`
also gained two new fields, **`NewBytes`** and **`DedupedBytes`** — the
previous version only returned tensor *counts* (`NewCount`/`DedupeCount`),
which makes it impossible to report the dedup ratio in the unit that
actually matters for storage cost (bytes). `ServerHandle.SaveTensors` now
tracks `len(tensor.Data)` for every tensor and attributes it to one counter
or the other.

**`server_snapshots.go`** — `GlobalSnapshot.WorkerSteps` and `.TensorCount`
are still part of the snapshot (useful for the coordinator's training-state
view), but are now populated by scanning `output/latest_*.ptr` and
`output/tensors_*` files on disk at snapshot time, rather than reading the
removed `s.workerSteps`/`s.tensorStore` server fields. No new server state
is needed.

**`server_handle.go`** — `LoadTensors` and `LoadCheckpointMeta` previously
called `Open`+`Read` without a matching `Close`, leaking a file descriptor
entry on the server for every tensor restored. Both now call `Close` after
reading.

## Code Map

| File | Role |
|---|---|
| `ml_common/tensor.go` | `TensorBlob` type, `HashTensor` (SHA-256 content hash), `SimulateTensor` |
| `ml_common/checkpoint.go` | `CheckpointMeta` type, path helpers (`MetaPath`, `TensorPath`, `LatestMetaPath`), (de)serialization, `DeduplicationStats` |
| `ml_common/types.go` | `SaveTensorsRequest` / `SaveTensorsResponse` |
| `server_handle.go` | Client-side facade: `SaveTensors`, `SaveCheckpointMeta`, `LoadCheckpointMeta`, `LoadTensors` — all implemented via the existing file RPCs |
| `server_snapshots.go` | Chandy-Lamport snapshot, extended with `WorkerSteps`/`TensorCount` |
| `ml_worker/worker.go` | Simulated training worker: runs a tiny MLP, checkpoints every N steps, can restore on startup |
| `ml_coordinator/coordinator.go` | Heartbeat tracking, dead-worker detection, periodic snapshots, HTTP dashboard |
| `experiments_ml/` | Baseline experiments (this revision's addition — see below) |

## How to Run

### 1. Start a 3-node Ikkat DFS cluster

```bash
go run test_server_1/simple_server.go -id=1 -port=5001
go run test_server_2/bu1_server.go     -id=2 -port=5002
go run test_server_3/bu2_server.go     -id=3 -port=5003
```

Wait a few seconds for leader election to settle.

### 2. Run a training worker

```bash
go run ml_worker/worker.go -id=worker_0 -total=1
```

This runs 1000 simulated training steps, checkpointing every 100 steps. Each
checkpoint log line shows the dedup ratio, e.g.:

```
Worker worker_0: step=500: 4/7 tensors deduped (57.1% storage saved)
```

To simulate a crash and resumption:

```bash
go run ml_worker/worker.go -id=worker_0 -crash=350   # dies at step 350
go run ml_worker/worker.go -id=worker_0              # resumes from last checkpoint (step 300)
```

### 3. Run the coordinator (optional)

```bash
go run ml_coordinator/coordinator.go
```

Exposes a dashboard at `http://localhost:8080/status` showing each worker's
current step, dedup stats, and any detected stragglers/dead workers.

### 4. Run the baseline experiments

```bash
bash experiments_ml/run_all.sh
bash experiments_ml/collect_results.sh   # renders results into this README
```

This runs the two experiments described below against the running cluster
and writes CSVs to `experiments_ml/results/`.

---

## Baseline Experiments

### Experiment 1 — Baseline Checkpoint (Dedup + Timing)

A single worker (`experiments_ml/01_baseline_checkpoint/main.go`) runs the
same `SimpleModel` as `ml_worker` — a 7-tensor MLP where the embedding table
stabilizes by step ~200, `layer0.*` by ~400, `layer1.*` by ~600, and
`layer2.*` (the output layer) never stabilizes. For every checkpoint (every
100 steps by default, 1000 steps total → 10 checkpoints), it measures:

- `new_tensors` / `deduped_tensors` / `total_tensors` — count-based dedup
- `new_bytes` / `deduped_bytes` / `total_bytes` — **byte-based** dedup (the
  metric that actually corresponds to storage cost)
- `save_tensors_time_s` — time to hash, dedup-check, and write new tensors
- `save_meta_time_s` — time to commit the small checkpoint metadata file
- `checkpoint_total_time_s` — sum of the two

**What to expect:** the first checkpoint (step 100) should show
`dedup_pct_bytes ≈ 0%` (every tensor is new). By step 300–400, as
`embedding.weight` (the largest tensor at 256 KB) and `layer0.*` stabilize,
`dedup_pct_bytes` should jump sharply — this is the headline number for the
resume bullet ("60–70% storage reduction").

```bash
go run experiments_ml/01_baseline_checkpoint/main.go
go run experiments_ml/01_baseline_checkpoint/main.go -steps=2000 -interval=200
```

### Experiment 2 — Baseline Restore (Crash Recovery Cost)

Using the worker ID printed by experiment 1
(`experiments_ml/02_baseline_restore/main.go -worker=<id>`), this simulates a
replacement worker starting up after a crash: it loads the latest checkpoint
metadata, then fetches every tensor it references by content hash. It
measures:

- `load_meta_time_s` — find the latest committed step
- `load_tensors_time_s` — fetch all referenced tensor blobs
- `total_restore_time_s`
- `num_tensors` / `total_bytes` — size of the restored checkpoint
- `restore_throughput_kb_per_sec`

**What this demonstrates:** restore cost is proportional to the *size of the
last checkpoint's tensor set*, not to total training history — the "O(changed
tensors), not O(full model)" claim, viewed from the recovery side.

```bash
go run experiments_ml/02_baseline_restore/main.go -worker=bench-1234567890
```

---

## Results

<!-- ML_RESULTS:START -->
_(Run `bash experiments_ml/run_all.sh` then `bash experiments_ml/collect_results.sh`
to populate this section automatically.)_
<!-- ML_RESULTS:END -->

---

## Known Issues / Future Work

- **`server_handle.go`'s `SaveTensors`** does its dedup check via `TestAuth`
  (one RPC per tensor). For models with hundreds of tensors this means
  hundreds of round trips per checkpoint. A batched "check these N hashes"
  RPC would reduce this to one round trip — noted as a follow-up, not
  implemented here to avoid adding a new proto service.
- **`s.files` map growth**: every `Open` (including the read-mode opens in
  `LoadTensors`/`LoadCheckpointMeta`, which now properly `Close`) allocates
  a new file descriptor entry in the server's `s.files` map; this map is
  never pruned even after `Close`. This is a pre-existing issue in the base
  DFS (Project 1), not introduced here, but it bounds how many total
  open/close cycles a long-running cluster can serve before hitting
  `MaxOpenFiles = 1000`. Fine for baseline runs; would need addressing for
  a long-lived training job with thousands of checkpoints.
- **Coordinator straggler detection** (`StragglerThreshold = 200` steps) is
  implemented but not covered by an automated experiment — would need a
  multi-worker harness with artificially staggered step rates.
