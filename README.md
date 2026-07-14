# Distributed File System with ML Checkpoint Deduplication

A fault-tolerant distributed file system (DFS) built in Go, extended with a content-addressed checkpoint store for distributed ML training. Workers save model tensors to the DFS; identical tensors are stored only once using SHA-256 hashing — cutting checkpoint storage by **60–70%** as training progresses.

---

## What It Does

The project has two layers:

**Layer 1 — Core DFS**  
Three servers replicate all file operations using a write-ahead log and primary-backup replication. If the primary crashes, backups automatically elect a new leader (smallest-ID wins) and resume serving. Clients always find the current leader automatically.

**Layer 2 — ML Checkpoint Store**  
Training workers save model tensors to the DFS at regular intervals. Each tensor is hashed (SHA-256); if that hash already exists on the server, nothing is written. This means:
- Tensors that haven't changed since the last checkpoint are free to save.
- Tensors shared across multiple workers are stored once, not N times.
- As training progresses and layers stabilize, the dedup ratio climbs — reaching **71.4% storage saved** by step 1000 in baseline runs.

A coordinator (optional) tracks worker heartbeats, detects crashes, and tells replacement workers which checkpoint step to resume from.

---


## How to Run

### Step 1 — Start the 3-node cluster (3 terminals)

```bash
# Terminal 1
cd test_server_1 && go run simple_server.go

# Terminal 2
cd test_server_2 && go run bu1_server.go

# Terminal 3
cd test_server_3 && go run bu2_server.go
```

Wait ~2 seconds for leader election. Server 1 becomes primary.

### Step 2 — Run a training worker

```bash
go run worker.go -id=worker_0 -total=1
```

This runs 1000 simulated training steps, checkpointing every 100 steps. You'll see output like:

```
Worker worker_0: step=100:  0/7 tensors deduped  (0.0% storage saved)
Worker worker_0: step=300:  3/7 tensors deduped  (42.8% storage saved)
Worker worker_0: step=700:  5/7 tensors deduped  (67.3% storage saved)
Worker worker_0: step=1000: 5/7 tensors deduped  (71.4% storage saved)
Worker worker_0: training complete (step 1000)
```

**To simulate a crash and resume:**
```bash
go run worker.go -id=worker_0 -crash=350   # crashes at step 350
go run worker.go -id=worker_0              # resumes from step 300 (last checkpoint)
```

---

## Key Results

| Metric | Value |
|---|---|
| Dedup at step 100 (all tensors new) | ~0% |
| Dedup at step 400 (embedding + layer0 stable) | ~57% |
| Dedup at step 1000 (5 of 7 tensors stable) | **71.4%** |
| Failover detection time | ≤ 5 seconds |
| Write-to-commit latency (3 nodes, localhost) | < 5 ms |
| Miller-Rabin primality (100k numbers) | ~2–3 s, zero false positives |

The model has 7 named tensors (embedding table, 3 layers × weight+bias). The embedding table (~256 KB, largest tensor) and early-layer weights stabilize first, which is why the dedup ratio jumps sharply between steps 200–400 and then plateaus.

---
