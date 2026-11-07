# Interface.ai Knowledge Hackathon Submission

Python implementation of the required `build()` / `ingest()` / `answer()` / `conflicts()` contract. The system retains snapshot history, answers against the latest snapshot on or before an optional `as_of` date, prioritizes authoritative source pages, cites supporting excerpts, and detects numeric changes in rate and fee pages. It uses the Python standard library only.

## Run the public evaluation

Place the five supplied snapshot folders in `snapshots/` (or pass their directory explicitly), then run:

```bash
python harness/run_eval.py --submission submission --snapshots snapshots --eval eval/public_eval.jsonl
```

The supplied website snapshot data is intentionally not included in this repository. See `submission/NOTES.md` for the design and data-handling details.
