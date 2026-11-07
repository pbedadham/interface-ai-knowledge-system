# Snapshot-Aware Knowledge System

A lightweight Python knowledge system for answering questions from versioned website snapshots. It preserves historical content, retrieves evidence from the appropriate point in time, cites its sources, and identifies numeric changes across rate and fee pages.

The implementation follows a simple `build()` / `ingest()` / `answer()` / `conflicts()` contract and uses only the Python standard library. It requires no external services, API keys, vector database, or package installation.

## Key capabilities

- **Snapshot-aware answers:** Uses the latest available snapshot by default or the newest snapshot on or before an optional `as_of` date.
- **Evidence-based retrieval:** Ranks short passages with lightweight BM25-style lexical scoring while keeping Markdown tables intact as coherent evidence.
- **Source prioritization:** Favors authoritative product and rate pages over blogs, calculators, newsroom articles, and stale promotional content.
- **Citations:** Returns the source URL, snapshot identifier, and a supporting excerpt with each grounded answer.
- **Cross-page conflict detection:** Normalizes product, term, tier, metric, and value so disagreements can be detected across different pages as well as across historical versions of one page.
- **Grounded conflict resolution:** Keeps qualifiers aligned, then uses source authority and snapshot freshness to select a value while preserving citations to the competing evidence.
- **Safe fallback behavior:** Returns a contact-oriented response with low confidence when the available snapshots do not contain enough relevant evidence.
- **Deterministic operation:** Runs locally without live website requests or model-generated output.

## Requirements

- Python 3.10 or newer
- The supplied website snapshot data

No third-party dependencies are required.

## Repository layout

```text
.
├── eval/
│   └── public_eval.jsonl        # Public evaluation questions and expected behavior
├── harness/
│   ├── contract.py              # Data models and knowledge-system protocol
│   ├── run_eval.py              # Evaluation runner
│   └── scoring.py               # Answer, citation, consistency, and conflict scoring
├── submission/
│   ├── solution.py              # Knowledge-system implementation
│   ├── NOTES.md                 # Additional design and data-handling notes
│   └── requirements.txt         # Dependency declaration
└── README.md
```

## Running the evaluation

Place the five supplied snapshot folders in `snapshots/` (or pass their directory explicitly), then run:

```bash
python harness/run_eval.py --submission submission --snapshots snapshots --eval eval/public_eval.jsonl
```

The runner ingests snapshots from oldest to newest, evaluates each question, checks repeat-answer consistency, and writes detailed results to:

```text
results/report.json
results/per_question.jsonl
```

Use `--out <directory>` to choose a different output location. If the data is provided as ZIP archives, place them in `snapshots_zip/`; the runner will unpack them into `snapshots/` on first use.

### Optional evaluation flags

```text
--conflicts-gold <file>  Score the output of conflicts() against a reference file
--repeat-sample <n>      Set the number of questions repeated for consistency checks
--judge                  Enable the optional free-text LLM judge (requires ANTHROPIC_API_KEY)
--out <directory>        Choose where evaluation reports are written
```

## System contract

The harness interacts with the implementation through four operations:

```python
system = build()
system.ingest(snapshot)
system.answer(question, as_of=None)
system.conflicts()
```

`answer()` returns member-facing text together with sources, a confidence indicator, and an optional note explaining how conflicting evidence was resolved. `conflicts()` returns structured descriptions of disagreements found within or across snapshots.

## Testing

Run the conflict-detection regression tests with:

```bash
python -m unittest discover -s tests -v
```

The tests cover stale values on secondary pages, changes across historical versions, and qualifier separation for different terms and balance tiers.

## Data handling

The supplied website snapshots are intentionally excluded from this repository. The system reads them only from the local filesystem and makes no live network requests. See `submission/NOTES.md` for further implementation details.
