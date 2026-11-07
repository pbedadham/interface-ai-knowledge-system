# Implementation notes

The solution retains every full site snapshot and selects the newest snapshot at or before the requested `as_of` date. It indexes short text passages while preserving Markdown tables as coherent evidence, then ranks passages with lightweight BM25-style lexical scoring. Query aliases handle common wording differences such as CD/certificate, auto/vehicle, and yield/APY. Page path, title, and content contribute to relevance.

Current answers favor primary product and rate pages. Calculators, blogs, newsroom stories, and historic passages receive lower ranking so that stale promotional or illustrative figures do not override the current terms page. Answers cite the page URL and snapshot, include a short supporting excerpt, and return a contact-oriented refusal when retrieval has too little grounded evidence.

Rate-bearing passages are also normalized into structured facts containing the product, term, tier, metric, value, source URL, snapshot, and source authority. The conflict report groups only facts with aligned qualifiers, then compares them both across versions of one URL and across different URLs in the same snapshot. It resolves disagreements using source authority and freshness and cites the competing passages. Answer retrieval remains independent so conflict-report heuristics cannot demote a correctly tiered answer. Different terms, model years, and balance tiers remain separate rather than being reported as contradictions.

An LLM could optionally help normalize unusual product wording in a production system, but it is deliberately not required here. Numeric extraction, comparison, and resolution remain deterministic and grounded in the supplied text, preserving repeatability, latency, and offline operation.

The implementation uses Python's standard library only. It makes no live website calls and requires no API keys. The supplied snapshots are needed locally to evaluate it and must not be committed to a public repository.
