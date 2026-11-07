# Implementation notes

The solution retains every full site snapshot and selects the newest snapshot at or before the requested `as_of` date. It indexes short text passages while preserving Markdown tables as coherent evidence, then ranks passages with lightweight BM25-style lexical scoring. Query aliases handle common wording differences such as CD/certificate, auto/vehicle, and yield/APY. Page path, title, and content contribute to relevance.

Current answers favor primary product and rate pages. Calculators, blogs, newsroom stories, and historic passages receive lower ranking so that stale promotional or illustrative figures do not override the current terms page. Answers cite the page URL and snapshot, include a short supporting excerpt, and return a contact-oriented refusal when retrieval has too little grounded evidence. The conflict report compares materially different numeric passages between versions of the same rate or fee page and records the newer page as its resolution.

The implementation uses Python's standard library only. It makes no live website calls and requires no API keys. The supplied snapshots are needed locally to evaluate it and must not be committed to a public repository.
