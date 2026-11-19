# Historical source archive, not active code

These two original project files are byte-for-byte copies from public commit
`8f538be45034eb4696c89973937ef1d16a1db178`. Their SHA256 values match the existing
Nov18 `results/cpu-kv-v1/report.json`, which remains unmodified.

Nov20 adds opt-in profiler scopes to the live files. The historical offline audit
now explicitly checks these archived versions for those two paths; other source
paths still resolve as before. No shell/git history/network is required by the audit.
Old timing results are not measurements of the instrumented/current code, even
with scopes disabled: a no-op context still adds Python overhead. New attribution
reports pin their own actual source hashes and make no timing comparison.
