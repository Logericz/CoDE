# CoDE project workflow

- This directory is the canonical workspace for the user's CoDE-Stop reproduction and related research. Keep older reasoning experiments outside this project.
- GitHub destination: https://github.com/Ericzhy0716/CoDE, default branch `main`. The user has authorized ongoing synchronization of relevant code and documents. Commit and push meaningful, verified changes when completing related work; preserve remote history and do not force-push.
- Keep `upstream/CoDE-Stop` as the pinned original reference. Put our implementations or patches in `src/` and `scripts/`, and document differences from the paper and upstream code.
- Never stage model weights, credentials, machine-specific authorization, raw datasets, or unreviewed run output. Put raw output in ignored `runs/`; share curated reports in `results/`.
- Use `docs/CODESTOP_NOVELTY_ASSESSMENT_20260920.md` for current research scope, and the reproduction requirements document for technical setup. Do not describe static checks, planned work or offline replay as completed GPU experiments or measured online speedups.

## Code-reading diagrams and Notion

- When the user confirms finishing a function or code block, inspect the current implementation and add or update an understandable diagram beside its notes in the corresponding Notion learning page. Include inputs, outputs, decisions, calls, and the boundary between returning, grading, and saving where relevant.
- Keep editable diagram sources and rendered images in `docs/diagrams/`; maintain the index there. Verify rendering and re-fetch Notion to verify placement. Preserve the user's original answers and actual completion status; preparing a diagram does not prove the reading or experiment is complete.
- Reuse an existing diagram when it still matches the code; cross-link review pages instead of duplicating it. Record the source version or audit date and distinguish upstream behavior from our teaching wrapper. Do not edit annotated upstream code as part of illustration work.
