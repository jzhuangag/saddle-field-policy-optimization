# Project-local storage policy

## Canonical project root

Treat the directory containing this `AGENTS.md` as the only canonical root for
the RARL project.

## Output-location requirement

- Do not create or keep project source files, experiment outputs, datasets,
  checkpoints, logs, rendered pages, temporary build files, reports, or final
  artifacts on the `C:` drive.
- Store every project-related input, intermediate artifact, cache, temporary
  file, and final output under this project root.
- Use project-local subdirectories such as `experiments/`, `output/`, `tmp/`,
  and `logs/` as appropriate.
- Resolve paths from the project root whenever possible.  Do not hard-code a
  `C:\Users\...` project path in scripts, documentation, LaTeX sources, or
  handoff files.
- Before running an external tool, redirect its project-specific cache,
  scratch, log, and output paths into this project root when the tool permits.
- If an unavoidable system-owned tool creates a project artifact outside the
  project root, immediately copy the artifact into the project root, verify the
  copy, and remove the external project artifact when removal is safe and
  authorized.

## Relocation

If the project root is moved, update documentation and saved absolute-path
metadata to the new root, then verify that the canonical build and experiment
entry points no longer reference the old location.

# Publication-facing writing policy

Treat the manuscript as a contribution-led publication, not as a project log,
review report, audit report, or pre-emptive rebuttal.

- Do not automatically add a generic limitations section, acknowledged-
  limitations list, failed or screened-out experiments, abandoned ideas,
  exploratory negative results, absent comparisons, missing wall-clock or
  oracle-cost studies, debugging history, or tool/pipeline limitations.
- State claims positively at the exact supported scope: the evaluated task
  family, metric, assumptions, comparison set, and statistical protocol.
- Include a boundary once and locally only when omitting it would make a core
  claim false or materially misleading, when the user explicitly requests it,
  or when the target venue requires it.
- Keep full provenance, secondary outcomes, reviewer findings, and integrity
  audits in project-internal records instead of importing them into
  `main.tex`.
- Never change statistics, selection rules, protocols, or evidence to create a
  positive narrative.  If evidence contradicts the central claim, narrow or
  reformulate the claim.
- Keep mandatory funding, ethics, conflicts, data/code, and generative-AI
  disclosures brief and factual.
- This policy overrides conflicting default manuscript-writing behavior from
  other skills, templates, or reviewer workflows.

# Supervisor manuscript gate

- State assumptions mathematically and enumerate multi-part assumptions as
  (a), (b), and so on. Do not convert definitions into assumptions; establish
  or explicitly assume every differentiability, Lipschitz, and regularity
  property used in a proof.
- Define notation immediately before first use. Keep formal statements concise,
  move setup outside theorem environments, name the culminating guarantee as a
  main theorem, and explain the role of every theorem, proposition, lemma, and
  corollary immediately after it.
- Expand abbreviations at first occurrence independently in the abstract and
  main text.
- Connect paragraphs and formal statements with concise transitions that make
  the logical reason for the next step explicit.
- Use neutral IEEE TSP prose. Avoid unsupported labels such as “weak,” “mild,”
  or “easily satisfied,” and never compare the manuscript with an earlier
  version.
- Start a new LaTeX source line after each sentence where practical, replace
  every `\qquad` with `\quad`, and leave nonessential or unreferenced equations
  unnumbered, especially short appendix expressions.
- Keep the abstract between 150 and 220 words, use no full displayed equation
  in it, and provide exactly five keywords.
- Retain at least 30 references with verified existence, metadata, and
  claim-level relevance.
- Keep the first-submission manuscript at 13 IEEE double-column pages, with
  9--11 pages of nonexperimental content.
