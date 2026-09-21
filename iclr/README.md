# ICLR 2027 manuscript workspace

This directory is the only working area for the ICLR version of the project.
It is intentionally isolated from the journal manuscript in the parent directory.

- `main.tex`: ICLR manuscript under active refactoring.
- `references.bib`: ICLR bibliography copied from the journal version and audited independently.
- `figures/`: figures created or adapted specifically for the ICLR paper.
- `experiments/`: ICLR-specific experiment code, configurations, and lightweight metadata.
- `output/`: compiled paper and generated artifacts.
- `docs/`: venue requirements, architecture decisions, and verification records.
- `template/`: untouched official ICLR 2027 style package extracted from the downloaded archive.

The parent journal files, including `../main.tex`, `../refs.bib`, and `../RARL_final.pdf`, are not build inputs for this workspace and must not be overwritten by ICLR builds.
All `.ppt` and `.pptx` files are preserved as irreplaceable editable source assets.
