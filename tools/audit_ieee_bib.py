#!/usr/bin/env python3
"""Static IEEEtran/BibTeX consistency audit for the submission manuscript."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from build_citation_passport import _entries, _fields, _plain


def cited_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite\w*\s*\{([^}]+)\}", tex, flags=re.S):
        keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def audit(tex_path: Path, bib_path: Path) -> dict:
    records = {}
    definite_errors = []
    judgment_calls = []
    for kind, key, body in _entries(bib_path.read_text(encoding="utf-8")):
        fields = _fields(body)
        records[key] = {"kind": kind, **fields}
        required = {"author", "title", "year"}
        required.add("journal" if kind == "article" else "booktitle" if kind == "inproceedings" else "publisher")
        for field in sorted(required):
            if not fields.get(field, "").strip():
                definite_errors.append(f"{key}: missing {field}")
        for field, value in fields.items():
            if not value.strip():
                definite_errors.append(f"{key}: empty field {field}")
        if kind == "article":
            for field in ("volume", "pages"):
                if not fields.get(field):
                    judgment_calls.append(f"{key}: article has no {field}")
        if kind == "inproceedings" and not fields.get("booktitle", "").startswith("Proc."):
            definite_errors.append(f"{key}: conference booktitle does not use Proc.")
        if fields.get("pages") and not re.fullmatch(r"[0-9]+\s*--\s*[0-9]+", fields["pages"]):
            definite_errors.append(f"{key}: noncanonical page range {fields['pages']!r}")
        if fields.get("doi") and not re.fullmatch(r"10\.[0-9]{4,9}/\S+", fields["doi"]):
            definite_errors.append(f"{key}: malformed DOI {fields['doi']!r}")

    cited = cited_keys(tex_path.read_text(encoding="utf-8"))
    missing = sorted(cited - records.keys())
    unused = sorted(records.keys() - cited)
    normalized_titles = Counter(_plain(record["title"]).casefold() for record in records.values())
    duplicate_titles = sorted(title for title, count in normalized_titles.items() if count > 1)
    normalized_dois = Counter(
        record["doi"].lower() for record in records.values() if record.get("doi")
    )
    duplicate_dois = sorted(doi for doi, count in normalized_dois.items() if count > 1)
    definite_errors.extend(f"citation missing from BibTeX: {key}" for key in missing)

    return {
        "manuscript": str(tex_path),
        "bibliography": str(bib_path),
        "style": "IEEEtran.bst",
        "entry_count": len(records),
        "cited_key_count": len(cited),
        "missing_cited_keys": missing,
        "unused_entries": unused,
        "duplicate_titles": duplicate_titles,
        "duplicate_dois": duplicate_dois,
        "definite_errors": definite_errors,
        "judgment_calls": judgment_calls,
        "status": "pass" if not definite_errors and not unused and not duplicate_titles and not duplicate_dois else "review",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tex", type=Path)
    parser.add_argument("bib", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = audit(args.tex, args.bib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
