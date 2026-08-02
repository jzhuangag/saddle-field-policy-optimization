#!/usr/bin/env python3
"""Live DOI metadata cross-check for all DOI-bearing BibTeX entries."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

from build_citation_passport import _authors, _entries, _fields, _plain


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _date_years(record: dict) -> set[int]:
    years = set()
    for key in ("published", "published-print", "published-online", "issued"):
        parts = record.get(key, {}).get("date-parts", [])
        if parts and parts[0]:
            years.add(int(parts[0][0]))
    return years


def _get_record(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RARL-citation-audit/1.0 (mailto:research-audit@example.invalid)"
        },
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)["message"]


def audit(bib_path: Path) -> dict:
    rows = []
    for _kind, key, body in _entries(bib_path.read_text(encoding="utf-8")):
        fields = _fields(body)
        doi = fields.get("doi")
        if not doi:
            continue
        record = _get_record(doi)
        bib_title = _plain(fields["title"])
        record_title = (record.get("title") or [""])[0]
        title_score = SequenceMatcher(None, _norm(bib_title), _norm(record_title)).ratio()
        bib_families = [_norm(item.get("family", item.get("literal", ""))) for item in _authors(fields["author"])]
        record_families = [_norm(item.get("family", "")) for item in record.get("author", [])]
        author_match = all(name in record_families for name in bib_families if name != "others")
        years = sorted(_date_years(record))
        bib_pages = _norm(fields.get("pages", ""))
        record_pages = _norm(record.get("page", ""))
        crossref_volume = str(record.get("volume") or "")
        crossref_issue = str(record.get("issue") or "")
        volume_match = (
            not fields.get("volume")
            or not crossref_volume
            or _norm(fields["volume"]) == _norm(crossref_volume)
        )
        issue_match = (
            not fields.get("number")
            or not crossref_issue
            or _norm(fields["number"]) == _norm(crossref_issue)
        )
        pages_match = not bib_pages or not record_pages or bib_pages == record_pages
        year_match = int(fields["year"]) in years
        checks = {
            "title": title_score >= 0.95,
            "authors": author_match,
            "year": year_match,
            "volume": volume_match,
            "issue": issue_match,
            "pages": pages_match,
            "doi": _norm(doi) == _norm(record.get("DOI", "")),
        }
        rows.append(
            {
                "citation_key": key,
                "doi": doi,
                "crossref_url": f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}",
                "bib": {
                    "title": bib_title,
                    "year": int(fields["year"]),
                    "venue": fields.get("journal") or fields.get("booktitle") or fields.get("publisher"),
                    "volume": fields.get("volume"),
                    "issue": fields.get("number"),
                    "pages": fields.get("pages"),
                },
                "crossref": {
                    "title": record_title,
                    "authors": [
                        " ".join(filter(None, (item.get("given"), item.get("family"))))
                        for item in record.get("author", [])
                    ],
                    "years": years,
                    "venue": (record.get("container-title") or [""])[0],
                    "publisher": record.get("publisher"),
                    "volume": record.get("volume"),
                    "issue": record.get("issue"),
                    "pages": record.get("page"),
                    "doi": record.get("DOI"),
                },
                "checks": checks,
                "crossref_coverage_gaps": [
                    name
                    for name, bib_value, record_value in (
                        ("volume", fields.get("volume"), crossref_volume),
                        ("issue", fields.get("number"), crossref_issue),
                        ("pages", fields.get("pages"), record.get("page")),
                    )
                    if bib_value and not record_value
                ],
                "status": "verified" if all(checks.values()) else "review",
            }
        )
        time.sleep(0.15)
    return {
        "audit_date": dt.date.today().isoformat(),
        "source": str(bib_path),
        "resolver": "Crossref live DOI endpoint",
        "record_count": len(rows),
        "verified_count": sum(row["status"] == "verified" for row in rows),
        "review_count": sum(row["status"] == "review" for row in rows),
        "records": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bib", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.bib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("record_count", "verified_count", "review_count")}))
    return 0 if result["review_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
