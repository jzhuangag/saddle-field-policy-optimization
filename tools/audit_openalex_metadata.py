#!/usr/bin/env python3
"""Live OpenAlex metadata cross-check for every BibTeX entry."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import urllib.error
from difflib import SequenceMatcher
from pathlib import Path

from build_citation_passport import _authors, _entries, _fields, _plain


def _norm(value: str) -> str:
    value = value or ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _request(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RARL-citation-audit/1.0 (mailto:research-audit@example.invalid)"},
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)


def _lookup(fields: dict[str, str]) -> tuple[dict, str]:
    doi = fields.get("doi")
    if doi:
        encoded = urllib.parse.quote(f"https://doi.org/{doi}", safe="")
        url = f"https://api.openalex.org/works/{encoded}"
        try:
            return _request(url), url
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
    search_title = re.sub(r"[^A-Za-z0-9 ]+", " ", _plain(fields["title"]))
    query = urllib.parse.urlencode({"search": search_title, "per-page": 10})
    url = f"https://api.openalex.org/works?{query}"
    results = _request(url).get("results", [])
    if not results:
        return {}, url
    title = _plain(fields["title"])

    def candidate_score(row: dict) -> float:
        biblio = row.get("biblio") or {}
        first_page = biblio.get("first_page")
        last_page = biblio.get("last_page")
        pages = "--".join(str(item) for item in (first_page, last_page) if item is not None)
        record_names = [_norm(item.get("author", {}).get("display_name", "")) for item in row.get("authorships", [])]
        bib_families = [
            _norm(item.get("family", item.get("literal", "")))
            for item in _authors(fields["author"])
            if item.get("family", item.get("literal", "")) != "others"
        ]
        author_fraction = sum(any(name in record_name for record_name in record_names) for name in bib_families) / max(1, len(bib_families))
        return (
            5 * SequenceMatcher(None, _norm(title), _norm(row.get("display_name", ""))).ratio()
            + 2 * (row.get("publication_year") == int(fields["year"]))
            + 2 * (not fields.get("pages") or not pages or _norm(fields["pages"]) == _norm(pages))
            + 1 * (not fields.get("volume") or not biblio.get("volume") or _norm(fields["volume"]) == _norm(str(biblio["volume"])))
            + author_fraction
        )

    result = max(results, key=candidate_score)
    return result, url


def audit(bib_path: Path) -> dict:
    rows = []
    for _kind, key, body in _entries(bib_path.read_text(encoding="utf-8")):
        fields = _fields(body)
        try:
            record, query_url = _lookup(fields)
        except urllib.error.HTTPError as exc:
            record, query_url = {}, exc.url
            print(f"OpenAlex HTTP {exc.code}: {key}")
        bib_title = _plain(fields["title"])
        record_title = record.get("display_name", "")
        title_score = SequenceMatcher(None, _norm(bib_title), _norm(record_title)).ratio()
        bib_families = [
            _norm(item.get("family", item.get("literal", "")))
            for item in _authors(fields["author"])
            if item.get("family", item.get("literal", "")) != "others"
        ]
        record_names = [
            _norm(item.get("author", {}).get("display_name", ""))
            for item in record.get("authorships", [])
            if item.get("author", {}).get("display_name")
        ]
        author_match = all(any(name in record_name for record_name in record_names) for name in bib_families)
        biblio = record.get("biblio") or {}
        first_page = biblio.get("first_page")
        last_page = biblio.get("last_page")
        record_pages = "--".join(str(item) for item in (first_page, last_page) if item is not None)
        bib_pages = _norm(fields.get("pages", ""))
        pages_match = not bib_pages or not record_pages or bib_pages == _norm(record_pages)
        bib_doi = _norm(fields.get("doi", ""))
        record_doi = _norm(re.sub(r"^https?://(?:dx\.)?doi\.org/", "", record.get("doi") or "", flags=re.I))
        checks = {
            "title": title_score >= 0.95,
            "authors": author_match,
            "year": int(fields["year"]) == record.get("publication_year"),
            "volume": not fields.get("volume") or not biblio.get("volume") or _norm(fields["volume"]) == _norm(str(biblio["volume"])),
            "issue": not fields.get("number") or not biblio.get("issue") or _norm(fields["number"]) == _norm(str(biblio["issue"])),
            "pages": pages_match,
            "doi": not bib_doi or not record_doi or bib_doi == record_doi,
        }
        rows.append(
            {
                "citation_key": key,
                "query_url": query_url,
                "openalex_id": record.get("id"),
                "landing_page_url": (record.get("primary_location") or {}).get("landing_page_url"),
                "bib": {
                    "title": bib_title,
                    "year": int(fields["year"]),
                    "venue": fields.get("journal") or fields.get("booktitle") or fields.get("publisher"),
                    "volume": fields.get("volume"),
                    "issue": fields.get("number"),
                    "pages": fields.get("pages"),
                    "doi": fields.get("doi"),
                },
                "openalex": {
                    "title": record_title,
                    "authors": [item.get("author", {}).get("display_name") for item in record.get("authorships", [])],
                    "year": record.get("publication_year"),
                    "venue": ((record.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "volume": biblio.get("volume"),
                    "issue": biblio.get("issue"),
                    "pages": record_pages or None,
                    "doi": record.get("doi"),
                },
                "checks": checks,
                "status": "verified" if record and all(checks.values()) else "review",
            }
        )
        time.sleep(0.12)
    return {
        "audit_date": dt.date.today().isoformat(),
        "source": str(bib_path),
        "resolver": "OpenAlex live works endpoint",
        "record_count": len(rows),
        "verified_count": sum(row["status"] == "verified" for row in rows),
        "review_count": sum(row["status"] == "review" for row in rows),
        "records": rows,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("bib", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.bib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("record_count", "verified_count", "review_count")}))
    for row in result["records"]:
        if row["status"] == "review":
            print(json.dumps({"citation_key": row["citation_key"], "checks": row["checks"], "openalex": row["openalex"]}, ensure_ascii=False))
    return 0 if result["review_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
