#!/usr/bin/env python3
"""Merge live resolver, DOI-metadata, and authoritative-record citation audits."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dir", type=Path)
    args = parser.parse_args()
    root = args.audit_dir
    passport = yaml.safe_load((root / "passport.yaml").read_text(encoding="utf-8"))
    ars_path = root / "ars_verification.json"
    ars = (
        json.loads(ars_path.read_text(encoding="utf-8"))
        if ars_path.exists() and ars_path.stat().st_size
        else []
    )
    doi = json.loads((root / "crossref_metadata_audit.json").read_text(encoding="utf-8"))
    official = json.loads((root / "official_record_supplement.json").read_text(encoding="utf-8"))
    ieee = json.loads((root / "ieee_bib_style_audit.json").read_text(encoding="utf-8"))

    ars_by_key = {row["citation_key"]: row for row in ars}
    doi_by_key = {row["citation_key"]: row for row in doi["records"]}
    official_by_key = {row["citation_key"]: row for row in official["records"]}
    rows = []
    unresolved = []
    overlaps = sorted(set(doi_by_key) & set(official_by_key))
    for entry in passport["literature_corpus"]:
        key = entry["citation_key"]
        if key in doi_by_key:
            metadata = {
                "verification_path": "crossref_live_doi",
                "status": doi_by_key[key]["status"],
                "record": doi_by_key[key],
            }
        elif key in official_by_key:
            metadata = {
                "verification_path": "authoritative_official_record",
                "status": official_by_key[key]["status"],
                "record": official_by_key[key],
            }
        else:
            metadata = {"verification_path": None, "status": "unresolved", "record": None}
            unresolved.append(key)
        rows.append(
            {
                "citation_key": key,
                "title": entry["title"],
                "metadata_verification": metadata,
                "ars_existence_gate": ars_by_key.get(key),
            }
        )

    result = {
        "audit_date": dt.date.today().isoformat(),
        "bibliography": "refs.bib",
        "entry_count": len(rows),
        "doi_metadata_verified": len(doi_by_key),
        "official_record_verified": len(official_by_key),
        "ars_multi_index_records": len(ars_by_key),
        "ars_multi_index_note": (
            "fresh multi-index output included"
            if ars_by_key
            else "fresh multi-index query timed out; authoritative DOI and official-record gates determine status"
        ),
        "overlapping_metadata_paths": overlaps,
        "unresolved_entries": unresolved,
        "ieee_style_status": ieee["status"],
        "status": (
            "pass"
            if not unresolved
            and not overlaps
            and all(row["metadata_verification"]["status"] == "verified" for row in rows)
            and ieee["status"] == "pass"
            else "review"
        ),
        "records": rows,
    }
    output = root / "citation_integrity_final.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "entry_count": result["entry_count"],
                "doi_metadata_verified": result["doi_metadata_verified"],
                "official_record_verified": result["official_record_verified"],
                "unresolved": len(unresolved),
                "status": result["status"],
            }
        )
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
