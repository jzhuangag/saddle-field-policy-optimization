#!/usr/bin/env python3
"""Build an ARS Material Passport from this project's BibTeX database."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


def _entries(text: str):
    start_re = re.compile(r"@(?P<kind>[A-Za-z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,")
    for match in start_re.finditer(text):
        depth = 1
        pos = match.end()
        begin = pos
        while pos < len(text) and depth:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth:
            raise ValueError(f"unclosed BibTeX entry {match.group('key')}")
        yield match.group("kind").lower(), match.group("key"), text[begin : pos - 1]


def _fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    pos = 0
    while pos < len(body):
        while pos < len(body) and (body[pos].isspace() or body[pos] == ","):
            pos += 1
        if pos >= len(body):
            break
        name_match = re.match(r"([A-Za-z][A-Za-z0-9_-]*)\s*=", body[pos:])
        if not name_match:
            raise ValueError(f"cannot parse BibTeX field near {body[pos:pos+50]!r}")
        name = name_match.group(1).lower()
        pos += name_match.end()
        while pos < len(body) and body[pos].isspace():
            pos += 1
        if body[pos] == "{":
            depth, begin = 1, pos + 1
            pos += 1
            while pos < len(body) and depth:
                if body[pos] == "{":
                    depth += 1
                elif body[pos] == "}":
                    depth -= 1
                pos += 1
            value = body[begin : pos - 1]
        elif body[pos] == '"':
            begin = pos + 1
            pos += 1
            while pos < len(body) and body[pos] != '"':
                pos += 2 if body[pos] == "\\" else 1
            value = body[begin:pos]
            pos += 1
        else:
            begin = pos
            while pos < len(body) and body[pos] != ",":
                pos += 1
            value = body[begin:pos].strip()
        fields[name] = value.strip()
    return fields


def _plain(value: str) -> str:
    value = value.replace("--", "–")
    replacements = {
        r"{\c{s}}": "s",
        r"{{\L}": "L",
        r"{\L}": "L",
        r"\"a": "a",
        r"\"o": "o",
        r"\'e": "e",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"\\[A-Za-z]+\s*", "", value)
    value = value.replace("{", "").replace("}", "").replace("$", "")
    return re.sub(r"\s+", " ", value).strip()


def _authors(value: str) -> list[dict[str, str]]:
    parsed = []
    for raw_name in re.split(r"\s+and\s+", value):
        name = _plain(raw_name)
        if name.lower() == "others":
            parsed.append({"literal": "others"})
        elif "," in name:
            family, given = (part.strip() for part in name.split(",", 1))
            parsed.append({"family": family, "given": given})
        else:
            parts = name.split()
            parsed.append(
                {"family": parts[-1], "given": " ".join(parts[:-1])}
                if len(parts) > 1
                else {"literal": name}
            )
    return parsed


def build(bib_path: Path) -> dict:
    corpus = []
    source_uri = bib_path.resolve().as_uri()
    for kind, key, body in _entries(bib_path.read_text(encoding="utf-8")):
        fields = _fields(body)
        venue = fields.get("journal") or fields.get("booktitle") or fields.get("publisher")
        item = {
            "citation_key": key,
            "title": _plain(fields["title"]),
            "authors": _authors(fields["author"]),
            "year": int(fields["year"]),
            "source_pointer": (
                f"https://doi.org/{fields['doi']}"
                if fields.get("doi")
                else f"{source_uri}#{key}"
            ),
            "venue": _plain(venue) if venue else "Unknown",
            "obtained_via": "other",
            "adapter_name": "rarl-bibtex-audit",
            "adapter_version": "1.0",
        }
        if fields.get("doi"):
            item["doi"] = fields["doi"]
        corpus.append(item)
    return {"literature_corpus": corpus}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bib", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    passport = build(args.bib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(passport, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {len(passport['literature_corpus'])} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
