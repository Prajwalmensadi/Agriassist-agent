"""Inventory local agricultural PDFs without changing their contents."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "pdf_inventory.json"
FIRST_PAGES_TO_INSPECT = 4
MINIMUM_TEXT_CHARS_PER_PAGE = 30


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _document_facts(first_pages: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Classify from extracted page text, using metadata only as supporting evidence."""

    text = _normalized(first_pages)
    lower = text.casefold()

    if "weather based agro advisory" in lower or "agromet advisory service bulletin" in lower:
        title_match = re.search(
            r"Agromet Advisory Service Bulletin for ([^\n]+ District)", first_pages, re.I
        )
        title = (
            f"Agromet Advisory Service Bulletin for {title_match.group(1).strip()}"
            if title_match
            else None
        )
        forecast = re.search(
            r"Weather Forecast from\s*([0-9.]+)\s*to\s*([0-9.]+)", text, re.I
        )
        issue_date = re.search(r"\bDate:\s*(\d{2}\.\d{2}\.\d{4})", text)
        return {
            "extracted_title": title,
            "publisher": (
                "India Meteorological Department (Pune & MC, Bengaluru) and "
                "Gramin Krishi Mausam Sewa (GKMS), Zonal Agricultural and Horticultural "
                "Research Station, Brahmavar"
                if "issued jointly by india meteorological department" in lower
                else None
            ),
            "publication_date": _to_iso_date(issue_date.group(1)) if issue_date else None,
            "topic_category": "agricultural_weather_advisory",
            "crop_relevance": (
                "Paddy (rice) is explicitly covered; the bulletin also includes advice for "
                "other crops/enterprises, including coconut and arecanut."
            ),
            "geographic_relevance": (
                "The page text identifies Dakshina Kannada District and coastal Karnataka; "
                "PDF metadata instead names Udupi District. The district scope needs verification."
            ),
            "document_type": "District agrometeorological bulletin combining a short-range forecast, past-weather observations, and crop-specific advisories.",
            "weather_content_types": [
                "weather_forecast",
                "weather_observation",
                "agricultural_weather_advisory",
            ],
            "weather_forecast_period": (
                f"{forecast.group(1)} to {forecast.group(2)} (as printed)" if forecast else None
            ),
            "source_url": None,
            "source_suitability": "needs_manual_review",
            "verification_notes": [
                "First-page text explicitly names the bulletin, issuing bodies, district, and paddy advisory section.",
                "The PDF metadata title says Udupi District while the body title and weather tables say Dakshina Kannada District.",
                "Forecast period is historical and expired; retain only as dated historical material unless separately reviewed.",
                "Bulletin mixes paddy advice with advice for other crops; retrieval must filter advice by crop.",
                "No source-document URL was identified; any embedded app links were not treated as source URLs.",
                "Authenticity, current authority, and redistribution permission have not been independently verified.",
            ],
            "confidence_or_uncertainty": "High confidence in document type, paddy coverage, and forecast/observation/advisory distinction from extracted text. District conflict, authority, current relevance, and reuse permission remain unverified.",
            "_text_date": issue_date.group(1) if issue_date else None,
        }

    rules = (
        (
            "management of bacterial leaf blight of rice caused by xanthomonas oryzae pv. oryzae through integrated approach",
            "Management of Bacterial leaf blight of rice caused by Xanthomonas oryzae pv. oryzae through Integrated approach in western undulating zones of Odisha",
            "rice_disease",
            "Rice (paddy); specifically bacterial leaf blight.",
            "Western undulating zones of Odisha, as stated in the article title; transfer to coastal Karnataka is unverified.",
            "Research article (explicitly labeled in the document).",
            "candidate_for_rag",
            "Society for Advancement of Wheat and Barley Research",
        ),
        (
            "management of bacterial blight in rice (oryza sativa) through combined application of endophytes and rhizosphere antagonist",
            "Management of bacterial blight in rice (Oryza sativa) through combined application of endophytes and rhizosphere antagonist",
            "rice_disease",
            "Rice (paddy); specifically bacterial blight and microbial antagonists.",
            "Assam Agricultural University, Jorhat, Assam is named as an affiliation; applicability to coastal Karnataka is not established.",
            "Journal research article with methods and results; the first page names the Indian Journal of Agricultural Sciences.",
            "needs_manual_review",
            None,
        ),
        (
            "increasing rice production from waterlogged rice ecologies",
            "Increasing rice production from waterlogged rice ecologies by adopting high yielding varieties and their package of practices",
            "soil_or_water_management",
            "Rice (paddy), especially waterlogged ecologies and associated varieties/practices.",
            "ICAR-National Rice Research Institute, Cuttack, Odisha is named; local applicability to coastal Karnataka needs review.",
            "Indian Farming article; the first page names the periodical and April 2021 issue.",
            "candidate_for_rag",
            None,
        ),
        (
            "management of micro-nutrient deficiency and toxicity in rice",
            "Management of micro-nutrient deficiency and toxicity in rice",
            "nutrient_management",
            "Rice (paddy); micronutrient deficiency and toxicity.",
            "ICAR-Indian Institute of Rice Research, Hyderabad, Telangana is named as an affiliation; advice is not specifically coastal Karnataka.",
            "Indian Farming article; the first page names the periodical and October 2022 issue.",
            "candidate_for_rag",
            None,
        ),
        (
            "emerging minor diseases of rice in odisha",
            "Emerging minor diseases of rice in Odisha: Identification and Management",
            "rice_disease",
            "Rice (paddy); identification and management of minor diseases discussed in the article.",
            "Odisha is explicit in the title; ICAR-National Rice Research Institute, Cuttack is named as an affiliation.",
            "Indian Farming article; the first page names the periodical and April 2021 issue.",
            "candidate_for_rag",
            None,
        ),
        (
            "rice sheath blight: reappraising the host-pathogen interplay and breeding strategies in the omics era",
            "Rice sheath blight: Reappraising the host-pathogen interplay and breeding strategies in the Omics Era",
            "rice_disease",
            "Rice (paddy); sheath blight, host-pathogen biology, and breeding strategies.",
            "Author affiliations named in the document include institutions in Uttar Pradesh, West Bengal, and Odisha; study applicability to coastal Karnataka is not established.",
            "Indian Farming article discussing sheath-blight biology and breeding strategies; the first page names the periodical and October 2023 issue.",
            "candidate_for_rag",
            None,
        ),
    )

    for signature, title, category, crop, geography, doc_type, suitability, publisher in rules:
        if signature in lower:
            source_url = _own_doi(first_pages)
            publication_date = _publication_date(text)
            notes = [
                "Title and subject were identified in extracted page text, not inferred from the filename.",
                geography,
                "Candidate status reflects topical usefulness only; authority, local applicability, and redistribution permission require manual verification.",
            ]
            if source_url:
                notes.append("The DOI URL is explicitly printed in the document.")
            else:
                notes.append("No source-document URL was explicitly identified in the inspected pages.")
            return {
                "extracted_title": title,
                "publisher": publisher,
                "publication_date": publication_date,
                "topic_category": category,
                "crop_relevance": crop,
                "geographic_relevance": geography,
                "document_type": doc_type,
                "weather_content_types": [],
                "weather_forecast_period": None,
                "source_url": source_url,
                "source_suitability": suitability,
                "verification_notes": notes,
                "confidence_or_uncertainty": "High confidence in title and topic from extracted text. Local transferability, authority, and permission to reuse are not verified.",
            }

    metadata_title = (metadata.get("title") or "").strip()
    if not metadata_title or re.search(r"\.(pdf|pmd)$", metadata_title, re.I):
        metadata_title = None
    return {
        "extracted_title": metadata_title,
        "publisher": None,
        "publication_date": None,
        "topic_category": "unrelated_or_unknown",
        "crop_relevance": "unknown",
        "geographic_relevance": "unknown",
        "document_type": "unknown",
        "weather_content_types": [],
        "weather_forecast_period": None,
        "source_url": None,
        "source_suitability": "needs_manual_review",
        "verification_notes": ["No supported topic rule matched the inspected text; manual classification is required."],
        "confidence_or_uncertainty": "Low confidence; document could not be classified from current text rules.",
    }


def _to_iso_date(value: str) -> str | None:
    try:
        return datetime.strptime(value, "%d.%m.%Y").date().isoformat()
    except ValueError:
        return None


def _publication_date(text: str) -> str | None:
    match = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b",
        text,
        re.I,
    )
    if match:
        return f"{match.group(1).title()} {match.group(2)}"
    citation = re.search(r"Journal of Cereal Research\s+16\s*\(1\).*?(20\d{2})", text, re.I)
    return citation.group(1) if citation else None


def _own_doi(first_pages: str) -> str | None:
    """Return an explicitly printed DOI URL when it appears in the opening pages."""

    match = re.search(
        r"https?://(?:dx\.)?doi\.org/\s*(10\.\d{4,9}/[^\s<>]+)",
        first_pages,
        re.I,
    )
    if not match:
        return None
    doi = match.group(1).rstrip(".,;)")
    if doi.endswith("-"):
        remainder = first_pages[match.end() :]
        continuation = re.match(r"\s+([0-9A-Za-z._;()/:+-]+)", remainder)
        if continuation:
            doi += continuation.group(1)
    return f"https://doi.org/{doi}"


def inventory_pdf(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "page_count": None,
        "extracted_character_count": 0,
        "extraction_status": "unreadable",
        "possible_scanned_pdf": True,
        "extracted_title": None,
        "publisher": None,
        "publication_date": None,
        "topic_category": "unrelated_or_unknown",
        "crop_relevance": "unknown",
        "geographic_relevance": "unknown",
        "document_type": "unknown",
        "source_url": None,
        "source_suitability": "needs_manual_review",
        "verification_notes": [],
        "confidence_or_uncertainty": "No classification confidence: PDF could not be read.",
        "weather_content_types": [],
        "weather_forecast_period": None,
    }
    try:
        with pymupdf.open(path) as document:
            record["page_count"] = len(document)
            all_text = "\n".join(page.get_text() for page in document)
            first_pages = "\n".join(
                document[index].get_text()
                for index in range(min(FIRST_PAGES_TO_INSPECT, len(document)))
            )
            char_count = len(all_text)
            record["extracted_character_count"] = char_count
            minimum = max(100, len(document) * MINIMUM_TEXT_CHARS_PER_PAGE)
            record["possible_scanned_pdf"] = char_count < minimum
            record["extraction_status"] = (
                "little_or_no_text_possible_scan"
                if record["possible_scanned_pdf"]
                else "ok"
            )
            facts = _document_facts(first_pages, document.metadata or {})
            record.update(facts)
            record.pop("_text_date", None)
            if record["possible_scanned_pdf"]:
                record["source_suitability"] = "needs_manual_review"
                record["verification_notes"].append(
                    "Little extractable text was found; visual review or OCR may be needed."
                )
    except Exception as error:  # Keep one damaged PDF from stopping the inventory.
        record["extraction_status"] = "unreadable"
        record["source_suitability"] = "needs_manual_review"
        record["verification_notes"] = [
            f"PDF extraction failed: {type(error).__name__}: {str(error)[:300]}"
        ]
        record["confidence_or_uncertainty"] = "Unclassified because the PDF could not be read."
    return record


def main() -> None:
    if not PDF_DIR.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")
    pdf_paths = sorted(
        (path for path in PDF_DIR.iterdir() if path.is_file() and path.suffix.casefold() == ".pdf"),
        key=lambda path: path.name.casefold(),
    )
    documents = [inventory_pdf(path) for path in pdf_paths]
    inventory = {
        "inventory_version": 1,
        "source_directory": "data/raw",
        "pdf_count": len(documents),
        "documents": documents,
    }
    OUTPUT_PATH.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"PDFs found: {len(documents)}")
    for item in documents:
        pages = item["page_count"] if item["page_count"] is not None else "?"
        print(
            f"{item['filename']} | pages={pages} | bytes={item['file_size_bytes']} | "
            f"chars~{item['extracted_character_count']} | {item['extraction_status']} | "
            f"{item['topic_category']} | {item['source_suitability']}"
        )
    print(f"Inventory written to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
