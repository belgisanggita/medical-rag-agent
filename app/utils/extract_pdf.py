"""
PDF extraction & structure-aware chunking for the Medical Encyclopedia PDF.

Boundary detection is driven entirely by font metadata that was verified
against the actual source file (Gale Encyclopedia of Medicine, 637 pages,
two-column layout) - NOT by regex on raw text and NOT by fixed
character-count windows. Verified rules:

  15.0pt "Optima-Bold"          -> new encyclopedia entry title
  11.0pt "Optima-Bold"          -> subheading within the current entry
                                    (Definition, Purpose, Description, ...)
  12.5pt "Optima-Bold"          -> "KEY TERMS" glossary box header
  10.0pt "Optima-Bold" (inside
         a KEY TERMS box)       -> one glossary term label
  12.0pt (any style)            -> "X see Y" cross-reference noise
  48.0pt single-char "Optima-Bold" -> alphabet drop-cap divider, not content

  pages 0-13 (0-indexed)  -> front matter (cover/copyright/advisory board),
                             no entries at all
  page 636 (0-indexed)    -> "ORGANIZATIONS" appendix, different structure

Because splits follow the document's own structure instead of a character
count, a chunk never gets cut mid-sentence or mid-topic - the only place a
hard character split can happen is the MAX_SECTION_CHARS fallback below,
and even then it never crosses a section/entry boundary.
"""

import re
import uuid

import pdfplumber

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

PAGE_LOG_INTERVAL = 50
ENTRY_LOG_INTERVAL = 25

FIRST_CONTENT_PAGE = 14   # 0-indexed page where the first real entry ("Abdominal ultrasound") starts
LAST_CONTENT_PAGE = 635   # 0-indexed; page 636 is the ORGANIZATIONS appendix and is dropped

TITLE_SIZE = 15.0
SUBHEADING_SIZE = 11.0
KEY_TERMS_SIZE = 12.5
GLOSSARY_TERM_SIZE = 10.0
REDIRECT_SIZE = 12.0
DROPCAP_SIZE = 48.0

MIN_SECTION_CHARS = 200     # sections shorter than this are folded into the previous one
MAX_SECTION_CHARS = 1800    # fallback split point for unusually long sections
OVERLAP_CHARS = 150         # only used by the fallback splitter - never crosses a section boundary


def _is_structural_bold(fontname: str) -> bool:
    """
    True only for the specific "Optima-Bold" face used by titles, subheadings,
    KEY TERMS headers and glossary-term labels. Body text also uses other bold
    faces (e.g. "Times-Bold") for inline emphasis on ordinary words - matching
    on "Bold" alone misreads those as structural boundaries (verified: e.g. a
    Times-Bold "pneumonia" mid-sentence was wrongly read as a new glossary term).
    """
    return "Optima-Bold" in fontname


def _keep_char(obj) -> bool:
    """Drop sideways running-header glyphs and out-of-page-bounds print-shop stamps."""
    if obj.get("object_type") != "char":
        return True
    if not obj.get("upright", True):
        return False
    return 0 <= obj["top"] <= obj["page_height"]


def extract_text_per_page(pdf_path: str):
    """
    Returns one entry per retained content page:
        {"page_number": <1-indexed>, "words": [{"text", "size", "fontname", "x0", "top"}, ...]}

    Front matter and the closing appendix are dropped here so the boundary
    detector in chunk_text() never has to special-case non-entry pages.
    Word order is column-aware (left column top-to-bottom, then right
    column) instead of raw PDF coordinate order, which would interleave
    the two columns line-by-line.
    """
    total_pages = LAST_CONTENT_PAGE - FIRST_CONTENT_PAGE + 1
    logger.info(f"Opening PDF and reading {total_pages} content pages ({pdf_path})...")

    pages_out = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, idx in enumerate(range(FIRST_CONTENT_PAGE, LAST_CONTENT_PAGE + 1), start=1):
            page = pdf.pages[idx]
            for c in page.chars:
                c["page_height"] = page.height
            filtered = page.filter(_keep_char)
            words = filtered.extract_words(extra_attrs=["size", "fontname"])

            mid_x = page.width / 2
            words.sort(key=lambda w: (w["x0"] >= mid_x, w["top"], w["x0"]))

            pages_out.append({"page_number": idx + 1, "words": words})

            if i % PAGE_LOG_INTERVAL == 0 or i == total_pages:
                logger.info(f"Extracted page {i}/{total_pages} ({100 * i // total_pages}%)")

    logger.info(f"Finished extracting {len(pages_out)} pages.")
    return pages_out


def _join(words) -> str:
    text = " ".join(w["text"] for w in words)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_break(text: str, pos: int, lookback: int = 100) -> int:
    """Nearest whitespace at/before `pos` (searching back up to `lookback`
    chars) so a split lands between words instead of inside one. Falls back
    to `pos` itself in the rare case of no whitespace in that window."""
    window_start = max(0, pos - lookback)
    ws = text.rfind(" ", window_start, pos)
    return ws + 1 if ws != -1 else pos


def _split_long_text(text: str):
    """Fallback for a single section/term that's unusually long. Splits on
    character count with overlap at word boundaries, but is only ever
    called on text that already belongs to one section - it never merges
    or crosses sections."""
    if len(text) <= MAX_SECTION_CHARS:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + MAX_SECTION_CHARS, len(text))
        if end < len(text):
            end = _find_break(text, end)
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        next_start = _find_break(text, max(start + 1, end - OVERLAP_CHARS))
        start = next_start if next_start > start else end
    return parts


def _flush_entry(entry_title, sections, page_start, page_end, chunks):
    """Merge too-short sections into their previous one, then emit final chunks."""
    if not entry_title or not sections:
        return

    merged = []
    for sec in sections:
        text = _join(sec["words"])
        if not text:
            continue
        if merged and len(text) < MIN_SECTION_CHARS:
            merged[-1]["name"] = f"{merged[-1]['name']} / {sec['name']}"
            merged[-1]["text"] = f"{merged[-1]['text']} {text}"
        else:
            merged.append({"name": sec["name"], "text": text})

    for sec in merged:
        for part in _split_long_text(sec["text"]):
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": f"{entry_title} - {sec['name']}: {part}",
                "metadata": {
                    "type": "content",
                    "entry": entry_title,
                    "section": sec["name"],
                    "page_start": page_start,
                    "page_end": page_end,
                },
            })


def _flush_glossary_term(entry_title, term_name, term_words, page_number, chunks):
    text = _join(term_words)
    if not term_name or not text:
        return
    full = f"{term_name} - {text}"
    for part in _split_long_text(full):
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": part,
            "metadata": {
                "type": "glossary",
                "entry": entry_title,
                "term": term_name,
                "page_start": page_number,
                "page_end": page_number,
            },
        })


def chunk_text(pages):
    """
    Walks the (already column-ordered) word stream for the whole document
    and turns it into RAG-ready chunks by tracking document structure:
    one chunk per (entry, subheading) pair, plus one chunk per KEY TERMS
    glossary definition. Redirect ("X see Y") lines and drop-cap dividers
    are recognized and discarded, never indexed.
    """
    chunks = []

    entry_title = None
    entry_page_start = None
    entry_page_end = None
    sections = []          # [{"name": str, "words": [...]}] for the current entry
    current_section = None

    in_key_terms = False
    term_name_words = []
    term_words = []
    current_term_name = None

    # run-accumulation state for multi-word titles/subheadings
    run_kind = None        # "title" | "subheading" | "key_terms_header" | "glossary_term"
    run_words = []

    entry_count = 0

    def start_section(name):
        nonlocal current_section
        current_section = {"name": name, "words": []}
        sections.append(current_section)

    def close_run():
        nonlocal run_kind, run_words, entry_title, current_term_name
        if run_kind is None or not run_words:
            run_kind, run_words = None, []
            return
        text = _join(run_words)
        if run_kind == "title":
            entry_title = text
        elif run_kind == "subheading":
            start_section(text)
        elif run_kind == "key_terms_header":
            pass  # the literal "KEY TERMS" label itself carries no content
        elif run_kind == "glossary_term":
            current_term_name = text
        run_kind, run_words = None, []

    for page_idx, page in enumerate(pages):
        page_no = page["page_number"]
        for w in page["words"]:
            size = round(w["size"], 1)
            bold = _is_structural_bold(w["fontname"])

            # --- pure noise: drop-cap divider letter and redirect lines ---
            if size == DROPCAP_SIZE:
                continue
            if size == REDIRECT_SIZE:
                continue

            # --- new entry title ---
            if size == TITLE_SIZE and bold:
                if run_kind != "title":
                    close_run()
                    _flush_entry(entry_title, sections, entry_page_start, entry_page_end, chunks)
                    if entry_title:
                        entry_count += 1
                        if entry_count % ENTRY_LOG_INTERVAL == 0:
                            logger.info(f"Chunked {entry_count} entries so far (last: '{entry_title}', page {entry_page_end})")
                    if in_key_terms:
                        _flush_glossary_term(entry_title, current_term_name, term_words, page_no, chunks)
                    entry_title, sections, current_section = None, [], None
                    in_key_terms = False
                    current_term_name, term_words = None, []
                    entry_page_start = page_no
                    run_kind = "title"
                run_words.append(w)
                entry_page_end = page_no
                continue

            # --- subheading label ---
            if size == SUBHEADING_SIZE and bold:
                if run_kind != "subheading":
                    close_run()
                    if in_key_terms:
                        _flush_glossary_term(entry_title, current_term_name, term_words, page_no, chunks)
                        in_key_terms = False
                        current_term_name, term_words = None, []
                    run_kind = "subheading"
                run_words.append(w)
                entry_page_end = page_no
                continue

            # --- KEY TERMS box header ---
            if size == KEY_TERMS_SIZE and bold:
                if run_kind != "key_terms_header":
                    close_run()
                    in_key_terms = True
                    current_term_name, term_words = None, []
                    run_kind = "key_terms_header"
                run_words.append(w)
                entry_page_end = page_no
                continue

            # --- "Resources" bibliography subheading shares the glossary-term font/size
            # (10pt Optima-Bold) but is a section boundary, not a term - verified against
            # 248 occurrences in the source PDF, all exactly this label. ---
            if bold and size == GLOSSARY_TERM_SIZE and w["text"].strip(":") == "Resources":
                if run_kind != "subheading":
                    close_run()
                    if in_key_terms:
                        _flush_glossary_term(entry_title, current_term_name, term_words, page_no, chunks)
                        in_key_terms = False
                        current_term_name, term_words = None, []
                    run_kind = "subheading"
                run_words.append(w)
                entry_page_end = page_no
                continue

            # --- inside a KEY TERMS box: bold 10pt run = new glossary term label ---
            if in_key_terms and size == GLOSSARY_TERM_SIZE and bold:
                if run_kind != "glossary_term":
                    close_run()
                    _flush_glossary_term(entry_title, current_term_name, term_words, page_no, chunks)
                    current_term_name, term_words = None, []
                    run_kind = "glossary_term"
                run_words.append(w)
                entry_page_end = page_no
                continue

            # --- regular body text ---
            close_run()
            entry_page_end = page_no
            if in_key_terms:
                term_words.append(w)
            elif current_section is not None:
                current_section["words"].append(w)
            elif entry_title is not None:
                # body text before the first subheading (entry intro paragraph)
                start_section("Overview")
                current_section["words"].append(w)

        # --- end of page: does the KEY TERMS box actually continue onto the
        # next page, or does normal body text resume here? The PDF gives no
        # font signal for "box ended" - a box that fills a whole page (verified:
        # page 17 of the source PDF) is followed by plain body text with zero
        # bold markers, which would otherwise get silently swallowed as more
        # of the last term's definition. So: close the box here unless the
        # very next page's first word is itself a new term label. ---
        if in_key_terms:
            close_run()
            next_words = pages[page_idx + 1]["words"] if page_idx + 1 < len(pages) else []
            box_continues = bool(next_words) and (
                round(next_words[0]["size"], 1) == GLOSSARY_TERM_SIZE
                and _is_structural_bold(next_words[0]["fontname"])
            )
            if not box_continues:
                _flush_glossary_term(entry_title, current_term_name, term_words, page_no, chunks)
                in_key_terms = False
                current_term_name, term_words = None, []

    close_run()
    if in_key_terms:
        _flush_glossary_term(entry_title, current_term_name, term_words, entry_page_end, chunks)
    _flush_entry(entry_title, sections, entry_page_start, entry_page_end, chunks)
    if entry_title:
        entry_count += 1

    content_count = sum(1 for c in chunks if c["metadata"]["type"] == "content")
    glossary_count = len(chunks) - content_count
    logger.info(
        f"Chunking done: {entry_count} entries -> {len(chunks)} chunks "
        f"({content_count} content, {glossary_count} glossary)"
    )

    return chunks
