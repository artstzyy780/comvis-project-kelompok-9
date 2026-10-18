"""Script to convert the UTS Proposal Markdown document to a professional PDF.

This module reads the markdown proposal file, processes markdown elements (including
tables, math notation, and callouts), wraps it in academic styling CSS, and compiles
it to PDF using Playwright (Microsoft Edge headless browser).
"""

from pathlib import Path
from typing import Final
import mistune
from playwright.sync_api import sync_playwright

# Constants for file paths and document parameters
MARKDOWN_PATH: Final[Path] = Path("docs/proposal/K09_DeteksiPedestrianLowLight_Proposal_v01.md")
OUTPUT_PDF_PATH: Final[Path] = Path("docs/proposal/K09_DeteksiPedestrianLowLight_Proposal_v01.pdf")
ROOT_PDF_PATH: Final[Path] = Path("K09_DeteksiPedestrianLowLight_Proposal_v01.pdf")

HTML_TEMPLATE: Final[str] = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Proposal Proyek Visi Komputer - Kelompok 9</title>
    <!-- KaTeX for math rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ]
        });"></script>
    <style>
        @page {
            size: A4;
            margin: 18mm 16mm 18mm 16mm;
        }

        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
            font-size: 8.8pt;
            line-height: 1.42;
            color: #1a1a1a;
            margin: 0;
            padding: 0;
            text-align: justify;
        }

        /* Title & Header Section */
        .title-block {
            text-align: center;
            border-bottom: 2px solid #1e3a8a;
            padding-bottom: 10px;
            margin-bottom: 14px;
        }

        .title-block h1 {
            font-size: 14.5pt;
            color: #1e3a8a;
            margin: 0 0 5px 0;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            font-weight: 700;
        }

        .title-block h2 {
            font-size: 11pt;
            color: #2563eb;
            margin: 0 0 6px 0;
            font-weight: 600;
            border-bottom: none;
            padding-bottom: 0;
        }

        .title-block .meta-info {
            font-size: 8.5pt;
            color: #4b5563;
            margin: 2px 0;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #1e3a8a;
            page-break-after: avoid;
            font-weight: 600;
        }

        h1 {
            font-size: 13.5pt;
            margin-top: 14px;
            margin-bottom: 6px;
            border-bottom: 2px solid #1e3a8a;
            padding-bottom: 3px;
        }

        h2 {
            font-size: 11pt;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 2px;
            margin-top: 12px;
            margin-bottom: 6px;
        }

        h3 {
            font-size: 9.8pt;
            color: #1e40af;
            margin-top: 10px;
            margin-bottom: 4px;
        }

        h4 {
            font-size: 9pt;
            margin-top: 8px;
            margin-bottom: 3px;
        }

        p {
            margin: 0 0 5px 0;
        }

        /* Lists */
        ul, ol {
            margin: 0 0 6px 0;
            padding-left: 18px;
        }

        li {
            margin-bottom: 2px;
        }

        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 8px 0 10px 0;
            font-size: 8pt;
            page-break-inside: avoid;
        }

        th, td {
            border: 1px solid #cbd5e1;
            padding: 4px 6px;
            vertical-align: middle;
        }

        th {
            background-color: #f1f5f9;
            color: #0f172a;
            font-weight: 600;
            text-align: center;
        }

        tr:nth-child(even) {
            background-color: #f8fafc;
        }

        /* Preformatted & Code Blocks */
        pre {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            padding: 6px 8px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 7.5pt;
            line-height: 1.3;
            overflow-x: auto;
            page-break-inside: avoid;
            margin: 6px 0;
        }

        code {
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 8pt;
            background-color: #f1f5f9;
            padding: 1px 3px;
            border-radius: 3px;
            color: #0f172a;
        }

        /* Blockquotes / Callout */
        blockquote {
            border-left: 3px solid #2563eb;
            background-color: #eff6ff;
            margin: 6px 0;
            padding: 5px 10px;
            font-style: italic;
            color: #1e3a8a;
            page-break-inside: avoid;
        }

        /* Horizontal rule */
        hr {
            border: 0;
            height: 1px;
            background-color: #cbd5e1;
            margin: 10px 0;
        }
    </style>
</head>
<body>
{content}
</body>
</html>
"""


def build_proposal_pdf() -> None:
    """Reads the proposal markdown, formats it to HTML, and exports it to PDF via Playwright."""
    if not MARKDOWN_PATH.exists():
        raise FileNotFoundError(f"Markdown proposal not found at {MARKDOWN_PATH}")

    # Read markdown content
    md_text: str = MARKDOWN_PATH.read_text(encoding="utf-8")

    # Set up mistune markdown parser with plugins
    markdown_parser = mistune.create_markdown(plugins=["table", "strikethrough"])
    rendered_body_html: str = markdown_parser(md_text)

    # Format full HTML document
    full_html: str = HTML_TEMPLATE.replace("{content}", rendered_body_html)

    temp_html_path: Path = Path("docs/proposal/temp_proposal.html")
    temp_html_path.write_text(full_html, encoding="utf-8")

    # Launch Playwright to generate PDF
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()

        # Load local HTML file
        page.goto(f"file:///{temp_html_path.resolve().as_posix()}", wait_until="networkidle")

        # Allow KaTeX math to settle
        page.wait_for_timeout(1500)

        # PDF print options
        header_html = """
        <div style="font-size: 8pt; font-family: 'Segoe UI', Arial, sans-serif; width: 100%; margin: 0 16mm; display: flex; justify-content: space-between; color: #64748b; border-bottom: 1px solid #e2e8f0; padding-bottom: 3px;">
            <span>Visi Komputer (IF0PP006) — Proposal Proyek UTS</span>
            <span>Kelas 5PITA • Semester Ganjil 2026/2027</span>
        </div>
        """

        footer_html = """
        <div style="font-size: 8pt; font-family: 'Segoe UI', Arial, sans-serif; width: 100%; margin: 0 16mm; display: flex; justify-content: space-between; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 3px;">
            <span>Kelompok 9: Deteksi Pedestrian Low-Light</span>
            <span>Halaman <span class="pageNumber"></span> dari <span class="totalPages"></span></span>
        </div>
        """

        pdf_bytes = page.pdf(
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template=header_html,
            footer_template=footer_html,
            margin={
                "top": "20mm",
                "bottom": "20mm",
                "left": "16mm",
                "right": "16mm",
            },
        )

        OUTPUT_PDF_PATH.write_bytes(pdf_bytes)
        ROOT_PDF_PATH.write_bytes(pdf_bytes)

        browser.close()

    if temp_html_path.exists():
        temp_html_path.unlink()

    # Use ASCII safe print
    print(f"[OK] Berhasil membuat PDF proposal di:\n  - {OUTPUT_PDF_PATH}\n  - {ROOT_PDF_PATH}")


if __name__ == "__main__":
    build_proposal_pdf()
