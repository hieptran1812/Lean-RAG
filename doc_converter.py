import os
from pathlib import Path
from docling.document_converter import DocumentConverter

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
OUTPUT_DIR = BASE_DIR / "markdown_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

converter = DocumentConverter()

for doc_path in DOCUMENTS_DIR.iterdir():
    if doc_path.suffix.lower() in (".docx", ".pdf", ".pptx", ".html", ".xlsx"):
        print(f"Converting: {doc_path.name}")
        try:
            result = converter.convert(str(doc_path))
            markdown_content = result.document.export_to_markdown()

            md_filename = doc_path.stem + ".md"
            output_path = OUTPUT_DIR / md_filename
            output_path.write_text(markdown_content, encoding="utf-8")
            print(f"  -> Saved to {output_path}")
        except Exception as e:
            print(f"  -> Failed: {e}")

print("\nDone. All markdown files saved to:", OUTPUT_DIR)