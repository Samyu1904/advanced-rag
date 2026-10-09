from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import TextSplitter

PDF_PATH = PROJECT_ROOT / "data" / "documents" / "air-quality-and-health.pdf"

if not PDF_PATH.exists():
    print(f"PDF not found: {PDF_PATH}")
    print("Upload the PDF in Streamlit first or copy it to data/documents.")
    raise SystemExit(1)

documents = PDFLoader().load_pdf(str(PDF_PATH))
chunks = TextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)

print("=" * 80)
print("RAG EVALUATION - CHUNK INSPECTION")
print("=" * 80)
print(f"Pages: {len(documents)}")
print(f"Chunks: {len(chunks)}")

for index, chunk in enumerate(chunks):
    print("\n" + "=" * 80)
    print(f"CHUNK ID: chunk_{index}")
    print("=" * 80)
    print(f"Metadata: {getattr(chunk, 'metadata', {})}")
    print(getattr(chunk, "page_content", str(chunk))[:1500])
