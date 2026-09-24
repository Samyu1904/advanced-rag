from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:

    def load_pdf(self, pdf_path: str):

        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "The provided file must be a PDF."
            )

        loader = PyPDFLoader(str(path))

        documents = loader.load()

        print(f"PDF loaded successfully!")
        print(f"Number of pages: {len(documents)}")

        return documents