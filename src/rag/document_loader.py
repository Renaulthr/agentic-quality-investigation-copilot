from pathlib import Path
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENT_DIR = PROJECT_ROOT / "data" / "documents"


def load_markdown_documents() -> list[Document]:
    """Load synthetic quality documents from the knowledge base."""

    if not DOCUMENT_DIR.exists():
        raise FileNotFoundError(
            f"Document directory not found: {DOCUMENT_DIR}"
        )

    documents = []

    for file_path in sorted(DOCUMENT_DIR.glob("*.md")):
        content = file_path.read_text(
            encoding="utf-8"
        )

        if not content.strip():
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "document_type": file_path.stem,
                },
            )
        )

    if not documents:
        raise ValueError(
            f"No Markdown documents found in {DOCUMENT_DIR}"
        )

    return documents


if __name__ == "__main__":
    docs = load_markdown_documents()

    print(f"\nLoaded documents: {len(docs)}\n")

    for doc in docs:
        print(
            f"{doc.metadata['source']:<40} "
            f"{len(doc.page_content):>6} characters"
        )