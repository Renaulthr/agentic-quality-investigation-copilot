from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from src.rag.document_loader import (
    load_markdown_documents,
)


CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def create_chunks():
    """Split quality documents into retrieval-ready chunks."""

    documents = load_markdown_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = (
            f"chunk-{index:04d}"
        )

    return chunks


if __name__ == "__main__":
    chunks = create_chunks()

    print(f"\nTotal chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "=" * 70)
        print(
            "Chunk:",
            chunk.metadata["chunk_id"],
        )
        print(
            "Source:",
            chunk.metadata["source"],
        )
        print(
            "Characters:",
            len(chunk.page_content),
        )
        print("-" * 70)
        print(chunk.page_content[:300])