from pathlib import Path

from dotenv import load_dotenv

from fastembed import TextEmbedding

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_FILE = BASE_DIR / "documents" / "ml_book.pdf"
CHROMA_PATH = BASE_DIR / "chroma_database"

ENV_FILE = BASE_DIR / ".env"


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(ENV_FILE)


# ============================================================
# START
# ============================================================

print()
print("=" * 60)
print("       AI COLLEGE STUDY ASSISTANT")
print("=" * 60)
print()

print("Embedding system : FastEmbed")
print("Embedding model  : BAAI/bge-small-en-v1.5")
print()


# ============================================================
# LOAD PDF
# ============================================================

print("Loading PDF...")
print()

loader = PyPDFLoader(
    str(PDF_FILE)
)

docs = loader.load()

print("PDF loaded successfully!")
print("Number of pages:", len(docs))
print()


# ============================================================
# SPLIT PDF
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(
    docs
)

print("Number of chunks:", len(chunks))
print()


# ============================================================
# CREATE FASTEMBED MODEL
# ============================================================

print("Loading embedding model...")
print()

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

print("Embedding model loaded successfully!")
print()


# ============================================================
# CREATE CHROMA DATABASE
# ============================================================

print("Creating Chroma database...")
print()


# Convert documents into plain text
texts = [
    document.page_content
    for document in chunks
]


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("Creating embeddings...")
print()

embeddings = []

total = len(texts)

for start in range(
    0,
    total,
    32
):

    end = min(
        start + 32,
        total
    )

    batch = texts[start:end]

    print(
        f"Embedding chunks "
        f"{start + 1} - {end} "
        f"of {total}..."
    )

    batch_embeddings = list(
        embedding_model.embed(
            batch
        )
    )

    embeddings.extend(
        [
            embedding.tolist()
            for embedding in batch_embeddings
        ]
    )


print()
print("All embeddings created successfully!")
print()


# ============================================================
# CREATE CHROMA
# ============================================================

vectorstore = Chroma(
    collection_name="college_study_documents",
    embedding_function=None,
    persist_directory=str(CHROMA_PATH)
)


# ============================================================
# ADD DOCUMENTS + EMBEDDINGS
# ============================================================

print("Adding documents to Chroma...")
print()


ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]


metadatas = [
    document.metadata
    for document in chunks
]


vectorstore.add_texts(
    texts=texts,
    ids=ids,
    metadatas=metadatas
)

print("\nAdding documents to Chroma...")

collection = vectorstore._collection

collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas
)

print("\nAll documents added successfully!")
print(f"Total documents in Chroma: {collection.count()}")
print("\nChroma database created successfully!")


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("   CHROMA DATABASE CREATED SUCCESSFULLY!")
print("=" * 60)
print()

print("PDF pages       :", len(docs))
print("Total chunks    :", len(chunks))
print("Embedding model :", "BAAI/bge-small-en-v1.5")
print("Vector database :", CHROMA_PATH)
print()

print("RAG database is ready!")
print()