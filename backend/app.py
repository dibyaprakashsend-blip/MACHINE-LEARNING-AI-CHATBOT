from pathlib import Path
import os

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

from fastembed import TextEmbedding


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

env_file = BASE_DIR / ".env"

if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY was not found. "
        "Add GOOGLE_API_KEY to your .env file locally "
        "or Render Environment Variables in production."
    )


# ============================================================
# 2. CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI College Study Assistant",
    description="AI College Study Assistant using RAG, Chroma, FastEmbed and Gemini",
    version="1.0.0"
)


# ============================================================
# 3. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 4. FASTEMBED EMBEDDING CLASS
# ============================================================

class FastEmbedEmbeddings(Embeddings):

    def __init__(self):

        print("\nLoading FastEmbed model...")

        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        print("FastEmbed model loaded successfully!")

    def embed_documents(self, texts):

        embeddings = self.model.embed(texts)

        return [
            embedding.tolist()
            for embedding in embeddings
        ]

    def embed_query(self, text):

        embedding = list(
            self.model.embed([text])
        )[0]

        return embedding.tolist()


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

embedding_model = FastEmbedEmbeddings()


# ============================================================
# 6. LOAD CHROMA DATABASE
# ============================================================

CHROMA_PATH = BASE_DIR / "chroma_database"


if not CHROMA_PATH.exists():

    raise RuntimeError(
        "Chroma database was not found.\n"
        "Run this command locally first:\n\n"
        "python vectorstore/database.py"
    )


print("\nLoading Chroma database...")

vectorstore = Chroma(
    collection_name="college_study_documents",
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding_model
)

print("Chroma database loaded successfully!")


# ============================================================
# 7. CHECK CHROMA DATABASE
# ============================================================

try:

    collection_count = vectorstore._collection.count()

    print(
        f"Chroma database contains "
        f"{collection_count} documents/chunks."
    )

except Exception as e:

    print(
        "Could not determine Chroma collection count:",
        str(e)
    )


# ============================================================
# 8. CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# ============================================================
# 9. GEMINI LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# 10. RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI College Study Assistant.

Answer the student's question using ONLY the information
provided in the document context below.

Rules:

1. Use only the provided document context.
2. Do not invent information.
3. If the answer is not present in the context, reply exactly:

I could not find the answer in the document.

4. Explain the answer clearly and simply.
5. Use bullet points when useful.
6. For educational questions, provide enough explanation
   for a college student to understand the topic.

DOCUMENT CONTEXT:

{context}
"""
        ),
        (
            "human",
            "{question}"
        )
    ]
)


# ============================================================
# 11. HELPER FUNCTION
# ============================================================

def normalize_answer(content):

    """
    Gemini/LangChain can sometimes return different
    content formats. This function converts them into
    a normal string.
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):

                parts.append(item)

            elif isinstance(item, dict):

                if "text" in item:
                    parts.append(str(item["text"]))

                elif "content" in item:
                    parts.append(str(item["content"]))

                else:
                    parts.append(str(item))

            else:

                parts.append(str(item))

        return "\n".join(parts)

    if isinstance(content, dict):

        if "text" in content:
            return str(content["text"])

        if "content" in content:
            return str(content["content"])

        return str(content)

    return str(content)


# ============================================================
# 12. HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI College Study Assistant API is running!"
    }


# ============================================================
# 13. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    try:

        count = vectorstore._collection.count()

    except Exception:

        count = "unknown"

    return {
        "status": "healthy",
        "service": "AI College Study Assistant",
        "vector_database": "Chroma",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "llm": "Gemini 3.6 Flash",
        "document_count": count
    }


# ============================================================
# 14. CHAT API
# ============================================================

@app.post("/api/chat")
async def chat(data: dict):

    try:

        # ----------------------------------------------------
        # GET USER QUESTION
        # ----------------------------------------------------

        user_message = data.get("message", "")

        if not user_message:

            return {
                "error": "Please provide a message."
            }


        print("\n")
        print("=" * 60)
        print("NEW CHAT REQUEST")
        print("=" * 60)

        print("User question:")
        print(user_message)


        # ----------------------------------------------------
        # CHROMA COUNT
        # ----------------------------------------------------

        try:

            chroma_count = vectorstore._collection.count()

        except Exception:

            chroma_count = "unknown"


        print("\nChroma document count:")
        print(chroma_count)


        # ----------------------------------------------------
        # RETRIEVE DOCUMENTS
        # ----------------------------------------------------

        print("\nSearching Chroma database...")

        docs = retriever.invoke(user_message)


        # ----------------------------------------------------
        # RAG DEBUGGING
        # ----------------------------------------------------

        print("\n")
        print("=" * 60)
        print("RAG DEBUG")
        print("=" * 60)

        print(
            "Chroma document count:",
            chroma_count
        )

        print(
            "Retrieved documents:",
            len(docs)
        )


        for i, doc in enumerate(docs):

            print("\n")
            print("-" * 60)

            print(
                f"Retrieved document {i + 1}"
            )

            print("-" * 60)

            print(
                doc.page_content[:1000]
            )

            print("\nMetadata:")

            print(
                doc.metadata
            )


        print("=" * 60)
        print("END RAG DEBUG")
        print("=" * 60)


        # ----------------------------------------------------
        # NO DOCUMENTS FOUND
        # ----------------------------------------------------

        if not docs:

            print(
                "\nWARNING: No documents were retrieved."
            )

            return {
                "response":
                    "I could not find the answer in the document."
            }


        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context_parts = []

        for i, doc in enumerate(docs):

            context_parts.append(
                f"""
--- Document Chunk {i + 1} ---

{doc.page_content}
"""
            )


        context = "\n".join(context_parts)


        # ----------------------------------------------------
        # DEBUG CONTEXT LENGTH
        # ----------------------------------------------------

        print("\nContext length:")

        print(
            len(context),
            "characters"
        )


        # ----------------------------------------------------
        # CREATE PROMPT
        # ----------------------------------------------------

        messages = prompt.format_messages(
            context=context,
            question=user_message
        )


        # ----------------------------------------------------
        # CALL GEMINI
        # ----------------------------------------------------

        print("\nCalling Gemini...")

        result = llm.invoke(messages)


        # ----------------------------------------------------
        # NORMALIZE GEMINI RESPONSE
        # ----------------------------------------------------

        answer = normalize_answer(
            result.content
        ).strip()


        print("\nGemini response:")

        print(answer)


        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if not answer:

            answer = (
                "I could not find the answer in the document."
            )


        print("\n")
        print("=" * 60)
        print("REQUEST COMPLETED")
        print("=" * 60)


        # ----------------------------------------------------
        # RETURN RESPONSE
        # ----------------------------------------------------

        return {
            "response": answer
        }


    except Exception as e:

        print("\n")
        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print(str(e))

        print("=" * 60)


        return {
            "error": str(e),
            "response":
                "Sorry, something went wrong while processing your question."
        }


# ============================================================
# 15. STARTUP INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("       AI COLLEGE STUDY ASSISTANT")
print("=" * 60)

print("Backend      : FastAPI")
print("Vector DB    : Chroma")
print("Embeddings   : FastEmbed")
print("Embedding    : BAAI/bge-small-en-v1.5")
print("LLM          : Gemini 3.6 Flash")

print("=" * 60)