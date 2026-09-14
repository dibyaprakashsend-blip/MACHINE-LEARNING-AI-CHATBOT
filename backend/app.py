import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI College Study Assistant",
    description="AI Study Assistant using RAG, Chroma and Gemini",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRONTEND_PATH = os.path.join(
    BASE_DIR,
    "frontend",
    "index.html"
)

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "chroma_database"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("=" * 60)
print("Loading FastEmbed model...")
print("=" * 60)

embedding_model = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

print("FastEmbed model loaded successfully!")


# ============================================================
# CHROMA VECTOR DATABASE
# ============================================================

print("=" * 60)
print("Loading Chroma database...")
print("=" * 60)

vectorstore = Chroma(
    collection_name="college_study_documents",
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)

print("Chroma database loaded successfully!")


# ============================================================
# GEMINI
# ============================================================

print("=" * 60)
print("Loading Gemini...")
print("=" * 60)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY
)

print("Gemini loaded successfully!")


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str


# ============================================================
# FRONTEND
# ============================================================

@app.get("/", include_in_schema=False)
def frontend():

    if not os.path.exists(FRONTEND_PATH):
        return {
            "error": "frontend/index.html was not found",
            "expected_path": FRONTEND_PATH
        }

    return FileResponse(FRONTEND_PATH)


# ============================================================
# HEALTH CHECK
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
# CHAT API
# ============================================================

@app.post("/api/chat")
async def chat(data: ChatRequest):

    try:

        user_message = data.message.strip()

        if not user_message:
            return {
                "response": "Please enter a question."
            }

        print("\n")
        print("=" * 60)
        print("NEW QUESTION")
        print("=" * 60)
        print("Question:", user_message)


        # ------------------------------------------------------
        # DATABASE COUNT
        # ------------------------------------------------------

        try:
            chroma_count = vectorstore._collection.count()
        except Exception:
            chroma_count = "unknown"


        # ------------------------------------------------------
        # RETRIEVE RELEVANT DOCUMENTS
        # ------------------------------------------------------

        docs = vectorstore.similarity_search(
            user_message,
            k=8
        )


        # ------------------------------------------------------
        # RAG DEBUG
        # ------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("RAG DEBUG")
        print("=" * 60)

        print("Chroma document count:", chroma_count)
        print("Retrieved documents:", len(docs))


        for i, doc in enumerate(docs):

            print("\n")
            print("-" * 60)
            print(f"Retrieved document {i + 1}")
            print("-" * 60)

            print(doc.page_content[:1000])

            print("\nMetadata:")
            print(doc.metadata)


        print("=" * 60)
        print("END RAG DEBUG")
        print("=" * 60)


        # ------------------------------------------------------
        # NO DOCUMENTS
        # ------------------------------------------------------

        if not docs:

            return {
                "response": "I could not find relevant information in the document."
            }


        # ------------------------------------------------------
        # CREATE CONTEXT
        # ------------------------------------------------------

        context_parts = []

        for doc in docs:

            page_number = doc.metadata.get(
                "page_label",
                doc.metadata.get("page", "unknown")
            )

            context_parts.append(
                f"""
SOURCE PAGE: {page_number}

{doc.page_content}
"""
            )


        context = "\n".join(context_parts)


        print("\n")
        print("=" * 60)
        print("CONTEXT LENGTH")
        print("=" * 60)

        print("Context length:", len(context), "characters")


        # ------------------------------------------------------
        # PROMPT
        # ------------------------------------------------------

        prompt = f"""
You are an AI College Study Assistant.

Answer the student's question using the provided document context.

Rules:

1. Use the document context as your primary source.
2. Give a clear and beginner-friendly answer.
3. Explain difficult concepts simply.
4. Do not invent information.
5. If the exact answer is present, answer it directly.
6. If related information is present, use it to explain the concept.
7. Mention the relevant page number when possible.
8. Do not unnecessarily say that the answer was not found.

STUDENT QUESTION:

{user_message}

DOCUMENT CONTEXT:

{context}

Now provide a clear answer suitable for a college student.
"""


        # ------------------------------------------------------
        # GEMINI
        # ------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("Calling Gemini...")
        print("=" * 60)

        response = llm.invoke(prompt)

        answer = response.content


        # ------------------------------------------------------
        # NORMALIZE RESPONSE
        # ------------------------------------------------------

        if isinstance(answer, list):

            text_parts = []

            for item in answer:

                if isinstance(item, str):
                    text_parts.append(item)

                elif isinstance(item, dict):

                    if "text" in item:
                        text_parts.append(str(item["text"]))

                    elif "content" in item:
                        text_parts.append(str(item["content"]))

                    else:
                        text_parts.append(str(item))

                else:
                    text_parts.append(str(item))

            answer = "\n".join(text_parts)

        elif not isinstance(answer, str):

            answer = str(answer)


        # ------------------------------------------------------
        # RESPONSE DEBUG
        # ------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("Gemini response:")
        print("=" * 60)

        print(answer)

        print("=" * 60)
        print("REQUEST COMPLETED")
        print("=" * 60)


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
            "response": "Sorry, something went wrong while processing your question.",
            "error": str(e)
        }