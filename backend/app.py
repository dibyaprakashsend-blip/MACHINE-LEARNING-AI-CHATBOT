from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# LOAD .ENV
# ============================================================

load_dotenv(BASE_DIR / ".env")


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
# EMBEDDING MODEL
# ============================================================

# IMPORTANT:
# This must be the SAME model that was used to create
# your Chroma database in vectorstore/database.py.
#
# all-MiniLM-L6-v2 = 384 dimensions

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CHROMA VECTOR DATABASE
# ============================================================

CHROMA_PATH = BASE_DIR / "chroma_database"

vectorstore = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding_model
)


# ============================================================
# RETRIEVER
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
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI College Study Assistant.

Answer the student's question using ONLY the provided
context from the uploaded study document.

Rules:

1. Use only information available in the context.
2. Do not invent or make up information.
3. If the answer is not available in the context,
   say exactly:

"I could not find the answer in the document."

4. Give clear and simple explanations.
5. If the question asks for an explanation,
   explain it step by step when possible.
6. If the question asks for a definition,
   give a clear definition based on the context.
"""
        ),
        (
            "human",
            """
Context:

{context}


Student Question:

{question}


Answer:
"""
        )
    ]
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI College Study Assistant API is running!"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "message": "RAG backend is running correctly."
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/api/chat")
def chat(request: ChatRequest):

    try:

        # ----------------------------------------------------
        # GET QUESTION
        # ----------------------------------------------------

        question = request.message.strip()

        if not question:

            return {
                "response": "Please enter a question."
            }


        # ----------------------------------------------------
        # SEARCH CHROMA DATABASE
        # ----------------------------------------------------

        documents = retriever.invoke(question)


        # ----------------------------------------------------
        # CHECK DOCUMENTS
        # ----------------------------------------------------

        if not documents:

            return {
                "response": "I could not find the answer in the document."
            }


        # ----------------------------------------------------
        # CREATE CONTEXT
        # ----------------------------------------------------

        context_parts = []

        for document in documents:

            context_parts.append(
                document.page_content
            )

        context = "\n\n".join(context_parts)


        # ----------------------------------------------------
        # CREATE PROMPT
        # ----------------------------------------------------

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )


        # ----------------------------------------------------
        # CALL GEMINI
        # ----------------------------------------------------

        result = llm.invoke(final_prompt)


        # ----------------------------------------------------
        # GET GEMINI RESPONSE
        # ----------------------------------------------------

        answer = result.content


        # ----------------------------------------------------
        # CONVERT GEMINI RESPONSE TO PLAIN TEXT
        # ----------------------------------------------------

        if isinstance(answer, str):

            final_answer = answer

        elif isinstance(answer, list):

            text_parts = []

            for item in answer:

                if isinstance(item, str):

                    text_parts.append(item)

                elif isinstance(item, dict):

                    # Gemini may return:
                    # {"text": "some answer"}

                    if "text" in item:

                        text_parts.append(
                            str(item["text"])
                        )

                    else:

                        text_parts.append(
                            str(item)
                        )

                else:

                    text_parts.append(
                        str(item)
                    )

            final_answer = "\n".join(text_parts)

        elif isinstance(answer, dict):

            if "text" in answer:

                final_answer = str(
                    answer["text"]
                )

            else:

                final_answer = str(answer)

        else:

            final_answer = str(answer)


        # ----------------------------------------------------
        # RETURN ANSWER TO FRONTEND
        # ----------------------------------------------------

        return {
            "response": final_answer
        }


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print()
        print("=" * 60)
        print("ERROR WHILE PROCESSING QUESTION")
        print("=" * 60)
        print(str(e))
        print("=" * 60)
        print()

        return {
            "error": str(e),
            "response": "Sorry, something went wrong while processing your question."
        }


# ============================================================
# STARTUP MESSAGE
# ============================================================

@app.on_event("startup")
def startup_event():

    print()
    print("=" * 60)
    print("       AI COLLEGE STUDY ASSISTANT")
    print("=" * 60)
    print()
    print("Backend      : FastAPI")
    print("Vector DB    : Chroma")
    print("Embeddings   : all-MiniLM-L6-v2")
    print("LLM          : Gemini")
    print()
    print("API          : http://127.0.0.1:8000")
    print("Swagger Docs : http://127.0.0.1:8000/docs")
    print("Chat API     : http://127.0.0.1:8000/api/chat")
    print()
    print("=" * 60)
    print()