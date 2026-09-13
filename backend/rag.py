from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_database",
    embedding_function=embedding
)

retriver = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k" : 10,
        "lambda_mult" : 0.5
    }
)

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("RAG System Created : ")
print("Press 0 to Exit..")

while True:
    query = input("Write Your Message..")
    if query=="0":
        break

    docs = retriver.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context" : context,
        "question"  : query
    })

    response = llm.invoke(final_prompt)

    print("\nAI : ",response.text)