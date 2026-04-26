from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

app = FastAPI()

# ── BUILD OR LOAD VECTORSTORE ─────────────────────────

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
CHROMA_PATH = "phase4/chroma_db"

if not os.path.exists(CHROMA_PATH):
    print("Building vector store...")
    loader = PyPDFLoader("my_file.pdf")
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Vector store ready!")
else:
    print("Loading existing vector store...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

# ── CHAIN ─────────────────────────────────────────────
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
Context: {context}
Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── API ───────────────────────────────────────────────
class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(body: Question):
    answer = chain.invoke(body.question)
    return {"question": body.question, "answer": answer}