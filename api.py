from fastapi import FastAPI, UploadFile, File
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
import shutil

app = FastAPI()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

llm = ChatGroq(model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
Context: {context}
Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# store chain globally so /ask can use it
chain = None

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    global chain

    # save uploaded PDF to disk
    pdf_path = f"uploaded_{file.filename}"
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # build vector store from uploaded PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"message": f"{file.filename} uploaded and ready. Now use /ask to ask questions."}


class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(body: Question):
    if chain is None:
        return {"error": "No PDF uploaded yet. Please call /upload first."}
    answer = chain.invoke(body.question)
    return {"question": body.question, "answer": answer}

@app.get("/")
def home():
    return {"status": "RAG API is running"}