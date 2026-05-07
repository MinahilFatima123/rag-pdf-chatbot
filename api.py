from fastapi import FastAPI, UploadFile, File, HTTPException
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
import tempfile
import logging

# ── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# ── LLM + Embeddings ────────────────────────────────────────────
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

llm = ChatGroq(model="llama-3.3-70b-versatile")  

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say: "I don't have enough information in the document to answer this."

Context: {context}
Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# global chain — rebuilt on every new PDF upload
chain = None


# ── Endpoints ────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status": "RAG API is running",
        "pdf_loaded": chain is not None
    }


@app.get("/health")
def health():
    return {"status": "ok", "pdf_loaded": chain is not None}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global chain

    # validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    tmp_path = None
    try:
        # save to temp file — no Railway filesystem dependency
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()

            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"PDF temporarily saved: {tmp_path}")

        # load + chunk
        docs = PyPDFLoader(tmp_path).load()
        if not docs:
            raise HTTPException(status_code=422, detail="PDF appears to be empty or unreadable.")

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        ).split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks from {len(docs)} pages")

        # embed + store in memory (no persist_directory = Railway safe)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        # build chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("RAG chain built successfully")
        return {
            "message": f"'{file.filename}' uploaded and ready. Use /ask to ask questions.",
            "pages": len(docs),
            "chunks": len(chunks)
        }

    except HTTPException:
        raise  # re-raise HTTP exceptions as-is

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process PDF.")

    finally:
        # always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info("Temp file cleaned up")


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(body: Question):
    global chain

    # validate PDF is loaded
    if chain is None:
        raise HTTPException(
            status_code=400,
            detail="No PDF uploaded yet. Please call /upload first."
        )

    # validate question
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if len(body.question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long. Max 1000 characters.")

    try:
        logger.info(f"Question: {body.question}")
        answer = chain.invoke(body.question)
        logger.info("Answer generated successfully")
        return {"question": body.question, "answer": answer}

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer. Please try again.")