from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


loader = PyPDFLoader(r"my_file.pdf")
docs = loader.load()
print(f"Loaded: {len(docs)} pages")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Chunks: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="phase4/chroma_db"
)
print("Stored in ChromaDB!")


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

query = "What is market segmentation?"
retrieved_chunks = retriever.invoke(query)

print(f"\nTop {len(retrieved_chunks)} chunks retrieved:")
for i, chunk in enumerate(retrieved_chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content)
    


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

question = "What is market segmentation?"
answer = chain.invoke(question)

print("\n=== ANSWER ===")
print(answer)