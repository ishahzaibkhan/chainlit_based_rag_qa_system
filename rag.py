import glob
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Load env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

# Load docs
text_files = glob.glob("./hospitals/hospital_*.txt")
if not text_files:
    raise FileNotFoundError("No hospital text files found in ./hospitals/")

docs = []
for path in text_files:
    loader = TextLoader(path)
    loaded = loader.load()
    if loaded:
        docs.append(loaded[0])

# FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter  = CharacterTextSplitter(chunk_size=300, chunk_overlap=50, separator=".")
chunks    = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Gemini LLM
gen_llm = ChatOpenAI(
    model="gpt-4o",
    api_key=api_key,
    temperature=0.2,
)

# RAG chain
rag_qa = RetrievalQA.from_chain_type(
    llm=gen_llm,
    chain_type="map_reduce",
    retriever=vectorstore.as_retriever(k=5),
)

def answer(question: str) -> str:
    return rag_qa.run(question).strip()

if __name__ == "__main__":
    q = "Tell me about Chap Medical Bhakkar Hospital."
    print("Q:", q)
    print("A:", answer(q))
