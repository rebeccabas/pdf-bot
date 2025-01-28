import time
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def process_pdf(pdf_path, db_path):
    start_time = time.time()
    
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"PDF loaded in {time.time() - start_time:.2f} seconds")
    
    print("\nSplitting text...")
    split_start = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    print(f"Text split in {time.time() - split_start:.2f} seconds")
    print(f"Created {len(texts)} text chunks")
    
    print("\nGenerating embeddings...")
    embed_start = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store with progress bar
    vectorstore = FAISS.from_documents(
        tqdm(texts, desc="Processing chunks"), 
        embeddings
    )
    print(f"Embeddings generated in {time.time() - embed_start:.2f} seconds")
    
    print("\nSaving database...")
    save_start = time.time()
    vectorstore.save_local(db_path)
    print(f"Database saved in {time.time() - save_start:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    process_pdf("book.pdf", "faiss_db")