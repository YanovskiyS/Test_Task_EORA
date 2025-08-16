from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("parsed/faiss_index", embeddings, allow_dangerous_deserialization=True)

def search_similar_chunks(query, top_k=3):
    results = db.similarity_search(query, k=top_k)
    return [{"text": doc.page_content, "source": doc.metadata.get("source", "неизвестный источник")} for doc in results]