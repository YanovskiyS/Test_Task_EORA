from langchain_gigachat import GigaChat
from src.config import settings
from src.embeddings import search_similar_chunks

llm = GigaChat(
    verify_ssl_certs=False,
    scope="GIGACHAT_API_PERS",
    model="GigaChat-2-Max",
    temperature=0.2, credentials=settings.GIGA_CHAT_API_KEY
)

def build_prompt(question, context_chunks):
    context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])
    prompt = f"Ответь кратко на вопрос, используя следующий контест. Контекст:\n{context_text}\n\nВопрос:{question}\nОтвет:"
    return prompt

def get_answer_with_sources(question):
    chunks = search_similar_chunks(question, top_k=5)
    prompt = build_prompt(question, chunks)

    response = llm.invoke(prompt)
    sources = list(set(chunk["source"] for chunk in chunks))
    return response, sources

