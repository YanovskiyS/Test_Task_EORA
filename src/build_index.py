from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

URLS = [
"https://eora.ru/cases/promyshlennaya-bezopasnost",
"https://eora.ru/cases/lamoda-systema-segmentacii-i-poiska-po-pohozhey-odezhde",
"https://eora.ru/cases/navyki-dlya-golosovyh-assistentov/karas-golosovoy-assistent",
"https://eora.ru/cases/assistenty-dlya-gorodov",
"https://eora.ru/cases/avtomatizaciya-v-promyshlennosti/chemrar-raspoznovanie-molekul",
"https://eora.ru/cases/zeptolab-skazki-pro-amnyama-dlya-sberbox",
"https://eora.ru/cases/goosegaming-algoritm-dlya-ocenki-igrokov",
"https://eora.ru/cases/dodo-pizza-robot-analitik-otzyvov",
"https://eora.ru/cases/ifarm-nejroset-dlya-ferm",
"https://eora.ru/cases/zhivibezstraha-navyk-dlya-proverki-rodinok",
"https://eora.ru/cases/sportrecs-nejroset-operator-sportivnyh-translyacij",
"https://eora.ru/cases/avon-chat-bot-dlya-zhenshchin",
"https://eora.ru/cases/navyki-dlya-golosovyh-assistentov/navyk-dlya-proverki-loterejnyh-biletov",
"https://eora.ru/cases/computer-vision/iss-analiz-foto-avtomobilej",
"https://eora.ru/cases/purina-master-bot",
"https://eora.ru/cases/skinclub-algoritm-dlya-ocenki-veroyatnostej",
"https://eora.ru/cases/skolkovo-chat-bot-dlya-startapov-i-investorov",
"https://eora.ru/cases/purina-podbor-korma-dlya-sobaki",
"https://eora.ru/cases/purina-navyk-viktorina",
"https://eora.ru/cases/dodo-pizza-pilot-po-avtomatizacii-kontakt-centra",
"https://eora.ru/cases/dodo-pizza-avtomatizaciya-kontakt-centra",
"https://eora.ru/cases/icl-bot-sufler-dlya-kontakt-centra",
"https://eora.ru/cases/s7-navyk-dlya-podbora-aviabiletov",
"https://eora.ru/cases/workeat-whatsapp-bot",
"https://eora.ru/cases/absolyut-strahovanie-navyk-dlya-raschyota-strahovki",
"https://eora.ru/cases/kazanexpress-poisk-tovarov-po-foto",
"https://eora.ru/cases/kazanexpress-sistema-rekomendacij-na-sajte",
"https://eora.ru/cases/intels-proverka-logotipa-na-plagiat",
"https://eora.ru/cases/karcher-viktorina-s-voprosami-pro-uborku",
"https://eora.ru/cases/chat-boty/purina-friskies-chat-bot-na-sajte",
"https://eora.ru/cases/nejroset-segmentaciya-video",
"https://eora.ru/cases/chat-boty/essa-nejroset-dlya-generacii-rolikov",
"https://eora.ru/cases/qiwi-poisk-anomalij",
"https://eora.ru/cases/frisbi-nejroset-dlya-raspoznavaniya-pokazanij-schetchikov",
"https://eora.ru/cases/skazki-dlya-gugl-assistenta",
"https://eora.ru/cases/chat-boty/hr-bot-dlya-magnit-kotoriy-priglashaet-na-sobesedovanie"
]

def load_and_split():
    loader = WebBaseLoader(URLS)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def build_faiss_index(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("parsed/faiss_index")
    print("Индекс сохранён в parsed/faiss_index")

if __name__ == "__main__":
    docs = load_and_split()
    build_faiss_index(docs)