import os
from pathlib import Path
from dotenv import load_dotenv

# Környezeti változók betöltése a .env fájlból
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ================= API KULCSOK =================
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# ================= MODELLEK ÉS HANG =================
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL_NAME = "gemini-3-flash-preview"
GEMINI_FALLBACK_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-3.8-flash",
]

# Magyar neurális hírolvasó hangok (Edge-TTS):
# - "hu-HU-TamasNeural" (férfi hírolvasó)
# - "hu-HU-NoemiNeural" (női hírolvasó)
TTS_VOICE = "hu-HU-TamasNeural"
TTS_RATE = "+5%"  # Kicsit lendületesebb, természetes rádiós tempó (pl. "+0%", "+5%", "+10%")

# Kimeneti mappa a helyi MP3 és szöveges fájloknak
OUTPUT_DIR = BASE_DIR / "output"
LOCAL_DB_PATH = BASE_DIR / "news.db"

# ================= HÍRFORRÁSOK (RSS) =================
# AI, Tech és Piaci/Gazdasági fókuszú hazai és nemzetközi hírforrások
RSS_FEEDS = [
    # --- AI & Nemzetközi Tech ---
    {
        "name": "TechCrunch AI",
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "category": "AI",
    },
    {
        "name": "The Verge AI",
        "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "category": "AI",
    },
    # --- Magyar Tech & Tudomány ---
    {
        "name": "HWSW",
        "url": "https://www.hwsw.hu/xml/latest_news_rss.xml",
        "category": "Tech",
    },
    {
        "name": "Prohardver",
        "url": "https://prohardver.hu/hirfolyam/anyagok/rss.xml",
        "category": "Tech",
    },
    {
        "name": "Qubit",
        "url": "https://qubit.hu/feed",
        "category": "Tech & Tudomány",
    },
    {
        "name": "Rakéta",
        "url": "https://raketa.hu/feed",
        "category": "Tech & Innováció",
    },
    # --- Piaci, Gazdasági & Tőzsdei hírek ---
    {
        "name": "Portfolio",
        "url": "https://www.portfolio.hu/rss/all.xml",
        "category": "Piac & Gazdaság",
    },
    # --- Általános hazai vezető portálok (gazdasági/tech/világpiaci szűréshez) ---
    {
        "name": "Telex",
        "url": "https://telex.hu/rss",
        "category": "Általános",
    },
    {
        "name": "HVG",
        "url": "https://hvg.hu/rss",
        "category": "Általános",
    },
]

# Maximálisan vizsgált cikkek száma RSS csatornánként egy futás során
MAX_ENTRIES_PER_FEED = 12

# ================= ÉRDEKLŐDÉSI KÖR ÉS SZŰRÉS =================
# Ezeket a témákat keressük és emeljük ki (vektoros előszűréshez + LLM kurátornak)
USER_INTERESTS = [
    "Mesterséges intelligencia (AI), nagy nyelvi modellek (LLM), gépi tanulás, OpenAI, Google, Anthropic, AI szabályozás és robotika",
    "Technológia, szoftverfejlesztés, félvezetőipar, chipek, kiberbiztonság, tech óriásvállalatok (Big Tech), hardver és innováció",
    "Piaci hírek, tőzsde, makrogazdaság, pénzügyek, infláció, kamatdöntések, árfolyamok, befektetések, vállalati felvásárlások és gazdaságpolitika",
]

# Témák, amiket ki akarunk szűrni (hogy ne kerüljenek be a reggeli híradóba)
EXCLUDED_TOPICS = [
    "Bulvár, celebek, sztárvilág, valóságshow, pletykák, horoszkóp",
    "Sport, labdarúgás, meccseredmények, Forma-1",
    "Helyi közlekedési balesetek, kékfény, apró bűnügyek, időjárás-jelentés",
]

# Vektoros klaszterezési küszöb (0.80 - 0.85 között az azonos eseményről szóló cikkek összevonásához)
CLUSTER_MATCH_THRESHOLD = 0.80

# Vektoros relevancia küszöb az Általános portálok (pl. Telex, HVG) cikkeinek előszűréséhez
# (A dedikált AI/Tech/Portfolio csatornák cikkeit alapból relevánsabbnak tekintjük)
MIN_RELEVANCE_SCORE = 0.22
