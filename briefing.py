import asyncio
import json
import re
import urllib.request
import urllib.parse
from datetime import datetime
import edge_tts
from google import genai

from config import (
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODELS,
    TTS_VOICE,
    TTS_RATE,
    OUTPUT_DIR,
    USER_INTERESTS,
    EXCLUDED_TOPICS,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)
from db import NewsDatabase
from collector import collect_news

HUNGARIAN_MONTHS = [
    "",
    "január",
    "február",
    "március",
    "április",
    "május",
    "június",
    "július",
    "augusztus",
    "szeptember",
    "október",
    "november",
    "december",
]


def format_hungarian_date(dt: datetime) -> str:
    return f"{dt.year}. {HUNGARIAN_MONTHS[dt.month]} {dt.day}."


def clean_script_for_tts(text: str) -> str:
    """Biztosítja, hogy a TTS-nek küldött szövegben ne maradjon markdown jelölés."""
    text = re.sub(r"[*_#`~>]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_radio_script(clusters: list[dict]) -> dict:
    """
    A Gemini kiválasztja a legfontosabb AI, Tech és Piaci eseményeket az elmúlt 24 órából,
    majd megírja a ~650-750 szavas (~5 perces) objektív rádiós híradó szövegét.
    """
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    today_str = format_hungarian_date(datetime.now())

    # Előkészítjük a jelölt eseményeket a Gemini számára (max 80 legfontosabb klaszter)
    candidates = []
    for c in clusters[:80]:
        sources = list({a["source_name"] for a in c["articles"]})
        article_titles = [f"{a['source_name']}: {a['title']}" for a in c["articles"][:4]]
        candidates.append(
            {
                "cluster_id": c["id"],
                "category": c.get("category", ""),
                "title": c["title"],
                "summary": c["summary"],
                "source_count": len(sources),
                "sources": sources,
                "articles": article_titles,
            }
        )

    prompt = f"""Te egy professzionális, pártatlan és objektív reggeli rádiós hírszerkesztő vagy.
A mai dátum: {today_str}

A hallgató érdeklődési köre (prioritási sorrendben):
{chr(10).join(f"- {t}" for t in USER_INTERESTS)}

Kizárandó témák (ezeket SOHA ne válogasd be):
{chr(10).join(f"- {t}" for t in EXCLUDED_TOPICS)}

FELADATOD:
1. Válogasd ki az alábbi események közül a 8–10 legfontosabb, legérdekesebb hírt (kiemelten kezelve az AI, a Technológia és a Piaci/Gazdasági híreket, valamint azokat, amelyekről több forrás is beszámolt).
2. Írj belőlük egy ~650–750 szavas (felolvasva kb. 5 perces), folyamatos, fülre írt magyar reggeli híradó szöveget ("script"), amelyet egyetlen hírolvasó olvas fel a munkába induló hallgatónak.

SZIGORÚ FORMAI ÉS STILÁRIS KÖVETELMÉNYEK A "script" MEZŐHÖZ:
- KIZÁRÓLAG felolvasásra szánt, természetes, gördülékeny magyar folyószöveget írj, bekezdésekre tagolva!
- TILOS bármilyen markdown jelölést (*, #, -, felsorolásjelek), zárójelet, URL-t vagy színpadi utasítást (pl. "[Szünet]", "Hírolvasó:") beleírni!
- Objektív, tényszerű, higgadt és informatív stílusban fogalmazz, mellőzve a kattintásvadász fordulatokat.
- Felépítés:
  1. Rövid, barátságos reggeli köszöntés ({today_str} említésével) és 1-2 mondatos tartalomjegyzék a mai vezető témákról.
  2. Vezető hírek és Piaci/Gazdasági áttekintés (a legfontosabb makrogazdasági, tőzsdei és vállalati események részletesebb kibontása).
  3. Mesterséges Intelligencia (AI) és Technológiai blokk (a legújabb AI modellek, tech óriások, chipipar, innovációk).
  4. Rövid, lendületes elköszönés jó munkát kívánva.

Válaszolj KIZÁRÓLAG érvényes JSON formátumban az alábbi sémával:
{{
  "selected_cluster_ids": [1, 2, 3],
  "headline_summary": "1 mondatos rövid összefoglaló a mai adás fő témáiról",
  "script": "A teljes, 650-750 szavas, felolvasásra kész magyar híradó szövege..."
}}

Mai elérhető híresemények:
{json.dumps(candidates, ensure_ascii=False, indent=2)}
"""

    last_err = None
    for model_name in GEMINI_FALLBACK_MODELS:
        try:
            print(f"[Gemini] Szkript generálása modellel: {model_name}...")
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            return json.loads(response.text)
        except Exception as e:
            last_err = e
            print(f"[!] {model_name} nem elérhető ({type(e).__name__}), váltás tartalék modellre...")
    raise RuntimeError(f"Minden Gemini modell hívása sikertelen volt: {last_err}")


async def synthesize_speech(script_text: str, output_mp3_path):
    """Legenerálja az MP3 hangfájlt az Edge-TTS magyar neurális hangjával."""
    communicate = edge_tts.Communicate(text=script_text, voice=TTS_VOICE, rate=TTS_RATE)
    await communicate.save(str(output_mp3_path))


def send_to_telegram(mp3_path, headline_summary: str, selected_clusters: list[dict]):
    """
    Ha a TELEGRAM_BOT_TOKEN és TELEGRAM_CHAT_ID be van állítva a .env-ben,
    elküldi a kész MP3 hangfájlt és a forráslinkeket a telefonra.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Nincs megadva TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID a .env-ben -> csak helyi mentés történt.")
        return

    print("[Telegram] Hanganyag és forráslinkek küldése a telefonodra...")
    today_short = datetime.now().strftime("%Y.%m.%d.")

    # 1. MP3 hangfájl feltöltése (multipart/form-data)
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    url_audio = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendAudio"

    with open(mp3_path, "rb") as f:
        audio_bytes = f.read()

    parts = []
    for name, val in [
        ("chat_id", TELEGRAM_CHAT_ID),
        ("title", f"Reggeli Híradó ({today_short})"),
        ("performer", "AI Hírszerkesztő"),
        ("caption", f"🎧 Reggeli ~5 perces összefoglaló ({today_short})\n{headline_summary}"),
    ]:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{val}\r\n".encode("utf-8")
        )

    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="{mp3_path.name}"\r\n'
        f"Content-Type: audio/mpeg\r\n\r\n".encode("utf-8")
        + audio_bytes
        + f"\r\n--{boundary}--\r\n".encode("utf-8")
    )
    body = b"".join(parts)

    req = urllib.request.Request(
        url_audio,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            print(f"[Telegram] MP3 sikeresen elküldve! (HTTP {resp.status})")
    except Exception as e:
        print(f"[Telegram] Hiba az MP3 küldésekor: {e}")

    # 2. Forráslinkek küldése külön szöveges üzenetben
    lines = [f"📰 <b>A mai adásban érintett hírek és források ({today_short}):</b>\n"]
    for idx, c in enumerate(selected_clusters, 1):
        links_html = ", ".join(
            f'<a href="{a["url"]}">{a["source_name"]}</a>' for a in c["articles"][:4]
        )
        lines.append(f"{idx}. <b>{c['title']}</b> ({links_html})")

    msg_text = "\n".join(lines)
    url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    msg_data = urllib.parse.urlencode(
        {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")

    try:
        urllib.request.urlopen(urllib.request.Request(url_msg, data=msg_data, method="POST"), timeout=30)
        print("[Telegram] Forráslinkek sikeresen elküldve!")
    except Exception as e:
        print(f"[Telegram] Hiba a linkek küldésekor: {e}")


def create_morning_briefing(hours: int = 24):
    print("==================================================")
    print(" 2. LÉPCSŐ: REGGELI HÍRADÓ ÉS HANGANYAG GENERÁLÁSA")
    print("==================================================")

    db = NewsDatabase()
    clusters = db.get_recent_clusters_with_articles(hours=hours)

    if not clusters:
        print("[Info] Nincs még hír az elmúlt 24 órából az adatbázisban -> Hírgyűjtő indítása...")
        collect_news()
        clusters = db.get_recent_clusters_with_articles(hours=hours)

    if not clusters:
        print("[!] Nem található feldolgozható hír.")
        return

    print(f"[Kurátor] {len(clusters)} esemény-klaszter elemzése és ~5 perces rádiós szkript írása...")
    ai_result = generate_radio_script(clusters)

    raw_script = ai_result.get("script", "")
    clean_script = clean_script_for_tts(raw_script)
    headline_summary = ai_result.get("headline_summary", "")
    selected_ids = set(ai_result.get("selected_cluster_ids", []))

    cluster_map = {c["id"]: c for c in clusters}
    selected_clusters = [cluster_map[cid] for cid in selected_ids if cid in cluster_map]
    if not selected_clusters:
        selected_clusters = clusters[:8]

    word_count = len(clean_script.split())
    est_minutes = word_count / 138.0
    print(f"[Szkript] Elkészült! Terjedelem: {word_count} szó (becsült hossz: ~{est_minutes:.1f} perc)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_slug = datetime.now().strftime("%Y-%m-%d")
    mp3_path = OUTPUT_DIR / f"reggeli_hirado_{date_slug}.mp3"
    md_path = OUTPUT_DIR / f"reggeli_hirado_{date_slug}.md"

    # Szöveges leirat és forráslinkek mentése
    md_lines = [
        f"# 🎧 Reggeli Híradó – {format_hungarian_date(datetime.now())}",
        f"\n> **Összefoglaló:** {headline_summary}\n",
        "## 🎙️ Felolvasott szöveg\n",
        clean_script,
        "\n---\n## 🔗 Felhasznált hírek és források\n",
    ]
    for idx, c in enumerate(selected_clusters, 1):
        links_md = " | ".join(f"[{a['source_name']}]({a['url']})" for a in c["articles"])
        md_lines.append(f"{idx}. **{c['title']}** ({links_md})\n   - *{c['summary']}*")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[Mentés] Szöveg és forráslinkek elmentve: {md_path}")

    # Hanggenerálás Edge-TTS-sel
    print(f"[TTS] Hanganyag generálása ({TTS_VOICE})...")
    asyncio.run(synthesize_speech(clean_script, mp3_path))
    size_kb = mp3_path.stat().st_size / 1024
    print(f"[Mentés] MP3 hangfájl elkészült: {mp3_path} ({size_kb:.0f} KB)")

    # Telegram küldés (ha be van állítva)
    send_to_telegram(mp3_path, headline_summary, selected_clusters)

    print("==================================================")
    print(" REGGELI HÍRADÓ SIKERESEN ELKÉSZÜLT!")
    print("==================================================")


if __name__ == "__main__":
    create_morning_briefing()
