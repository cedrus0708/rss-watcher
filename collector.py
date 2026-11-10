import html
import json
import re
import feedparser
from sentence_transformers import SentenceTransformer
from google import genai

from config import (
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODELS,
    EMBEDDING_MODEL_NAME,
    RSS_FEEDS,
    MAX_ENTRIES_PER_FEED,
    USER_INTERESTS,
    EXCLUDED_TOPICS,
    CLUSTER_MATCH_THRESHOLD,
    MIN_RELEVANCE_SCORE,
)
from db import NewsDatabase, cosine_similarity

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"


def clean_html(raw_html: str) -> str:
    """Eltávolítja a HTML tageket és felesleges szóközöket az RSS összefoglalókból."""
    if not raw_html:
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", raw_html)
    unescaped = html.unescape(no_tags)
    cleaned = re.sub(r"\s+", " ", unescaped).strip()
    return cleaned[:1000]


def is_relevant_article(
    article_emb: list[float],
    interest_embs: list[list[float]],
    excluded_embs: list[list[float]],
    feed_category: str,
) -> tuple[bool, float, str]:
    """
    Megvizsgálja, hogy a cikk illeszkedik-e az érdeklődési körhöz (AI, Tech, Piac/Gazdaság),
    illetve nem tartozik-e a kizárt témákhoz (bulvár, sport, baleset).
    """
    interest_labels = ["AI", "Tech", "Piac & Gazdaság"]
    interest_scores = [cosine_similarity(article_emb, i_emb) for i_emb in interest_embs]
    excluded_scores = [cosine_similarity(article_emb, e_emb) for e_emb in excluded_embs]

    max_interest_idx = int( max(range(len(interest_scores)), key=lambda i: interest_scores[i]) )
    max_interest = interest_scores[max_interest_idx]
    best_topic = interest_labels[max_interest_idx] if max_interest_idx < len(interest_labels) else feed_category
    max_excluded = max(excluded_scores) if excluded_scores else 0.0

    # Ha egyértelműen bulvár/sport/kékfény, kiszűrjük
    if max_excluded > max_interest + 0.06 and max_excluded > 0.25:
        return False, max_interest, best_topic

    # Ha dedikált szakmai csatornáról jön (AI, Tech, Piac), megtartjuk, hacsak nem kizárt téma
    if feed_category != "Általános":
        return True, max_interest, feed_category

    # Általános hírportáloknál (Telex, HVG) csak az érdeklődési körbe vágó cikkeket engedjük át
    return (max_interest >= MIN_RELEVANCE_SCORE and max_interest >= max_excluded), max_interest, best_topic


def summarize_batch_with_gemini(gemini_client: genai.Client, items: list[dict]) -> dict[int, dict]:
    """
    Egyetlen kötegelt (batch) Gemini API hívással készít magyar nyelvű, objektív,
    tényszerű címet és 2-3 mondatos összefoglalót az új eseményekhez.
    Így elkerüljük az API rate limitet és az angol cikkeket is rögtön magyarítja.
    """
    if not items or not gemini_client:
        return {}

    payload = [
        {"idx": item["idx"], "title": item["title"], "text": item["text"][:600]}
        for item in items
    ]

    prompt = f"""Te egy objektív, tényszerű hírszerkesztő vagy.
Az alábbi JSON listában új híresemények nyers címei és kivonatai szerepelnek (magyar vagy angol nyelven).
Készíts mindegyikhez MAGYAR nyelven:
1. Egy rövid, kattintásvadászattól mentes, tényszerű esemény-címet ("title").
2. Egy 2-3 mondatos, objektív összefoglalót ("summary").

Válaszolj KIZÁRÓLAG érvényes JSON listával az alábbi formátumban:
[
  {{"idx": 0, "title": "Magyar esemény cím", "summary": "2-3 mondatos objektív összefoglaló."}}
]

Bemeneti hírek:
{json.dumps(payload, ensure_ascii=False)}
"""
    for model_name in GEMINI_FALLBACK_MODELS:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            parsed = json.loads(response.text)
            return {int(row["idx"]): row for row in parsed if "idx" in row}
        except Exception as e:
            print(f" [!] {model_name} hiba, próbálkozás a következő modellel... ({type(e).__name__})")
    return {}


def collect_news(max_per_feed: int = MAX_ENTRIES_PER_FEED):
    print("==================================================")
    print(" 1. LÉPCSŐ: HÍRGYŰJTÉS, SZŰRÉS ÉS KLASZTEREZÉS")
    print("==================================================")

    db = NewsDatabase()
    gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

    print(f"[Modell] Nyelvi vektorizáló betöltése ({EMBEDDING_MODEL_NAME})...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Érdeklődési kör és kizárt témák vektorizálása az előszűréshez
    interest_embs = [model.encode(t).tolist() for t in USER_INTERESTS]
    excluded_embs = [model.encode(t).tolist() for t in EXCLUDED_TOPICS]

    pending_new_clusters = []
    stats = {"seen": 0, "filtered_out": 0, "clustered": 0, "new_events": 0}

    for feed_cfg in RSS_FEEDS:
        feed_name = feed_cfg["name"]
        feed_url = feed_cfg["url"]
        feed_cat = feed_cfg["category"]

        print(f"\n[RSS] {feed_name} ({feed_cat}) -> {feed_url}")
        try:
            feed = feedparser.parse(feed_url, request_headers={"User-Agent": USER_AGENT})
        except Exception as e:
            print(f" [!] Hiba az RSS letöltésekor: {e}")
            continue

        entries = feed.entries[:max_per_feed]
        if not entries:
            print(" [!] Nem található bejegyzés a csatornában.")
            continue

        for entry in entries:
            title = clean_html(getattr(entry, "title", ""))
            link = getattr(entry, "link", "").strip()
            if not title or not link:
                continue

            # 1. Ellenőrzés: szerepel-e már az adatbázisban?
            if db.article_exists(link):
                stats["seen"] += 1
                continue

            raw_summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
            summary = clean_html(raw_summary)
            published_at = getattr(entry, "published", None)

            text_to_embed = f"{title}. {summary}" if summary else title
            embedding = model.encode(text_to_embed).tolist()

            # 2. Vektoros relevancia- és érdeklődés-szűrés (AI, Tech, Piac)
            is_rel, score, detected_topic = is_relevant_article(
                embedding, interest_embs, excluded_embs, feed_cat
            )
            if not is_rel:
                stats["filtered_out"] += 1
                print(f" [-] Kiszűrve (nem releváns: {score:.2f}): {title[:65]}...")
                continue

            # 3. Megnézzük, tartozik-e már meglévő esemény-klaszterhez
            match = db.match_cluster(embedding, threshold=CLUSTER_MATCH_THRESHOLD)
            if match:
                cluster_id = match["id"]
                db.insert_article(
                    cluster_id=cluster_id,
                    source_name=feed_name,
                    title=title,
                    summary=summary,
                    url=link,
                    published_at=published_at,
                )
                stats["clustered"] += 1
                print(f" [+] Hozzákapcsolva meglévő eseményhez (#{cluster_id}): {title[:60]}...")
            else:
                # Megnézzük, hogy a mostani futás során várólistán lévő új klaszterek között van-e már
                matched_pending = None
                for p in pending_new_clusters:
                    if cosine_similarity(embedding, p["embedding"]) >= CLUSTER_MATCH_THRESHOLD:
                        matched_pending = p
                        break

                if matched_pending:
                    matched_pending["articles"].append(
                        {
                            "source_name": feed_name,
                            "title": title,
                            "summary": summary,
                            "url": link,
                            "published_at": published_at,
                        }
                    )
                    stats["clustered"] += 1
                    print(f" [+] Összevonva új eseménnyel: {title[:60]}...")
                else:
                    idx = len(pending_new_clusters)
                    pending_new_clusters.append(
                        {
                            "idx": idx,
                            "title": title,
                            "text": text_to_embed,
                            "summary": summary or title,
                            "category": detected_topic,
                            "embedding": embedding,
                            "articles": [
                                {
                                    "source_name": feed_name,
                                    "title": title,
                                    "summary": summary,
                                    "url": link,
                                    "published_at": published_at,
                                }
                            ],
                        }
                    )
                    stats["new_events"] += 1
                    print(f" [*] Új esemény ({detected_topic}, {score:.2f}): {title[:60]}...")

    # 4. Új események kötegelt összefoglalása Geminivel és mentése az adatbázisba
    if pending_new_clusters:
        print(f"\n[Gemini] {len(pending_new_clusters)} új esemény összefoglalása kötegekben...")
        batch_size = 15
        ai_summaries = {}
        for i in range(0, len(pending_new_clusters), batch_size):
            batch = pending_new_clusters[i : i + batch_size]
            ai_summaries.update(summarize_batch_with_gemini(gemini_client, batch))

        for item in pending_new_clusters:
            ai_data = ai_summaries.get(item["idx"], {})
            final_title = ai_data.get("title") or item["title"]
            final_summary = ai_data.get("summary") or item["summary"]

            cluster_id = db.create_cluster(
                title=final_title,
                summary=final_summary,
                embedding=item["embedding"],
                category=item["category"],
            )
            for art in item["articles"]:
                db.insert_article(
                    cluster_id=cluster_id,
                    source_name=art["source_name"],
                    title=art["title"],
                    summary=art["summary"],
                    url=art["url"],
                    published_at=art["published_at"],
                )

    print("\n==================================================")
    print(
        f" KÉSZ! Új események: {stats['new_events']} | "
        f"Meglévőhöz kapcsolva: {stats['clustered']} | "
        f"Kiszűrve: {stats['filtered_out']} | "
        f"Már látott: {stats['seen']}"
    )
    print("==================================================")


if __name__ == "__main__":
    collect_news()
