import json
import sqlite3
from datetime import datetime, timedelta, timezone
import numpy as np
from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_KEY, LOCAL_DB_PATH


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class NewsDatabase:
    """
    Egységes adatbázis-réteg:
    - Elsődlegesen a Supabase-t használja (articles, clusters, match_clusters RPC).
    - Ha a Supabase projekt szünetel (paused) vagy nem érhető el, automatikusan
      helyi SQLite adatbázisra (news.db) vált, ugyanazzal a vektoros kereséssel.
    """

    def __init__(self):
        self.use_supabase = False
        self.supabase: Client | None = None

        if SUPABASE_URL and SUPABASE_KEY:
            try:
                client = create_client(SUPABASE_URL, SUPABASE_KEY)
                # Gyors kapcsolat-teszt
                client.table("clusters").select("id").limit(1).execute()
                self.supabase = client
                self.use_supabase = True
                print("[DB] Csatlakozva a Supabase felhő adatbázishoz.")
            except Exception as e:
                print(
                    f"[DB] Supabase nem elérhető ({type(e).__name__} - valószínűleg szünetel az ingyenes projekt)."
                )
                print(f"[DB] -> Automatikus átváltás helyi SQLite adatbázisra: {LOCAL_DB_PATH.name}")
        else:
            print(f"[DB] Helyi SQLite adatbázis használata: {LOCAL_DB_PATH.name}")

        if not self.use_supabase:
            self._init_sqlite()

    def _get_sqlite_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self):
        with self._get_sqlite_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    category TEXT DEFAULT '',
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id INTEGER NOT NULL,
                    source_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    url TEXT UNIQUE NOT NULL,
                    published_at TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (cluster_id) REFERENCES clusters(id)
                )
                """
            )
            conn.commit()

    def article_exists(self, url: str) -> bool:
        if self.use_supabase:
            res = self.supabase.table("articles").select("id").eq("url", url).execute()
            return len(res.data) > 0
        else:
            with self._get_sqlite_conn() as conn:
                row = conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()
                return row is not None

    def match_cluster(self, embedding: list[float], threshold: float = 0.80) -> dict | None:
        if self.use_supabase:
            res = self.supabase.rpc(
                "match_clusters",
                {
                    "query_embedding": embedding,
                    "match_threshold": threshold,
                    "match_count": 1,
                },
            ).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]
            return None
        else:
            # Csak az elmúlt 72 óra klasztereiben keresünk egyező eseményt
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
            with self._get_sqlite_conn() as conn:
                rows = conn.execute(
                    "SELECT id, title, summary, embedding FROM clusters WHERE created_at >= ?",
                    (cutoff,),
                ).fetchall()

            best_match = None
            best_score = -1.0
            for r in rows:
                cluster_emb = json.loads(r["embedding"])
                score = cosine_similarity(embedding, cluster_emb)
                if score >= threshold and score > best_score:
                    best_score = score
                    best_match = {
                        "id": r["id"],
                        "title": r["title"],
                        "summary": r["summary"],
                        "similarity": score,
                    }
            return best_match

    def create_cluster(
        self, title: str, summary: str, embedding: list[float], category: str = ""
    ) -> int | str:
        now_iso = datetime.now(timezone.utc).isoformat()
        if self.use_supabase:
            res = (
                self.supabase.table("clusters")
                .insert(
                    {
                        "title": title,
                        "summary": summary,
                        "embedding": embedding,
                    }
                )
                .execute()
            )
            return res.data[0]["id"]
        else:
            with self._get_sqlite_conn() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO clusters (title, summary, category, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (title, summary, category, json.dumps(embedding), now_iso),
                )
                conn.commit()
                return cur.lastrowid

    def insert_article(
        self,
        cluster_id: int | str,
        source_name: str,
        title: str,
        summary: str,
        url: str,
        published_at: str | None = None,
    ):
        now_iso = datetime.now(timezone.utc).isoformat()
        if self.use_supabase:
            self.supabase.table("articles").insert(
                {
                    "cluster_id": cluster_id,
                    "source_name": source_name,
                    "title": title,
                    "url": url,
                    "published_at": published_at,
                }
            ).execute()
        else:
            with self._get_sqlite_conn() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO articles
                    (cluster_id, source_name, title, summary, url, published_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (cluster_id, source_name, title, summary, url, published_at, now_iso),
                )
                conn.commit()

    def get_recent_clusters_with_articles(self, hours: int = 24) -> list[dict]:
        """
        Lekéri az elmúlt `hours` órában keletkezett (vagy friss cikket kapott) klasztereket,
        a hozzájuk tartozó összes cikkel és forrással együtt.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        if self.use_supabase:
            # Lekérjük a friss cikkeket, majd a hozzájuk tartozó klasztereket
            articles_res = (
                self.supabase.table("articles")
                .select("*")
                .order("id", desc=True)
                .limit(200)
                .execute()
            )
            articles = articles_res.data or []
            if not articles:
                return []

            cluster_ids = list({a["cluster_id"] for a in articles if a.get("cluster_id")})
            clusters_res = (
                self.supabase.table("clusters")
                .select("id, title, summary")
                .in_("id", cluster_ids)
                .execute()
            )
            clusters_map = {c["id"]: {**c, "articles": []} for c in (clusters_res.data or [])}
            for a in articles:
                cid = a.get("cluster_id")
                if cid in clusters_map:
                    clusters_map[cid]["articles"].append(a)

            result = list(clusters_map.values())
            result.sort(key=lambda c: len(c["articles"]), reverse=True)
            return result
        else:
            with self._get_sqlite_conn() as conn:
                clusters_rows = conn.execute(
                    """
                    SELECT id, title, summary, category, created_at
                    FROM clusters
                    WHERE created_at >= ?
                    ORDER BY id DESC
                    """,
                    (cutoff,),
                ).fetchall()

                result = []
                for c in clusters_rows:
                    art_rows = conn.execute(
                        """
                        SELECT source_name, title, summary, url, published_at
                        FROM articles
                        WHERE cluster_id = ?
                        ORDER BY id ASC
                        """,
                        (c["id"],),
                    ).fetchall()
                    articles = [dict(a) for a in art_rows]
                    if articles:
                        result.append(
                            {
                                "id": c["id"],
                                "title": c["title"],
                                "summary": c["summary"],
                                "category": c["category"],
                                "created_at": c["created_at"],
                                "articles": articles,
                            }
                        )

                # Rangsoroljuk előre azokat az eseményeket, amelyekről több forrás is írt
                result.sort(key=lambda x: len(x["articles"]), reverse=True)
                return result
