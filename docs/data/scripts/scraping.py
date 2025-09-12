import requests
from bs4 import BeautifulSoup
import csv
import time
from tqdm import tqdm
import os
from requests.adapters import HTTPAdapter, Retry
from urllib.parse import urljoin

BASE = "https://www.hltv.org"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://www.hltv.org/",
}

# Requests session with retries
session = requests.Session()
retries = Retry(total=5, connect=5, read=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["HEAD", "GET", "OPTIONS"])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))

DEBUG = False
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "_debug")


def debug_write(name: str, content: str):
    if not DEBUG:
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    path = os.path.join(DEBUG_DIR, name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"[debug] failed to write {path}: {e}")


def looks_like_block_page(html: str) -> bool:
    # Be conservative: only treat as blocked if standard interstitial hints exist
    lower = html.lower()
    if "attention required" in lower and "cloudflare" in lower:
        return True
    if "just a moment" in lower and "cloudflare" in lower:
        return True
    if "verify you are a human" in lower:
        return True
    if "cf-error-details" in lower:
        return True
    return False


def get_match_links(pages=5):
    links = []
    for offset in range(0, pages * 100, 100):
        url = f"{BASE}/results?offset={offset}"
        r = session.get(url, headers=HEADERS, timeout=20)
        if DEBUG:
            print(f"[debug] GET {url} -> {r.status_code}, {len(r.text)} bytes")
        if r.status_code != 200:
            print(f"Warning: non-200 status {r.status_code} for {url}")
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        matches = soup.select("div.result-con a")
        if DEBUG:
            print(f"[debug] Found {len(matches)} anchors with selector 'div.result-con a' on offset {offset}")
        if offset == 0:
            debug_write("results_offset0.html", r.text)
        for match in matches:
            href = match.get("href", "")
            if href.startswith("/"):
                href = BASE + href
            if "/matches/" in href:
                links.append(href)
        time.sleep(1)
    # de-dupe preserving order
    seen = set()
    unique_links = []
    for h in links:
        if h not in seen:
            seen.add(h)
            unique_links.append(h)
    if DEBUG:
        print(f"[debug] Total unique match links collected: {len(unique_links)}")
    return unique_links


def extract_match_id_from_url(url: str) -> str:
    try:
        path = url.split("?")[0]
        parts = path.split("/")
        # https://www.hltv.org/matches/<id>/...
        for i, part in enumerate(parts):
            if part == "matches" and i + 1 < len(parts):
                return parts[i + 1]
    except Exception:
        return ""
    return ""


def extract_demo_link(soup: BeautifulSoup) -> str | None:
    # Prefer data attribute on the button
    btn = soup.select_one(".streams [data-demo-link]")
    if btn and btn.get("data-demo-link"):
        return urljoin(BASE, btn.get("data-demo-link"))
    # Fallback hidden link anchor
    hidden = soup.select_one(".streams a[data-manuel-download]")
    if hidden and hidden.get("href"):
        return urljoin(BASE, hidden.get("href"))
    return None


def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in (".", "-", "_") else "_" for c in name)


def resolve_demo_filename(resp: requests.Response, default_name: str) -> str:
    cd = resp.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        # naive parse
        fname = cd.split("filename=")[-1].strip().strip('"').strip("'")
        return safe_filename(fname) if fname else default_name
    # try content-type for extension
    ctype = resp.headers.get("Content-Type", "").lower()
    if "zip" in ctype:
        return default_name if default_name.endswith(".zip") else default_name + ".zip"
    if "x-rar" in ctype or "rar" in ctype:
        return default_name if default_name.endswith(".rar") else default_name + ".rar"
    if "octet-stream" in ctype:
        return default_name
    return default_name


def detect_ext_from_magic(b: bytes) -> str | None:
    if not b:
        return None
    # ZIP signatures: PK\x03\x04, PK\x05\x06, PK\x07\x08
    if b.startswith(b"PK"):
        return ".zip"
    # RAR v4/v5 signatures
    if b.startswith(b"Rar!\x1a\x07\x00") or b.startswith(b"Rar!\x1a\x07\x01\x00"):
        return ".rar"
    # GZIP
    if b.startswith(b"\x1f\x8b"):
        return ".gz"
    # 7z
    if b.startswith(b"7z\xbc\xaf'\x1c"):
        return ".7z"
    return None


def download_demo_file(demo_url: str, match_url: str, match_id: str, demos_dir: str) -> str | None:
    try:
        os.makedirs(demos_dir, exist_ok=True)
        demo_id = demo_url.rstrip("/").split("/")[-1]
        default_basename = safe_filename(f"{match_id}_{demo_id}")
        headers = dict(HEADERS)
        headers["Referer"] = match_url

        # First, try a small non-stream GET to detect interstitial HTML or immediate file
        probe = session.get(demo_url, headers=headers, timeout=30, allow_redirects=True, stream=False)
        if probe.status_code != 200:
            print(f"Warning: demo GET {demo_url} -> {probe.status_code}")
            return None
        ctype = probe.headers.get("Content-Type", "").lower()
        content_len = int(probe.headers.get("Content-Length", 0))

        # Helper to stream-download a URL to file with verification
        def stream_download(url: str, base_name: str) -> str | None:
            with session.get(url, headers=headers, timeout=300, allow_redirects=True, stream=True) as resp:
                if resp.status_code != 200:
                    print(f"Warning: demo stream GET {url} -> {resp.status_code}")
                    return None
                final_name = resolve_demo_filename(resp, base_name)
                known_exts = (".zip", ".rar", ".gz", ".7z", ".dem")
                iter_stream = resp.iter_content(chunk_size=1024 * 256)
                # Peek first bytes for magic
                first_chunk = b""
                for chunk in iter_stream:
                    if chunk:
                        first_chunk = chunk
                        break
                detected_ext = detect_ext_from_magic(first_chunk)
                if not any(final_name.lower().endswith(ext) for ext in known_exts):
                    if detected_ext:
                        final_name = final_name + detected_ext
                dest_path = os.path.join(demos_dir, final_name)
                tmp_path = dest_path + ".part"
                downloaded = 0
                with open(tmp_path, "wb") as f:
                    if first_chunk:
                        f.write(first_chunk)
                        downloaded += len(first_chunk)
                    for chunk in iter_stream:
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                # Verify integrity: size and magic
                min_bytes = 1_000_000  # 1MB
                ok = downloaded >= min_bytes
                if ok:
                    try:
                        with open(tmp_path, "rb") as vf:
                            head = vf.read(16)
                        if not detect_ext_from_magic(head):
                            ok = False
                    except Exception:
                        ok = False
                if not ok:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                    print(f"Warning: downloaded demo invalid (small or not archive): {url}")
                    return None
                os.replace(tmp_path, dest_path)
                if DEBUG:
                    total_hdr = resp.headers.get("Content-Length", "?")
                    print(f"[debug] Saved demo: {dest_path} ({downloaded} bytes; header {total_hdr})")
                return dest_path

        # If content-type indicates archive, directly stream to file
        if any(x in ctype for x in ("application/x-rar", "application/zip", "application/x-zip", "application/octet-stream")) and content_len > 0:
            return stream_download(probe.url, default_basename)

        # If looks like HTML, try to resolve final link
        text_sample = probe.text[:4096].lower() if ("html" in ctype or content_len < 2048) else ""
        final_url = None
        if text_sample:
            # meta refresh
            try:
                psoup = BeautifulSoup(probe.text, "html.parser")
                meta = psoup.find("meta", attrs={"http-equiv": lambda v: v and v.lower() == "refresh"})
                if meta and meta.get("content"):
                    content = meta["content"]
                    if "url=" in content.lower():
                        final_url = urljoin(probe.url, content.split("url=")[-1].strip())
                if not final_url:
                    # direct archive anchors
                    a = psoup.find("a", href=lambda h: h and any(h.lower().endswith(ext) for ext in (".rar", ".zip", ".7z", ".gz", ".dem")))
                    if a and a.get("href"):
                        final_url = urljoin(probe.url, a.get("href"))
            except Exception:
                final_url = None
        if final_url:
            return stream_download(final_url, default_basename)

        # Fallback: attempt to stream original URL anyway
        return stream_download(probe.url, default_basename)

    except Exception as e:
        print(f"Error downloading demo {demo_url}: {e}")
        return None


def fetch_match_data(match_url):
    match_id = extract_match_id_from_url(match_url)
    url = match_url if match_url.startswith("http") else (BASE + match_url)
    r = session.get(url, headers=HEADERS, timeout=20)
    if DEBUG:
        print(f"[debug] GET {url} -> {r.status_code}, {len(r.text)} bytes")
    if r.status_code != 200:
        return {"match_id": match_id, "error": f"status_{r.status_code}", "players": []}

    blocked = looks_like_block_page(r.text)
    soup = BeautifulSoup(r.text, "html.parser")

    teams = soup.select(".teamName")
    team1 = teams[0].text.strip() if teams else None
    team2 = teams[1].text.strip() if len(teams) > 1 else None

    score = None
    score_el = soup.select_one(".score")
    if score_el:
        score = score_el.get_text(strip=True).replace(" ", "")  # e.g. 3:1

    date = None
    date_el = soup.select_one(".date[data-unix]") or soup.select_one(".date")
    if date_el:
        date = date_el.get_text(strip=True)

    players = []
    lineup_links = soup.select("#lineups .lineup a[href^='/player/']")
    if DEBUG:
        print(f"[debug] lineup player links: {len(lineup_links)} for match {match_id}")
    for a in lineup_links:
        name = a.get_text(strip=True)
        if name:
            players.append({"name": name, "kd": None, "adr": None, "rating": None})

    demo_link = extract_demo_link(soup)

    if DEBUG:
        fname = f"match_{match_id or 'unknown'}{'_blocked' if blocked else ''}.html"
        debug_write(fname, r.text)

    return {
        "match_id": match_id,
        "date": date,
        "team1": team1,
        "team2": team2,
        "score": score,
        "demo_url": demo_link,
        "players": players,
        "blocked": blocked,
        "match_url": url,
    }


def scrape_and_save(pages=3, outfile="hltv_matches.csv", download_demos=False, demos_dir="demos"):
    links = get_match_links(pages)
    if not links:
        print("No match links found. Check _debug/results_offset0.html and your network/headers.")
        return
    # Limit to first 3 matches
    links = links[:3]
    all_data = []
    downloaded_count = 0
    max_demos = 3
    for link in tqdm(links, desc="Scraping matches"):
        data = fetch_match_data(link)
        all_data.append(data)
        if download_demos and data.get("demo_url") and downloaded_count < max_demos:
            result_path = download_demo_file(data["demo_url"], data.get("match_url") or link, data.get("match_id") or "", demos_dir)
            if result_path:
                downloaded_count += 1
        time.sleep(1)

    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["match_id", "date", "team1", "team2", "score", "player", "kd", "adr", "rating", "demo_url"])
        for m in all_data:
            if m.get("players"):
                for p in m["players"]:
                    writer.writerow([
                        m["match_id"], m.get("date"), m.get("team1"), m.get("team2"),
                        m.get("score"), p.get("name"), p.get("kd"), p.get("adr"), p.get("rating"),
                        m.get("demo_url")
                    ])


if __name__ == "__main__":
    # Hardcoded run settings
    DEBUG = True
    PAGES = 1
    OUTFILE = os.path.join(os.path.dirname(__file__), "hltv_matches.csv")
    DOWNLOAD_DEMOS = True
    DEMOS_DIR = os.path.join(os.path.dirname(__file__), "demos")

    if DEBUG:
        print("[debug] Debug mode enabled")
    scrape_and_save(pages=PAGES, outfile=OUTFILE, download_demos=DOWNLOAD_DEMOS, demos_dir=DEMOS_DIR)