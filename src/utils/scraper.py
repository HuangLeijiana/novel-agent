"""Platform ranking scraper — auto-fetch Feilu and Fanqie ranking data.

Feilu: Headless browser (Playwright) rendering of JS-rendered ranking pages.
Fanqie: Multi-category JSON API + SSR book ID fallback for broader coverage.
"""

import asyncio
import logging
import re
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Feilu ranking pages — JS-rendered, need Playwright
# Rank_1=标新榜(月票), Rank_2=总榜(点击), Rank_3=分类榜(收藏), Rank_4=女生榜(新书)
FEILU_RANK_URLS = [
    ("月票榜(标新榜)", "https://b.faloo.com/Rank_1.html"),
    ("总点击榜", "https://b.faloo.com/Rank_2.html"),
    ("收藏分类榜", "https://b.faloo.com/Rank_3.html"),
    ("女生新书榜", "https://b.faloo.com/Rank_4.html"),
]

FANQIE_RANK_API = "https://fanqienovel.com/api/rank/list"
FANQIE_BOOK_API = "https://fanqienovel.com/api/book"

# Fanqie rank categories to try (the API may or may not respect these)
FANQIE_RANK_CATEGORIES = [
    # (rank_type, gender, label)
    (1, 1, "新书榜·男频"),
    (1, 2, "新书榜·女频"),
    (2, 1, "阅读榜·男频"),
    (2, 2, "阅读榜·女频"),
    (3, 1, "热搜榜·男频"),
    (3, 2, "热搜榜·女频"),
    (7, 1, "推荐榜·男频"),
    (5, 1, "完结榜·男频"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


# ================================================================
# Feilu (飞卢)
# ================================================================


async def scrape_feilu_rank() -> Optional[str]:
    """Scrape Feilu (飞卢) ranking pages using headless Chromium.

    Feilu rank pages (b.faloo.com/Rank_*.html) are JavaScript-rendered.
    We use Playwright to render the page and extract book titles, authors,
    and rankings from the DOM after JS execution.

    Returns formatted text with book data, or a clear error message
    if scraping fails.
    """
    logger.info("Scraping Feilu ranking data via headless browser...")

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.error("Playwright not installed — cannot scrape Feilu")
        return (
            "===== 飞卢小说榜单数据（来源：飞卢排名页面）=====\n"
            "⚠️ 飞卢榜单需要使用浏览器渲染抓取，但Playwright未安装。\n"
            "请联系管理员安装: pip install playwright && playwright install chromium\n"
        )

    all_books: list[dict] = []
    seen_titles: set[str] = set()

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
            )
            page = await context.new_page()

            for list_name, url in FEILU_RANK_URLS:
                try:
                    logger.info(f"Feilu: navigating to {list_name} — {url}")
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                    # Extra wait for dynamic content to settle
                    await page.wait_for_timeout(3000)

                    # Extract all book links from the rendered page
                    # Strategy: find all <a> with book URLs, group by bookId,
                    # pick the one with longest text (image links have no text)
                    book_links = await page.evaluate("""() => {
                        const allLinks = document.querySelectorAll('a[href]');
                        // Group by bookId, keep track of best title and author
                        const bookMap = new Map();
                        for (const a of allLinks) {
                            const href = a.getAttribute('href') || '';
                            // Match book detail pages: /XXXXX.html where XXXXX is 5-8 digits
                            const match = href.match(/\\/(\\d{5,8})\\.html/);
                            if (!match) continue;
                            const bookId = match[1];
                            const text = (a.textContent || '').trim();
                            // Also check img alt as fallback
                            let title = text;
                            if (!title || title.length < 2) {
                                const img = a.querySelector('img');
                                if (img) {
                                    const alt = img.getAttribute('alt');
                                    if (alt && alt.length >= 2) title = alt;
                                }
                            }
                            // Check if this is an author link (child of book entry)
                            const parentDiv = a.closest('div, li, section');
                            const parentText = parentDiv ? (parentDiv.textContent || '') : '';
                            const isAuthorLink = href.includes('l_0_1.html');

                            if (isAuthorLink) {
                                // This is an author link — attach to the last book in this container
                                // Authors appear near book links in the same parent
                                if (parentDiv) {
                                    // Find all book entries in this parent and add author
                                    const bookLinks = parentDiv.querySelectorAll('a[href*=\".html\"]:not([href*=\"l_0_1\"])');
                                    for (const bl of bookLinks) {
                                        const bh = bl.getAttribute('href') || '';
                                        const bm = bh.match(/\\/(\\d{5,8})\\.html/);
                                        if (bm && bookMap.has(bm[1])) {
                                            const entry = bookMap.get(bm[1]);
                                            if (!entry.author) entry.author = text;
                                        }
                                    }
                                }
                                continue;
                            }

                            if (!title || title.length < 2) continue;

                            // Track the best entry per bookId (prefer ones with longer text)
                            if (bookMap.has(bookId)) {
                                const existing = bookMap.get(bookId);
                                if (title.length > existing.title.length) {
                                    existing.title = title;
                                }
                            } else {
                                // Extract rank from nearby text
                                let rank = 0;
                                const noMatch = parentText.match(/NO\\.(\\d+)/i);
                                if (noMatch) rank = parseInt(noMatch[1]);
                                bookMap.set(bookId, {
                                    bookId: bookId,
                                    title: title,
                                    rank: rank,
                                    author: '',
                                    url: 'https://b.faloo.com/' + bookId + '.html',
                                });
                            }
                        }
                        return Array.from(bookMap.values());
                    }""")

                    for book in book_links:
                        title = book["title"]
                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            book["list_name"] = list_name
                            all_books.append(book)

                    logger.info(f"Feilu {list_name}: extracted {len(book_links)} books")

                except Exception as e:
                    logger.warning(f"Feilu {list_name} scrape failed: {e}")

            await browser.close()
            logger.info(f"Feilu scrape complete: {len(all_books)} unique books across {len(FEILU_RANK_URLS)} pages")

    except Exception as e:
        logger.error(f"Feilu Playwright scrape failed: {e}")
        return (
            "===== 飞卢小说榜单数据（来源：飞卢排名页面）=====\n"
            f"⚠️ 飞卢榜单抓取失败：{e}\n"
            "请以番茄小说榜单数据为准进行分析。\n"
        )

    if not all_books:
        return (
            "===== 飞卢小说榜单数据（来源：飞卢排名页面）=====\n"
            "⚠️ 飞卢榜单抓取完成，但未提取到书本数据。\n"
            "请以番茄小说榜单数据为准进行分析。\n"
        )

    return _format_feilu_books(all_books)


def _format_feilu_books(books: list[dict]) -> str:
    """Format extracted Feilu book data as structured text."""
    # Group books by list
    by_list: dict[str, list[dict]] = {}
    list_order = []
    for b in books:
        ln = b.get("list_name", "其他")
        if ln not in by_list:
            by_list[ln] = []
            list_order.append(ln)
        by_list[ln].append(b)

    lines = [f"===== 飞卢小说榜单数据（共{len(books)}本，来源：浏览器实时抓取）=====\n"]
    for ln in list_order:
        items = by_list[ln]
        lines.append(f"--- {ln}（{len(items)}本）---")
        for i, b in enumerate(items, 1):
            rank = b.get("rank", i)
            title = b.get("title", "未知")
            author = b.get("author", "")
            url = b.get("url", "")
            rank_str = f" 排名#{rank}" if rank > 0 else ""
            author_str = f" 作者：{author}" if author else ""
            lines.append(f"#{i} 《{title}》{author_str}{rank_str}")
            if url:
                lines.append(f"    链接：{url}")
        lines.append("")
    return "\n".join(lines)


# ================================================================
# Fanqie (番茄)
# ================================================================


async def _fetch_fanqie_rank(rank_type: int, gender: int, limit: int = 10) -> list[dict]:
    """Fetch one page of a Fanqie JSON API ranking list."""
    headers = {**HEADERS, "Referer": "https://fanqienovel.com/rank"}
    books = []
    try:
        url = f"{FANQIE_RANK_API}?rank_type={rank_type}&offset=0&limit={limit}&gender={gender}&_t={int(time.time())}"
        async with httpx.AsyncClient(timeout=20, headers=headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return []
            for b in data.get("data", {}).get("list", []):
                bid = b.get("bookId", "")
                if bid:
                    books.append(b)
    except Exception as e:
        logger.debug(f"Fanqie rank_type={rank_type} gender={gender}: {e}")
    return books[:limit]


async def scrape_fanqie_rank() -> Optional[str]:
    """Scrape Fanqie (番茄) ranking data from multiple sources.

    Strategy (in order):
    1. Query JSON API with multiple rank_type/gender combinations.
       Deduplicate across all calls. Target: 15+ unique books.
    2. If API returns < 10 books, extract book IDs from SSR HTML
       (clean integers, not font-obfuscated) and resolve metadata
       via the book detail API.
    """
    logger.info("Scraping Fanqie ranking data...")

    # ---------------------------------------------------------------
    # Strategy 1: Multi-category API
    # ---------------------------------------------------------------
    seen_ids: set[str] = set()
    all_books: list[dict] = []

    async def _fetch_category(rank_type: int, gender: int, label: str) -> int:
        """Fetch one category, add new books to all_books. Returns count added."""
        books = await _fetch_fanqie_rank(rank_type=rank_type, gender=gender, limit=10)
        added = 0
        for b in books:
            bid = str(b.get("bookId", ""))
            if bid and bid not in seen_ids:
                seen_ids.add(bid)
                all_books.append(b)
                added += 1
        if added:
            logger.debug(f"Fanqie {label}: +{added} new books")
        return added

    # Query all categories concurrently
    tasks = [_fetch_category(rt, g, lbl) for rt, g, lbl in FANQIE_RANK_CATEGORIES]
    await asyncio.gather(*tasks)

    logger.info(
        f"Fanqie multi-category API: {len(all_books)} unique books from {len(FANQIE_RANK_CATEGORIES)} categories"
    )

    # ---------------------------------------------------------------
    # Strategy 2: SSR book ID extraction + detail API lookup
    # ---------------------------------------------------------------
    if len(all_books) < 10:
        logger.info(f"Fanqie API returned only {len(all_books)} books, trying SSR fallback...")
        ssr_books = await _scrape_fanqie_ssr_ids(seen_ids)
        if ssr_books:
            all_books.extend(ssr_books)
            logger.info(f"Fanqie SSR fallback: +{len(ssr_books)} books, total={len(all_books)}")

    if all_books:
        return _format_fanqie_books(all_books)

    logger.warning("Fanqie: no books from any source")
    return None


async def _scrape_fanqie_html() -> Optional[str]:
    """Fetch the Fanqie ranking page HTML (contains SSR data)."""
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=HEADERS) as client:
            resp = await client.get(f"https://fanqienovel.com/rank?_t={int(time.time())}")
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        logger.warning(f"Fanqie HTML fetch failed: {e}")
        return None


def _extract_fanqie_ssr_book_ids(html: Optional[str]) -> list[str]:
    """Extract clean book IDs from Fanqie's SSR __INITIAL_STATE__ JSON.

    Book IDs are integers — they are NOT affected by the custom font
    obfuscation that garbles book titles in the SSR HTML.
    """
    if not html:
        return []

    import json

    # Find the start of __INITIAL_STATE__
    start_marker = "window.__INITIAL_STATE__={"
    idx = html.find(start_marker)
    if idx >= 0:
        pos = idx + len(start_marker) - 1
    else:
        start_marker = "window.__INITIAL_STATE__ = "
        idx = html.find(start_marker)
        if idx >= 0:
            pos = idx + len(start_marker)
        else:
            return []

    if pos >= len(html) or html[pos] != "{":
        return []

    # Brace counting to find matching closing brace
    depth = 0
    end = pos
    for i in range(pos, len(html)):
        ch = html[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    json_str = html[pos:end]
    try:
        data = json.loads(json_str)
        book_list = data.get("rank", {}).get("book_list", [])
        if not isinstance(book_list, list):
            return []
        ids = []
        for b in book_list:
            bid = str(b.get("bookId", ""))
            if bid:
                ids.append(bid)
        return ids
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Fanqie SSR book ID extraction failed: {e}")
        return []


async def _resolve_fanqie_book_detail(book_id: str) -> Optional[dict]:
    """Look up a single book's clean metadata via the Fanqie book detail API."""
    headers = {**HEADERS, "Referer": f"https://fanqienovel.com/page/{book_id}"}
    try:
        url = f"{FANQIE_BOOK_API}/info?bookId={book_id}"
        async with httpx.AsyncClient(timeout=15, headers=headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return None
            info = data.get("data", {})
            return {
                "bookId": book_id,
                "bookName": info.get("bookName", info.get("title", "")),
                "author": info.get("author", ""),
                "abstract": (info.get("abstract", info.get("description", "")) or "")[:200],
            }
    except Exception as e:
        logger.debug(f"Fanqie book detail {book_id}: {e}")
        return None


async def _scrape_fanqie_ssr_ids(existing_ids: set[str]) -> list[dict]:
    """Extract book IDs from SSR HTML, resolve clean metadata via detail API.

    Only fetches books whose IDs are not already in existing_ids.
    Returns list of book dicts with clean UTF-8 metadata.
    """
    html = await _scrape_fanqie_html()
    if not html:
        return []

    ssr_ids = _extract_fanqie_ssr_book_ids(html)
    if not ssr_ids:
        logger.warning("Fanqie SSR: no book IDs found")
        return []

    # Filter to new IDs only
    new_ids = [bid for bid in ssr_ids if bid not in existing_ids]
    if not new_ids:
        logger.info("Fanqie SSR: all book IDs already covered by API")
        return []

    logger.info(f"Fanqie SSR: {len(new_ids)} new book IDs to resolve")

    # Resolve book details concurrently (max 5 at a time to be polite)
    sem = asyncio.Semaphore(5)

    async def _resolve_one(bid: str) -> Optional[dict]:
        async with sem:
            return await _resolve_fanqie_book_detail(bid)

    tasks = [_resolve_one(bid) for bid in new_ids[:15]]
    results = await asyncio.gather(*tasks)

    books = [r for r in results if r and r.get("bookName")]
    logger.info(f"Fanqie SSR: resolved {len(books)}/{len(new_ids[:15])} book details")
    return books


def _format_fanqie_books(books: list[dict]) -> str:
    """Format extracted Fanqie book data as structured text."""
    lines = [f"===== 番茄小说榜单数据（共{len(books)}本，来源：综合热榜）=====\n"]
    for i, b in enumerate(books):
        name = b.get("bookName", b.get("title", "未知"))
        author = b.get("author", "未知")
        abstract = (b.get("abstract", b.get("description", "")) or "")[:200]
        lines.append(f"#{i + 1} 《{name}》 作者：{author}")
        if abstract:
            lines.append(f"    简介：{abstract}")
    return "\n".join(lines)


async def scrape_all() -> dict[str, Optional[str]]:
    """Scrape both platforms concurrently."""
    import asyncio

    feilu_task = scrape_feilu_rank()
    fanqie_task = scrape_fanqie_rank()
    feilu_html, fanqie_html = await asyncio.gather(feilu_task, fanqie_task)
    return {"feilu": feilu_html, "fanqie": fanqie_html}
