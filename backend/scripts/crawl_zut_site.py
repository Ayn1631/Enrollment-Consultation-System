from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
import time
from typing import Iterable, Sequence
from urllib.parse import urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

import httpx


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "review" / "zyit_pending"
DEFAULT_REFERENCE_DIR = ROOT_DIR / "docs" / "zyit"
DEFAULT_DOMAIN_SUFFIXES = ("zut.edu.cn",)
DEFAULT_SEED_URLS = (
    "https://www.zut.edu.cn/",
    "https://zsc.zut.edu.cn/",
    "https://hq.zut.edu.cn/",
    "https://lib.zut.edu.cn/",
    "https://xljk.zut.edu.cn/",
    "https://job.zut.edu.cn/",
)
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_DELAY_SECONDS = 0.15
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/132.0.0.0 Safari/537.36 CodexZutCrawler/1.0"
)
SOURCE_LINE_PATTERN = re.compile(r"^# 原文（来源：(.*?)）$")
INDEXED_NAME_PATTERN = re.compile(r"^(\d+)-")
DATE_PATTERN = re.compile(r"(20\d{2}-\d{2}-\d{2})")
COMMON_PUBLISH_PATTERNS = (
    re.compile(r"发布时间[:：]\s*(20\d{2}-\d{2}-\d{2})"),
    re.compile(r"发布(?:日期|时间)[:：]\s*(20\d{2}-\d{2}-\d{2})"),
    re.compile(r"(20\d{2}-\d{2}-\d{2})\s*(?:来源|作者|浏览|点击|阅读量)"),
)
NON_HTML_SUFFIXES = {
    ".7z",
    ".apk",
    ".avi",
    ".bmp",
    ".csv",
    ".doc",
    ".docm",
    ".docx",
    ".exe",
    ".flv",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".rss",
    ".svg",
    ".ts",
    ".txt",
    ".xls",
    ".xlsm",
    ".xlsx",
    ".xml",
    ".zip",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="递归抓取中原工学院官网网页并按现有 Markdown 格式落盘。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Markdown 文档输出目录。")
    parser.add_argument(
        "--reference-dir",
        action="append",
        dest="reference_dirs",
        help="已存在文档参考目录，可重复传入；其中来源 URL 会被视为已抓取并跳过写入。",
    )
    parser.add_argument("--seed-url", action="append", dest="seed_urls", help="可重复传入起始 URL；默认使用内置站点入口。")
    parser.add_argument("--domain-suffix", action="append", dest="domain_suffixes", help="允许递归的域名后缀；默认 zut.edu.cn。")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="单次请求超时时间（秒）。")
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS, help="每次请求后的等待秒数。")
    parser.add_argument("--max-pages", type=int, default=0, help="最多抓取多少个网页；0 表示不限。")
    parser.add_argument("--no-sitemap", action="store_true", help="禁用 robots/sitemap 预扩展。")
    parser.add_argument("--skip-existing-discovery", action="store_true", help="本地已存在时直接跳过，不再远端取回链接。")
    return parser


def normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    scheme = (parts.scheme or "https").lower()
    hostname = (parts.hostname or "").lower()
    if not hostname:
        return ""
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    query = parts.query.strip()
    return urlunsplit((scheme, netloc, path or "/", query, ""))


def is_allowed_url(url: str, domain_suffixes: Sequence[str]) -> bool:
    normalized = normalize_url(url)
    if not normalized:
        return False
    hostname = urlsplit(normalized).hostname or ""
    return any(hostname == suffix or hostname.endswith(f".{suffix}") for suffix in domain_suffixes)


def is_html_candidate(url: str) -> bool:
    normalized = normalize_url(url)
    if not normalized:
        return False
    parts = urlsplit(normalized)
    if parts.scheme not in {"http", "https"}:
        return False
    lowered_path = parts.path.lower()
    if any(lowered_path.endswith(suffix) for suffix in NON_HTML_SUFFIXES):
        return False
    return True


def sanitize_filename_component(value: str) -> str:
    compact = re.sub(r"\s+", " ", unescape(value or "")).strip()
    compact = re.sub(r'[<>:"/\\\\|?*]', " ", compact)
    compact = compact.replace("\u3000", " ")
    compact = re.sub(r"\s+", " ", compact).strip(" .")
    return compact[:80] if compact else ""


def derive_title_from_url(url: str) -> str:
    parts = urlsplit(url)
    tail = parts.path.rstrip("/").split("/")[-1] if parts.path else ""
    if tail:
        return sanitize_filename_component(tail)
    return sanitize_filename_component(parts.netloc)


def extract_publish_date(*, text: str, html: str, meta_candidates: Iterable[str]) -> str:
    for candidate in meta_candidates:
        matched = DATE_PATTERN.search(candidate or "")
        if matched:
            return matched.group(1)
    merged = "\n".join([text, html[:4000]])
    for pattern in COMMON_PUBLISH_PATTERNS:
        matched = pattern.search(merged)
        if matched:
            return matched.group(1)
    return ""


def normalize_text_blocks(chunks: Iterable[str]) -> str:
    lines: list[str] = []
    previous_blank = False
    for raw in chunks:
        line = unescape(raw).replace("\xa0", " ")
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            if previous_blank:
                continue
            previous_blank = True
            lines.append("")
            continue
        previous_blank = False
        lines.append(line)
    return "\n".join(lines).strip()


class HtmlDocumentParser(HTMLParser):
    block_tags = {
        "article",
        "aside",
        "br",
        "div",
        "dl",
        "dt",
        "dd",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "tr",
        "ul",
    }
    ignored_tags = {"script", "style", "noscript", "svg", "iframe"}

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: set[str] = set()
        self.text_chunks: list[str] = []
        self.title_parts: list[str] = []
        self.meta_candidates: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): (value or "") for key, value in attrs}
        lowered = tag.lower()
        if lowered in self.ignored_tags:
            self._ignored_depth += 1
            return
        if lowered == "title":
            self._in_title = True
            return
        if lowered == "a":
            href = attr_map.get("href", "").strip()
            if href:
                self.links.add(href)
        if lowered == "meta":
            key = " ".join(filter(None, [attr_map.get("name", ""), attr_map.get("property", ""), attr_map.get("http-equiv", "")])).lower()
            if any(token in key for token in ("publish", "date", "time", "updated", "article:published_time")):
                self.meta_candidates.append(attr_map.get("content", ""))
        if lowered in self.block_tags:
            self.text_chunks.append("")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in self.ignored_tags and self._ignored_depth > 0:
            self._ignored_depth -= 1
            return
        if lowered == "title":
            self._in_title = False
            return
        if lowered in self.block_tags:
            self.text_chunks.append("")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        self.text_chunks.append(data)


@dataclass
class ParsedPage:
    url: str
    title: str
    publish_date: str
    text: str
    links: list[str]


@dataclass
class CrawlStats:
    fetched_pages: int = 0
    written_pages: int = 0
    skipped_existing: int = 0
    skipped_non_html: int = 0
    skipped_empty: int = 0
    failed_pages: int = 0
    discovered_urls: int = 0
    written_files: list[Path] = field(default_factory=list)
    failed_urls: list[str] = field(default_factory=list)


def parse_html_page(url: str, html: str, domain_suffixes: Sequence[str]) -> ParsedPage:
    parser = HtmlDocumentParser(base_url=url)
    parser.feed(html)
    parser.close()
    title = sanitize_filename_component("".join(parser.title_parts)) or derive_title_from_url(url)
    text = normalize_text_blocks(parser.text_chunks)
    publish_date = extract_publish_date(text=text, html=html, meta_candidates=parser.meta_candidates)
    links = sorted(
        {
            normalized
            for raw_link in parser.links
            for normalized in [normalize_url(urljoin(url, raw_link))]
            if normalized and is_allowed_url(normalized, domain_suffixes) and is_html_candidate(normalized)
        }
    )
    return ParsedPage(
        url=normalize_url(url),
        title=title,
        publish_date=publish_date,
        text=text,
        links=links,
    )


def render_markdown(page: ParsedPage, grabbed_at: str) -> str:
    lines = [
        f"# 原文（来源：{page.url}）",
        f"网页标题：{page.title}",
    ]
    if page.publish_date:
        lines.append(f"发布时间：{page.publish_date}")
    lines.append(f"抓取时间：{grabbed_at}")
    lines.append("")
    lines.append(page.text.strip())
    return "\n".join(lines).strip() + "\n"


def load_existing_documents(output_dir: Path) -> tuple[dict[str, Path], set[str], int]:
    source_map: dict[str, Path] = {}
    used_names: set[str] = set()
    max_index = 0
    if not output_dir.exists():
        return source_map, used_names, max_index
    for path in sorted(output_dir.glob("*.md")):
        used_names.add(path.name.lower())
        if path.name.lower() == "readme.md":
            continue
        matched_index = INDEXED_NAME_PATTERN.match(path.stem)
        if matched_index:
            max_index = max(max_index, int(matched_index.group(1)))
        try:
            first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
        except Exception:
            continue
        matched_source = SOURCE_LINE_PATTERN.match(first_line)
        if matched_source:
            normalized = normalize_url(matched_source.group(1))
            if normalized:
                source_map[normalized] = path
    return source_map, used_names, max_index


def merge_document_indexes(paths: Sequence[Path]) -> tuple[dict[str, Path], set[str], int]:
    merged_sources: dict[str, Path] = {}
    merged_names: set[str] = set()
    max_index = 0
    for path in paths:
        source_map, used_names, current_max = load_existing_documents(path)
        merged_sources.update(source_map)
        merged_names.update(used_names)
        max_index = max(max_index, current_max)
    return merged_sources, merged_names, max_index


def reserve_output_path(output_dir: Path, used_names: set[str], index: int, title: str, url: str) -> Path:
    label = sanitize_filename_component(title) or derive_title_from_url(url) or "未命名页面"
    while True:
        candidate = output_dir / f"{index:02d}-{label}.md"
        lowered = candidate.name.lower()
        if lowered not in used_names and not candidate.exists():
            used_names.add(lowered)
            return candidate
        index += 1


def is_html_response(response: httpx.Response, url: str) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    if "text/html" in content_type:
        return True
    if "application/xhtml+xml" in content_type:
        return True
    if not content_type:
        return is_html_candidate(url)
    return False


def upgrade_to_https(url: str) -> str:
    normalized = normalize_url(url)
    if not normalized:
        return ""
    parts = urlsplit(normalized)
    if parts.scheme != "http":
        return normalized
    return urlunsplit(("https", parts.netloc, parts.path, parts.query, ""))


def fetch_page(client: httpx.Client, url: str) -> httpx.Response:
    last_error: Exception | None = None
    candidates = [normalize_url(url)]
    https_candidate = upgrade_to_https(url)
    if https_candidate and https_candidate not in candidates:
        candidates.append(https_candidate)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            response = client.get(candidate, headers={"User-Agent": DEFAULT_USER_AGENT})
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("no valid request candidates")


def extract_sitemap_urls(client: httpx.Client, seed_urls: Sequence[str], domain_suffixes: Sequence[str]) -> list[str]:
    discovered: set[str] = set()
    queued_sitemaps: deque[str] = deque()
    processed_sitemaps: set[str] = set()
    site_roots = {
        urlunsplit((urlsplit(seed).scheme or "https", urlsplit(seed).netloc, "", "", ""))
        for seed in seed_urls
        if normalize_url(seed)
    }
    for site_root in sorted(site_roots):
        try:
            response = client.get(f"{site_root}/robots.txt")
            if response.status_code >= 400:
                continue
            for line in response.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    candidate = normalize_url(line.split(":", 1)[1].strip())
                    if candidate and candidate not in processed_sitemaps:
                        queued_sitemaps.append(candidate)
        except Exception:
            continue

    while queued_sitemaps:
        sitemap_url = queued_sitemaps.popleft()
        if sitemap_url in processed_sitemaps:
            continue
        processed_sitemaps.add(sitemap_url)
        try:
            response = client.get(sitemap_url)
            response.raise_for_status()
            root = ElementTree.fromstring(response.text)
        except Exception:
            continue
        loc_nodes = [node.text.strip() for node in root.findall(".//{*}loc") if node.text and node.text.strip()]
        for loc in loc_nodes:
            normalized = normalize_url(loc)
            if not normalized or not is_allowed_url(normalized, domain_suffixes):
                continue
            if normalized.endswith(".xml"):
                if normalized not in processed_sitemaps:
                    queued_sitemaps.append(normalized)
                continue
            if is_html_candidate(normalized):
                discovered.add(normalized)
    return sorted(discovered)


def crawl_site(
    *,
    client: httpx.Client,
    output_dir: Path,
    reference_dirs: Sequence[Path],
    seed_urls: Sequence[str],
    domain_suffixes: Sequence[str],
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    max_pages: int = 0,
    discover_from_existing: bool = True,
    use_sitemaps: bool = True,
) -> CrawlStats:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_paths = [*reference_dirs, output_dir]
    existing_sources, used_names, max_index = merge_document_indexes(index_paths)
    next_index = max_index + 1 if max_index else 1
    queued: deque[str] = deque()
    scheduled: set[str] = set()
    stats = CrawlStats()

    def enqueue(url: str) -> None:
        normalized = normalize_url(url)
        if not normalized:
            return
        if not is_allowed_url(normalized, domain_suffixes):
            return
        if not is_html_candidate(normalized):
            return
        if normalized in scheduled:
            return
        scheduled.add(normalized)
        queued.append(normalized)

    for seed in seed_urls:
        enqueue(seed)
    for existing_url in existing_sources:
        enqueue(existing_url)
    if use_sitemaps:
        for sitemap_url in extract_sitemap_urls(client=client, seed_urls=seed_urls, domain_suffixes=domain_suffixes):
            enqueue(sitemap_url)
    stats.discovered_urls = len(scheduled)

    while queued:
        if max_pages > 0 and stats.fetched_pages >= max_pages:
            break
        url = queued.popleft()
        should_fetch = discover_from_existing or url not in existing_sources
        if not should_fetch:
            stats.skipped_existing += 1
            continue
        try:
            response = fetch_page(client=client, url=url)
            stats.fetched_pages += 1
        except Exception:
            stats.failed_pages += 1
            stats.failed_urls.append(url)
            continue
        if not is_html_response(response, url):
            stats.skipped_non_html += 1
            continue
        parsed_page = parse_html_page(url=str(response.url), html=response.text, domain_suffixes=domain_suffixes)
        for linked_url in parsed_page.links:
            enqueue(linked_url)
        stats.discovered_urls = len(scheduled)
        if url in existing_sources:
            stats.skipped_existing += 1
        elif not parsed_page.text:
            stats.skipped_empty += 1
        else:
            output_path = reserve_output_path(
                output_dir=output_dir,
                used_names=used_names,
                index=next_index,
                title=parsed_page.title,
                url=parsed_page.url,
            )
            matched = INDEXED_NAME_PATTERN.match(output_path.stem)
            if matched:
                next_index = int(matched.group(1)) + 1
            output_path.write_text(
                render_markdown(parsed_page, grabbed_at=datetime.now().date().isoformat()),
                encoding="utf-8",
            )
            existing_sources[parsed_page.url] = output_path
            stats.written_pages += 1
            stats.written_files.append(output_path)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return stats


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    reference_dirs = [Path(item).resolve() for item in (args.reference_dirs or [str(DEFAULT_REFERENCE_DIR)])]
    seed_urls = list(args.seed_urls or DEFAULT_SEED_URLS)
    domain_suffixes = tuple(args.domain_suffixes or DEFAULT_DOMAIN_SUFFIXES)
    with httpx.Client(timeout=args.timeout, follow_redirects=True, trust_env=False) as client:
        stats = crawl_site(
            client=client,
            output_dir=output_dir,
            reference_dirs=reference_dirs,
            seed_urls=seed_urls,
            domain_suffixes=domain_suffixes,
            delay_seconds=max(args.delay_seconds, 0.0),
            max_pages=max(args.max_pages, 0),
            discover_from_existing=not args.skip_existing_discovery,
            use_sitemaps=not args.no_sitemap,
        )
    print(f"输出目录: {output_dir}")
    print("参考目录:")
    for path in reference_dirs:
        print(f"- {path}")
    print(f"已发现 URL: {stats.discovered_urls}")
    print(f"已请求页面: {stats.fetched_pages}")
    print(f"新增文档: {stats.written_pages}")
    print(f"跳过已存在: {stats.skipped_existing}")
    print(f"跳过非 HTML: {stats.skipped_non_html}")
    print(f"跳过空内容: {stats.skipped_empty}")
    print(f"请求失败: {stats.failed_pages}")
    if stats.written_files:
        print("新增文件:")
        for path in stats.written_files[:20]:
            print(f"- {path}")
        if len(stats.written_files) > 20:
            print(f"- ... 共 {len(stats.written_files)} 个")
    if stats.failed_urls:
        print("失败 URL（前 20 条）:")
        for url in stats.failed_urls[:20]:
            print(f"- {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
