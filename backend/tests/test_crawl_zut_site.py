from __future__ import annotations

from pathlib import Path

import httpx

from scripts.crawl_zut_site import crawl_site, load_existing_documents, merge_document_indexes, normalize_url, parse_html_page, render_markdown


def test_load_existing_documents_should_index_source_urls(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "README.md").write_text("# index\n", encoding="utf-8")
    (docs_dir / "08-招生章程.md").write_text(
        "# 原文（来源：https://zsc.zut.edu.cn/info/1124/2673.htm）\n网页标题：招生章程\n抓取时间：2026-03-30\n",
        encoding="utf-8",
    )

    source_map, used_names, max_index = load_existing_documents(docs_dir)

    assert normalize_url("https://zsc.zut.edu.cn/info/1124/2673.htm") in source_map
    assert "08-招生章程.md".lower() in used_names
    assert max_index == 8


def test_merge_document_indexes_should_include_reference_dirs(tmp_path):
    docs_dir = tmp_path / "docs"
    review_dir = tmp_path / "review"
    docs_dir.mkdir()
    review_dir.mkdir()
    (docs_dir / "08-招生章程.md").write_text(
        "# 原文（来源：https://zsc.zut.edu.cn/info/1124/2673.htm）\n网页标题：招生章程\n抓取时间：2026-03-30\n",
        encoding="utf-8",
    )
    (review_dir / "09-学校概况.md").write_text(
        "# 原文（来源：https://www.zut.edu.cn/xxgk/xxjj.htm）\n网页标题：学校概况\n抓取时间：2026-03-30\n",
        encoding="utf-8",
    )

    source_map, used_names, max_index = merge_document_indexes([docs_dir, review_dir])

    assert normalize_url("https://zsc.zut.edu.cn/info/1124/2673.htm") in source_map
    assert normalize_url("https://www.zut.edu.cn/xxgk/xxjj.htm") in source_map
    assert "08-招生章程.md".lower() in used_names
    assert "09-学校概况.md".lower() in used_names
    assert max_index == 9


def test_parse_html_page_should_extract_title_text_links_and_publish_date():
    html = """
    <html>
      <head>
        <title>招生信息网</title>
        <meta name="publishdate" content="2026-03-29 08:00:00" />
      </head>
      <body>
        <div>当前位置</div>
        <h1>中原工学院 2026 招生公告</h1>
        <p>发布时间：2026-03-29</p>
        <p>欢迎报考。</p>
        <a href="/info/1000/1.htm">详情</a>
        <a href="javascript:void(0)">忽略</a>
        <script>window.alert("skip")</script>
      </body>
    </html>
    """

    page = parse_html_page("https://zsc.zut.edu.cn/index.htm", html, ("zut.edu.cn",))
    markdown = render_markdown(page, grabbed_at="2026-03-30")

    assert page.title == "招生信息网"
    assert page.publish_date == "2026-03-29"
    assert "中原工学院 2026 招生公告" in page.text
    assert "https://zsc.zut.edu.cn/info/1000/1.htm" in page.links
    assert markdown.startswith("# 原文（来源：https://zsc.zut.edu.cn/index.htm）")
    assert "网页标题：招生信息网" in markdown
    assert "发布时间：2026-03-29" in markdown
    assert "抓取时间：2026-03-30" in markdown


def test_crawl_site_should_skip_existing_write_but_continue_discovery(tmp_path):
    output_dir = tmp_path / "zyit"
    output_dir.mkdir()
    existing_path = output_dir / "01-学校概况.md"
    existing_path.write_text(
        "# 原文（来源：https://www.zut.edu.cn/）\n网页标题：学校概况\n抓取时间：2026-03-29\n已有内容\n",
        encoding="utf-8",
    )

    responses = {
        "https://www.zut.edu.cn/": """
        <html><head><title>首页</title></head><body>
        <a href="/info/1000/1.htm">招生公告</a>
        </body></html>
        """,
        "https://www.zut.edu.cn/info/1000/1.htm": """
        <html><head><title>招生公告</title></head><body>
        <h1>中原工学院招生公告</h1>
        <p>发布时间：2026-03-28</p>
        <p>这是新增页面。</p>
        </body></html>
        """,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url not in responses:
            return httpx.Response(status_code=404, text="not found")
        return httpx.Response(
            status_code=200,
            text=responses[url],
            headers={"content-type": "text/html; charset=utf-8"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True, trust_env=False) as client:
        stats = crawl_site(
            client=client,
            output_dir=output_dir,
            reference_dirs=[output_dir],
            seed_urls=["https://www.zut.edu.cn/"],
            domain_suffixes=("zut.edu.cn",),
            delay_seconds=0.0,
            max_pages=10,
            discover_from_existing=True,
            use_sitemaps=False,
        )

    written_files = sorted(path for path in output_dir.glob("*.md") if path != existing_path)

    assert stats.skipped_existing == 1
    assert stats.written_pages == 1
    assert existing_path.read_text(encoding="utf-8").startswith("# 原文（来源：https://www.zut.edu.cn/）")
    assert len(written_files) == 1
    assert "中原工学院招生公告" in written_files[0].read_text(encoding="utf-8")
    assert "发布时间：2026-03-28" in written_files[0].read_text(encoding="utf-8")


def test_crawl_site_should_skip_urls_existing_in_reference_dir(tmp_path):
    reference_dir = tmp_path / "docs"
    output_dir = tmp_path / "review"
    reference_dir.mkdir()
    output_dir.mkdir()
    (reference_dir / "01-学校概况.md").write_text(
        "# 原文（来源：https://www.zut.edu.cn/xxgk/xxjj.htm）\n网页标题：学校概况\n抓取时间：2026-03-29\n已有内容\n",
        encoding="utf-8",
    )

    responses = {
        "https://www.zut.edu.cn/xxgk/xxjj.htm": """
        <html><head><title>学校概况</title></head><body>
        <a href="/info/1000/1.htm">招生公告</a>
        </body></html>
        """,
        "https://www.zut.edu.cn/info/1000/1.htm": """
        <html><head><title>招生公告</title></head><body>
        <h1>中原工学院招生公告</h1>
        <p>发布时间：2026-03-28</p>
        </body></html>
        """,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url not in responses:
            return httpx.Response(status_code=404, text="not found")
        return httpx.Response(status_code=200, text=responses[url], headers={"content-type": "text/html; charset=utf-8"})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True, trust_env=False) as client:
        stats = crawl_site(
            client=client,
            output_dir=output_dir,
            reference_dirs=[reference_dir],
            seed_urls=["https://www.zut.edu.cn/xxgk/xxjj.htm"],
            domain_suffixes=("zut.edu.cn",),
            delay_seconds=0.0,
            max_pages=10,
            discover_from_existing=True,
            use_sitemaps=False,
        )

    generated_files = sorted(output_dir.glob("*.md"))
    assert stats.skipped_existing == 1
    assert stats.written_pages == 1
    assert len(generated_files) == 1
    assert generated_files[0].name.startswith("02-")
