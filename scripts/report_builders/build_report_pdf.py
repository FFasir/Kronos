# -*- coding: utf-8 -*-
"""把 CICC 风格 HTML 报告导出为 PDF（自包含，图片已内嵌）。

用法：python scripts/report_builders/build_report_pdf.py
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parents[2] / "docs" / "训练记录"
SRC = DOCS / "20260908_Kronos择时_中证风格报告.html"
OUT = DOCS / "20260908_Kronos择时_报告.pdf"


def main() -> None:
    if not SRC.is_file():
        print(f"缺少输入：{SRC}", file=sys.stderr)
        sys.exit(1)
    url = SRC.resolve().as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="load")
        page.pdf(
            path=str(OUT),
            format="A4",
            print_background=True,
            margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"},
        )
        browser.close()
    print(f"written {OUT} {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
