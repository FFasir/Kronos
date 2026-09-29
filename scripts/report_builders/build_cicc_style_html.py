# -*- coding: utf-8 -*-
"""从 md 报告生成 CICC（中证）风格自包含 HTML —— 保证 md 与 html 始终一致。

用法：python scripts/report_builders/build_cicc_style_html.py
输入：docs/训练记录/20260908_Kronos择时_报告.md（图片引用 figs_cicc/*.png）
输出：docs/训练记录/20260908_Kronos择时_中证风格报告.html（图片以 base64 内嵌）
"""
import base64
import html as html_mod
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs" / "训练记录"
SRC = DOCS / "20260908_Kronos择时_报告.md"
OUT = DOCS / "20260908_Kronos择时_中证风格报告.html"

CSS = """
body{font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 20px;color:#222;line-height:1.75}
h1{font-size:21px;text-align:center;margin:6px 0 4px;color:#1f4e79}
h2{font-size:17px;color:#1f4e79;border-left:5px solid #1f4e79;padding-left:9px;margin-top:32px}
h3{font-size:15px;color:#2f5f8f;margin:20px 0 6px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}
th,td{border:1px solid #ccc;padding:6px 8px;text-align:right}
th{background:#eef3fa;text-align:center;font-weight:600}
td:first-child,th:first-child{text-align:left}
tr:nth-child(even){background:#f8fafc}
img{max-width:100%;border:1px solid #ddd;border-radius:3px;margin:8px 0}
blockquote{background:#f4f7fb;border-left:4px solid #9ec5ea;margin:12px 0;padding:8px 14px;color:#444;font-size:13.5px}
ul,ol{margin:8px 0 8px 22px;padding:0}
li{margin:3px 0}
p{margin:9px 0}
code{background:#eef1f4;border-radius:3px;padding:1px 5px;font-size:12.5px}
hr{border:0;border-top:1px solid #ddd;margin:26px 0}
.disc{color:#888;font-size:12px;border-top:1px solid #ddd;margin-top:30px;padding-top:10px}
"""


def inline(text: str) -> str:
    """行内标记：**粗体**、`代码`。先转义，再替换。"""
    out = html_mod.escape(text, quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    return out


def img_tag(path_rel: str, alt: str) -> str:
    p = (DOCS / path_rel).resolve()
    if not p.is_file():
        return f'<p style="color:#c00">（缺图：{path_rel}）</p>'
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="{html_mod.escape(alt)}">'


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 表格
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            header = [c.strip() for c in stripped.strip("|").split("|")]
            out.append("<table>")
            out.append("<tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr>")
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</table>")
            continue

        # 图片
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if m:
            out.append(img_tag(m.group(2), m.group(1)))
            i += 1
            continue

        # 标题
        if stripped.startswith("### "):
            out.append(f"<h3>{inline(stripped[4:])}</h3>")
            i += 1
            continue
        if stripped.startswith("## "):
            out.append(f"<h2>{inline(stripped[3:])}</h2>")
            i += 1
            continue
        if stripped.startswith("# "):
            out.append(f"<h1>{inline(stripped[2:])}</h1>")
            i += 1
            continue

        # 引用
        if stripped.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + "<br>".join(inline(b) for b in buf) + "</blockquote>")
            continue

        # 有序/无序列表
        if re.match(r"^\d+\.\s", stripped) or stripped.startswith("- "):
            ordered = bool(re.match(r"^\d+\.\s", stripped))
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>")
            while i < len(lines):
                s = lines[i].strip()
                if re.match(r"^\d+\.\s", s):
                    out.append(f"<li>{inline(re.sub(r'^\d+\.\s', '', s))}</li>")
                elif s.startswith("- "):
                    out.append(f"<li>{inline(s[2:])}</li>")
                else:
                    break
                i += 1
            out.append(f"</{tag}>")
            continue

        if stripped == "---":
            out.append("<hr>")
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        # 普通段落
        out.append(f"<p>{inline(stripped)}</p>")
        i += 1

    return "\n".join(out)


def main() -> None:
    md = SRC.read_text(encoding="utf-8")
    body = convert(md)
    html = (
        '<!DOCTYPE html>\n<html lang="zh"><head><meta charset="utf-8">\n'
        "<title>中证1000 · Kronos 大模型择时应用（复现实验报告）</title>\n"
        f"<style>{CSS}</style></head><body>\n{body}\n"
        '<p class="disc">免责声明：本报告基于历史数据回测，仅供研究参考，不构成任何投资建议。'
        "模型：NeoQuasar/Kronos-base（官方 2025-06-30 快照，哈希已核）；数据：国泰安 CSMAR 及公开行情。</p>\n"
        "</body></html>"
    )
    OUT.write_text(html, encoding="utf-8")
    print(f"written {OUT} {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
