"""Inline engine.js and data.json into page.html -> hippo.html (one file).

    python -m viz.build
"""
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "hippo.html"


def render() -> str:
    page = (HERE / "page.html").read_text()
    engine = (HERE / "engine.js").read_text()
    data = (HERE / "data.json").read_text().replace("</", "<\\/")
    for marker in ("/*ENGINE*/", "/*DATA*/"):
        if page.count(marker) != 1:
            raise SystemExit(f"page.html must contain {marker} exactly once")
    return page.replace("/*ENGINE*/", engine).replace("/*DATA*/", data)


def main():
    out = render()
    OUT.write_text(out)
    print(f"wrote {OUT} ({len(out) // 1024} KB)")


if __name__ == "__main__":
    main()
