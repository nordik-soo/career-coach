"""Generate simple text-layer PDFs for the synthetic resume test pack.

No external PDF dependency is used. The generated PDFs are deliberately plain:
Helvetica text, multiple pages if needed, selectable text for pdfplumber.
"""
from __future__ import annotations

from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parent
PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT = 54
TOP = 742
LINE_HEIGHT = 13
FONT_SIZE = 10
CHARS_PER_LINE = 92
LINES_PER_PAGE = 51


def _escape_pdf_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def _wrapped_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            out.append("")
            continue
        if line.isupper() and len(line) < 80:
            out.append(line)
            continue
        out.extend(textwrap.wrap(
            line,
            width=CHARS_PER_LINE,
            break_long_words=False,
            replace_whitespace=False,
        ) or [""])
    return out


def _page_stream(lines: list[str]) -> bytes:
    commands = [
        "BT",
        f"/F1 {FONT_SIZE} Tf",
        f"{LEFT} {TOP} Td",
    ]
    for i, line in enumerate(lines):
        if i:
            commands.append(f"0 -{LINE_HEIGHT} Td")
        commands.append(f"({_escape_pdf_text(line)}) Tj")
    commands.append("ET")
    return ("\n".join(commands) + "\n").encode("latin-1", errors="replace")


def write_pdf(text_path: Path) -> Path:
    pdf_path = text_path.with_suffix(".pdf")
    lines = _wrapped_lines(text_path.read_text(encoding="utf-8"))
    pages = [
        lines[i:i + LINES_PER_PAGE]
        for i in range(0, len(lines), LINES_PER_PAGE)
    ] or [[]]

    objects: list[bytes] = []

    def add(obj: str | bytes) -> int:
        data = obj.encode("latin-1", errors="replace") if isinstance(obj, str) else obj
        objects.append(data)
        return len(objects)

    catalog_id = add("<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add(b"")
    font_id = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []
    for page_lines in pages:
        stream = _page_stream(page_lines)
        stream_id = add(
            b"<< /Length " + str(len(stream)).encode("ascii") +
            b" >>\nstream\n" + stream + b"endstream"
        )
        page_id = add(
            f"<< /Type /Page /Parent {pages_id} 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {stream_id} 0 R >>"
        )
        page_ids.append(page_id)

    objects[pages_id - 1] = (
        f"<< /Type /Pages /Count {len(page_ids)} /Kids "
        f"[{' '.join(f'{pid} 0 R' for pid in page_ids)}] >>"
    ).encode("latin-1")

    output = bytearray()
    output.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{idx} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_at = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n".encode("ascii")
    )
    pdf_path.write_bytes(output)
    return pdf_path


def main() -> None:
    for text_path in sorted(ROOT.glob("cv_*.txt")):
        pdf_path = write_pdf(text_path)
        print(f"wrote {pdf_path.name}")


if __name__ == "__main__":
    main()
