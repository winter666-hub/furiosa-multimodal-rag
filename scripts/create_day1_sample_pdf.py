"""Create a small deterministic two-page PDF for the Day 1 RAG smoke test."""

from __future__ import annotations

from pathlib import Path


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 12 Tf", "72 740 Td", "14 TL"]
    for index, line in enumerate(lines):
        if index:
            commands.append("T*")
        commands.append(f"({_escape(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("ascii")


def create_pdf(path: Path) -> None:
    page_one = _content(
        [
            "Day 1 Text RAG Test Document - Page 1",
            "Scaled dot-product attention computes scores from a query and keys.",
            "The raw dot products can grow large when the key dimension d_k increases.",
            "Large values push softmax into regions with very small gradients.",
        ]
    )
    page_two = _content(
        [
            "Day 1 Text RAG Test Document - Page 2",
            "The attention score is divided by the square root of d_k before softmax.",
            "This scaling keeps the variance of dot products at a manageable magnitude.",
            "It prevents softmax saturation and helps training remain stable.",
            "The relevant formula is Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V.",
        ]
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(page_one)} >>\nstream\n".encode() + page_one + b"\nendstream",
        f"<< /Length {len(page_two)} >>\nstream\n".encode() + page_two + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


if __name__ == "__main__":
    create_pdf(Path("data/day1_text_rag_sample.pdf"))

