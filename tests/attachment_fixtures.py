"""Archivos sintéticos para las pruebas de adjuntos (sin datos reales)."""

from __future__ import annotations

import struct
import zlib
from io import BytesIO

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

# Cabeceras mínimas válidas: la visión se sustituye por un modelo falso.
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"\x00" * 64
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 32


def png(width: int = 2, height: int = 2) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + b"\xff" * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _font_resources() -> DictionaryObject:
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    return DictionaryObject({NameObject("/F1"): font})


def text_pdf(lines: list[str], *, pages: int = 1, encrypt_password: str | None = None) -> bytes:
    """PDF digital con capa de texto."""
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): _font_resources()})
        ops = ["BT /F1 12 Tf 72 720 Td 14 TL"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({escaped}) Tj T*")
        ops.append("ET")
        stream = DecodedStreamObject()
        stream.set_data(" ".join(ops).encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypt_password is not None:
        writer.encrypt(user_password=encrypt_password, owner_password="owner", algorithm="AES-256")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def scanned_pdf(*, image: str = "jpeg") -> bytes:
    """PDF sin capa de texto con una imagen por página (escaneo simulado)."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    xobject = DecodedStreamObject()
    if image == "jpeg":
        xobject.set_data(JPEG)
        xobject.update(
            {
                NameObject("/Filter"): NameObject("/DCTDecode"),
                NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
            }
        )
    elif image == "gray":
        xobject.set_data(b"\x00\xff\xff\x00")
        xobject.update({NameObject("/ColorSpace"): NameObject("/DeviceGray")})
    else:  # formato que la visión no admite
        xobject.set_data(b"\x00" * 16)
        xobject.update(
            {
                NameObject("/Filter"): NameObject("/JPXDecode"),
                NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
            }
        )
    xobject.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(2),
            NameObject("/Height"): NumberObject(2),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/XObject"): DictionaryObject(
                {NameObject("/Im0"): writer._add_object(xobject)}
            )
        }
    )
    contents = DecodedStreamObject()
    contents.set_data(b"q 200 0 0 200 72 500 cm /Im0 Do Q")
    page[NameObject("/Contents")] = writer._add_object(contents)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()
