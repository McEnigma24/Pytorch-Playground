"""Download PDF from http(s) URL with size limits and basic SSRF checks."""

from __future__ import annotations

import ipaddress
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, OpenerDirector, Request, build_opener

import pymupdf

MAX_BYTES = 50 * 1024 * 1024
TIMEOUT_SEC = 60
MAX_REDIRECTS = 5
_USER_AGENT = "TTS-PDF-Reader/1.0"

REDIRECT_CODES = {301, 302, 303, 307, 308}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _urlopen_no_redirect(req: Request, timeout: int):
    opener: OpenerDirector = build_opener(_NoRedirect)
    return opener.open(req, timeout=timeout)


class PdfFetchError(ValueError):
    """Invalid URL, blocked host, or response is not a PDF."""


def _host_blocked(host: str) -> bool:
    h = host.strip().lower().rstrip(".")
    if h in ("localhost", "localhost.localdomain"):
        return True
    if h.endswith(".local") or h.endswith(".internal"):
        return True
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise PdfFetchError(f"Cannot resolve host: {host}") from exc
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
        ):
            return True
    return False


def _validate_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise PdfFetchError("URL is empty")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise PdfFetchError("URL must start with http:// or https://")
    if not parsed.netloc:
        raise PdfFetchError("URL has no host")
    hostname = parsed.hostname
    if not hostname:
        raise PdfFetchError("URL has no host")
    if _host_blocked(hostname):
        raise PdfFetchError("URL host is not allowed")
    return url


def _read_limited_response(resp, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := resp.read(65536):
        total += len(chunk)
        if total > max_bytes:
            raise PdfFetchError(f"PDF exceeds {max_bytes // (1024 * 1024)} MiB limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _assert_pdf_bytes(data: bytes) -> None:
    if len(data) < 5 or not data.startswith(b"%PDF"):
        raise PdfFetchError("Response is not a PDF file")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        doc.close()
    except (RuntimeError, ValueError) as exc:
        raise PdfFetchError(f"Invalid PDF: {exc}") from exc


def fetch_pdf_bytes(url: str) -> bytes:
    """Fetch PDF bytes from url. Raises PdfFetchError on failure."""
    current = _validate_url(url)

    for _ in range(MAX_REDIRECTS + 1):
        req = Request(current, headers={"User-Agent": _USER_AGENT})
        try:
            resp = _urlopen_no_redirect(req, TIMEOUT_SEC)
        except HTTPError as exc:
            if exc.code in REDIRECT_CODES:
                location = exc.headers.get("Location")
                if not location:
                    raise PdfFetchError(f"HTTP {exc.code} without Location") from exc
                current = _validate_url(urljoin(current, location.strip()))
                continue
            raise PdfFetchError(f"HTTP {exc.code}: {exc.reason}") from exc
        except URLError as exc:
            raise PdfFetchError(f"Download failed: {exc.reason}") from exc
        else:
            try:
                code = resp.getcode()
                if code in REDIRECT_CODES:
                    location = resp.headers.get("Location")
                    if not location:
                        raise PdfFetchError(f"HTTP {code} without Location")
                    current = _validate_url(urljoin(current, location.strip()))
                    continue
                data = _read_limited_response(resp, MAX_BYTES)
            finally:
                resp.close()

        _assert_pdf_bytes(data)
        return data

    raise PdfFetchError("Too many redirects")
