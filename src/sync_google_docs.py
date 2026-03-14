#!/usr/bin/env python3
"""
Sync Google Docs to local markdown files.

Usage:
    python src/sync_google_docs.py              # sync all configured docs
    python src/sync_google_docs.py --list       # show configured docs
    python src/sync_google_docs.py --add URL NAME  # add a new doc

Requirements: pandoc (brew install pandoc)
"""

import subprocess
import re
import sys
import json
import urllib.parse
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DOCS_DIR = REPO_ROOT / "docs" / "google-docs"
CONFIG_FILE = REPO_ROOT / "src" / "docs_config.json"

DEFAULT_DOCS = [
    {
        "url": "https://docs.google.com/document/d/16eeltCaTpiiM1t_m_5BSxRnqxoMoiJ-xn4cy0x-TFgc/edit",
        "name": "challenge-1-sparse-parity",
        "description": "Sutro Group Challenge #1: Sparse Parity"
    },
    {
        "url": "https://docs.google.com/document/d/1B9867EN6Bg4ZVQK9vI_ZqykZ5HEtMAHJ7zBGGas4szQ/edit",
        "name": "sutro-group-main",
        "description": "Sutro Group main page (meetings index)"
    },
    {
        "url": "https://docs.google.com/document/d/1344Vld2n9-8B-OfeeqI5sP9fqPYCLQTglc9pSAmeeEM/edit",
        "name": "yaroslav-technical-sprint-1",
        "description": "Yaroslav Technical Sprint 1 (02 Mar 2026)"
    },
]


def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return DEFAULT_DOCS


def save_config(docs):
    CONFIG_FILE.write_text(json.dumps(docs, indent=2) + "\n")


def extract_doc_id(url):
    """Extract document ID from a Google Docs URL."""
    m = re.search(r'/document/d/([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract doc ID from: {url}")


def download_html(doc_id):
    """Download a Google Doc as HTML via the public export URL."""
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=html"
    print(f"  Downloading {export_url[:80]}...")
    req = urllib.request.Request(export_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def html_to_markdown(html_content):
    """Convert HTML to markdown using pandoc."""
    result = subprocess.run(
        ["pandoc", "-f", "html", "-t", "markdown", "--wrap=none"],
        input=html_content,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pandoc failed: {result.stderr}")
    return result.stdout


def clean_markdown(md):
    """Remove pandoc artifacts, decode Google redirect URLs, strip base64 images."""
    # Remove pandoc span/class attributes like {.c0}, {#h.xxx .c7}
    md = re.sub(r'\{[^}]*\}', '', md)

    # Decode Google redirect URLs
    def decode_google_url(m):
        full_url = m.group(0)
        parsed = re.search(r'q=([^&)]+)', full_url)
        if parsed:
            return f"({urllib.parse.unquote(parsed.group(1))})"
        return full_url

    md = re.sub(
        r'\(https://www\.google\.com/url\?[^)]+\)',
        decode_google_url,
        md,
    )

    # Remove base64 embedded images
    md = re.sub(r'!\[[^\]]*\]\(data:image[^)]*\)', '[embedded image]', md)

    # Remove empty bracket lines
    md = re.sub(r'^\s*\[\s*\]\s*$', '', md, flags=re.MULTILINE)

    # Unwrap single brackets [text] that aren't links
    md = re.sub(r'\[([^\]]+)\](?!\()', r'\1', md)

    # Fix escaped quotes
    md = md.replace("\\'", "'")

    # Fix anchor-only links (won't work outside Google Docs)
    md = re.sub(r'\[([^\]]+)\]\(#h\.[^)]+\)', r'\1', md)

    # Remove lines that are just "[embedded image]" noise (optional: keep as markers)
    # md = re.sub(r'^\[embedded image\]\s*$', '', md, flags=re.MULTILINE)

    # Collapse excessive blank lines
    md = re.sub(r'\n{3,}', '\n\n', md)

    # Strip very long lines (table artifacts with encoded data)
    lines = md.split('\n')
    lines = [('[large table/output removed]' if len(l) > 2000 else l) for l in lines]
    md = '\n'.join(lines)

    md = md.strip() + '\n'
    return md


def extract_links(html_content):
    """Extract and decode all hyperlinks from HTML."""
    links = set()
    for href in re.findall(r'href="([^"]*)"', html_content):
        if href.startswith('#'):
            continue
        href = href.replace('&amp;', '&')
        if 'google.com/url?q=' in href:
            m = re.search(r'q=([^&]+)', href)
            if m:
                links.add(urllib.parse.unquote(m.group(1)))
        else:
            links.add(urllib.parse.unquote(href))
    return sorted(links)


def extract_header(file_path):
    """Extract any cross-reference header from an existing file.

    Preserves everything before the first markdown heading that starts
    with '# ' (the Google Doc's own title). This keeps manually-added
    !!! info blocks and other front-matter across syncs.
    """
    if not file_path.exists():
        return ""
    content = file_path.read_text()
    lines = content.split('\n')
    header_lines = []
    for line in lines:
        if line.startswith('# '):
            break
        header_lines.append(line)
    header = '\n'.join(header_lines).strip()
    return header + '\n\n' if header else ""


def sync_doc(doc_config):
    """Download, convert, and save a single Google Doc."""
    doc_id = extract_doc_id(doc_config["url"])
    name = doc_config["name"]
    out_path = DOCS_DIR / f"{name}.md"

    print(f"\nSyncing: {doc_config.get('description', name)}")

    # Preserve any cross-reference header before overwriting
    preserved_header = extract_header(out_path)
    if preserved_header.strip():
        print(f"  Preserving cross-reference header")

    html = download_html(doc_id)
    print(f"  Downloaded {len(html):,} bytes of HTML")

    links = extract_links(html)
    print(f"  Found {len(links)} hyperlinks")

    md = html_to_markdown(html)
    print(f"  Converted to {len(md):,} bytes of markdown")

    md = clean_markdown(md)
    print(f"  Cleaned to {len(md):,} bytes")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(preserved_header + md)
    print(f"  Saved to {out_path.relative_to(REPO_ROOT)}")

    return links


def check_pandoc():
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return

    docs = load_config()

    if "--list" in sys.argv:
        print("Configured docs:")
        for d in docs:
            print(f"  {d['name']}: {d['url']}")
            if d.get('description'):
                print(f"    {d['description']}")
        return

    if "--add" in sys.argv:
        idx = sys.argv.index("--add")
        if idx + 2 >= len(sys.argv):
            print("Usage: --add URL NAME [DESCRIPTION]")
            sys.exit(1)
        url = sys.argv[idx + 1]
        name = sys.argv[idx + 2]
        desc = sys.argv[idx + 3] if idx + 3 < len(sys.argv) else ""
        extract_doc_id(url)  # validate
        docs.append({"url": url, "name": name, "description": desc})
        save_config(docs)
        print(f"Added: {name} -> {url}")
        return

    if not check_pandoc():
        print("Error: pandoc is required. Install with: brew install pandoc")
        sys.exit(1)

    print(f"Syncing {len(docs)} Google Docs...")

    all_links = []
    for doc in docs:
        try:
            links = sync_doc(doc)
            all_links.extend(links)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Deduplicate and save links summary
    unique_links = sorted(set(all_links))
    if unique_links:
        links_file = REPO_ROOT / "docs" / "references_auto.md"
        with open(links_file, "w") as f:
            f.write("# Auto-extracted References\n\n")
            f.write(f"Generated by `sync_google_docs.py` - {len(unique_links)} unique links\n\n")
            for link in unique_links:
                f.write(f"- <{link}>\n")
        print(f"\nSaved {len(unique_links)} unique links to docs/references_auto.md")

    print("\nDone!")


if __name__ == "__main__":
    main()
