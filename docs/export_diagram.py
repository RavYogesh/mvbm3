"""Rasterise the architecture diagram to a high-resolution JPEG.

The diagram is inline SVG styled by the page's stylesheet, including CSS custom
properties. Standalone SVG rasterisers do not resolve either, so they render a
black-on-transparent skeleton. A real browser engine does, which is why this
drives headless Chrome rather than a Python SVG library.

Single source of truth: the style block and the <svg> element are extracted from
`architecture-diagram.html` at export time, so the JPEG cannot drift from the
published page.

    python docs/export_diagram.py [--scale 3] [--quality 95]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "architecture-diagram.html"
OUT_JPG = HERE / "architecture-diagram.jpg"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

# Rendered on a light ground regardless of the exporter's OS theme. A JPEG has
# no theme to respond to, and dark-on-dark art pasted into a light document is
# the usual result of letting the host decide.
RENDER_PAGE = """<!doctype html>
<html data-theme="light"><head><meta charset="utf-8">
{style}
<style>
  html, body {{ background:#FFFFFF; }}
  body {{ margin:0; padding:44px 48px 34px; width:{width}px; }}
  .plate {{ width:{width}px; }}
  .hd {{ font-family:var(--mono); font-size:12px; letter-spacing:.14em;
         text-transform:uppercase; color:var(--accent); margin:0 0 8px; }}
  .ti {{ font-family:var(--sans); font-size:29px; font-weight:650; letter-spacing:-.02em;
         color:var(--ink); margin:0 0 22px; }}
  .canvas {{ border:none; padding:0; background:none; overflow:visible; }}
  svg {{ min-width:0; width:{width}px; }}
  .ft {{ font-family:var(--mono); font-size:11.5px; color:var(--muted);
         margin:20px 0 0; padding-top:12px; border-top:1px solid var(--line); }}
</style></head>
<body><div class="plate">
  <p class="hd">GenAI Platform Engineering &middot; model validation</p>
  <p class="ti">Compressed-model validation harness</p>
  <div class="canvas">{svg}</div>
  <p class="ft">github.com/RavYogesh/mvbm3 &middot; benchmark v2.0.0 &middot;
     verdicts: PASS / BLOCK / INCONCLUSIVE &middot; demo bundles are synthetic and labelled as such</p>
</div></body></html>
"""


def find_browser() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in ("chrome", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    sys.exit("no Chrome or Edge binary found; install one or export the SVG manually")


def extract(html: str) -> tuple[str, str]:
    style = re.search(r"<style>.*?</style>", html, re.DOTALL)
    svg = re.search(r"<svg\b.*?</svg>", html, re.DOTALL)
    if not style or not svg:
        sys.exit("could not locate the <style> or <svg> block in the source page")
    return style.group(0), svg.group(0)


def trim(image: Image.Image, background: tuple[int, int, int], margin: int) -> Image.Image:
    """Crop the uniform border, then re-add an even margin.

    The window is deliberately oversized so no content is ever clipped; the real
    extent is discovered here rather than hard-coded, which keeps the export
    correct when the diagram grows.
    """
    canvas = Image.new("RGB", image.size, background)
    box = ImageChops.difference(image, canvas).getbbox()
    if not box:
        return image
    left, top, right, bottom = box
    return image.crop((
        max(left - margin, 0),
        max(top - margin, 0),
        min(right + margin, image.width),
        min(bottom + margin, image.height),
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=3.0, help="device pixel ratio")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--width", type=int, default=1240, help="CSS width of the plate")
    parser.add_argument("--out", default=str(OUT_JPG))
    args = parser.parse_args()

    style, svg = extract(SOURCE.read_text(encoding="utf-8"))
    browser = find_browser()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        page = tmp_path / "render.html"
        page.write_text(
            RENDER_PAGE.format(style=style, svg=svg, width=args.width), encoding="utf-8"
        )
        png = tmp_path / "shot.png"

        # Generous window: the trim step finds the true extent afterwards, so an
        # oversized canvas costs a little memory and removes any risk of clipping.
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--default-background-color=FFFFFFFF",
                f"--force-device-scale-factor={args.scale}",
                f"--window-size={args.width + 200},1500",
                "--virtual-time-budget=3000",
                f"--screenshot={png}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        if not png.exists():
            sys.exit("headless render produced no image")

        image = Image.open(png).convert("RGB")
        image = trim(image, (255, 255, 255), margin=int(24 * args.scale))

        out = Path(args.out)
        image.save(
            out,
            "JPEG",
            quality=args.quality,
            subsampling=0,      # 4:4:4 -- chroma subsampling smears fine rules and small type
            optimize=True,
            progressive=True,
            dpi=(300, 300),
        )

    inches = image.width / 300
    print(f"wrote {out}")
    print(f"  {image.width} x {image.height} px  ({inches:.1f} in wide at 300 dpi)")
    print(f"  quality {args.quality}, 4:4:4 chroma, {out.stat().st_size / 1_000_000:.2f} MB")


if __name__ == "__main__":
    main()
