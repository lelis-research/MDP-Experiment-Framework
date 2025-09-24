"""
Helper functions for rendering Ansi (minihack)
"""
import os

def get_mono_font(size=14):
    """Try to load a monospaced font; fall back to PIL default."""
    from PIL import ImageFont
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/Library/Fonts/Menlo.ttc",
        "DejaVuSansMono.ttf",
        "Menlo.ttc",
    ):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

def normalize_ansi_frames(text_frames):
    """
    Convert list of ANSI strings to (frames_lines, max_cols, max_rows) so
    every frame can be drawn on an identical canvas.
    """
    frames_lines = []
    max_cols = 0
    max_rows = 0
    for t in text_frames:
        if isinstance(t, bytes):
            t = t.decode("utf-8", errors="ignore")
        lines = t.splitlines()
        while lines and lines[-1].strip() == "":
            lines.pop()
        if not lines:
            lines = [" "]
        # keep monospace alignment stable
        lines = [ln.replace("\t", "    ") for ln in lines]
        max_cols = max(max_cols, max(len(ln) for ln in lines))
        max_rows = max(max_rows, len(lines))
        frames_lines.append(lines)
    return frames_lines, max_cols, max_rows

def render_fixed_ansi(lines, max_cols, max_rows, font,
                       scale=2, margin=8, line_spacing=2,
                       fg=(230, 230, 230), bg=(0, 0, 0)):
    """
    Draw one ANSI frame's lines onto a fixed-size canvas, return numpy array.
    """
    from PIL import Image, ImageDraw
    import numpy as np

    # char metrics
    bbox = font.getbbox("M")
    ch_w = bbox[2] - bbox[0]
    ch_h = bbox[3] - bbox[1]

    W = margin * 2 + ch_w * max_cols
    H = margin * 2 + (ch_h + line_spacing) * max_rows - line_spacing

    # pad lines to grid
    padded = [ln.ljust(max_cols) for ln in lines]
    padded += [" " * max_cols] * (max_rows - len(padded))

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    y = margin
    for ln in padded:
        draw.text((margin, y), ln, font=font, fill=fg)
        y += ch_h + line_spacing

    if scale and scale != 1:
        img = img.resize((W * scale, H * scale), resample=Image.NEAREST)
    return np.array(img, dtype=np.uint8)