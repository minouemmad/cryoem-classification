#!/usr/bin/env python3
"""(11-25) Deck of every ab initio + NU class map of the label-free classes for the five starting axes (no-cryoDRGN, codebook, MLP curve, diffusion map, latent).
Input: DATA/compare_<DS>.json + DATA/renders/solid_<DS>_*.png (hudson model_pipeline/od/cmp).  Output: OUT.pptx and, with --preview, OUT_preview/slide_NNN.png
(approximate renders of the same layout, for checking).  Usage: build_deck.py DATA OUT.pptx [--preview]"""
import json, os, sys, time
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

DATA, OUT = sys.argv[1], sys.argv[2]; PREV = "--preview" in sys.argv
DSS = {"J4001": (range(2, 6), "J4001 · gP25 WT"), "J207": (range(2, 6), "J207 · hP12 IDOR4"), "J2708": (range(2, 6), "J2708 · eP30 F508del E1371W"),
       "J399": (range(2, 11), "J399 · hP7 E1371Q ATP")}
STARTS = [("start_free", "No-cryoDRGN start", "PC1 of the unconditioned residual-model coordinates (no network)"),
          ("start_vq", "Codebook start", "300 k-means cells in the residual-model coordinates, back-projected cell maps, PCA of the cell maps"),
          ("start_mlp", "MLP-curve start", "bottleneck of a small MLP autoencoder on the residual-model coordinates (nonlinear principal curve)"),
          ("start_diffmap", "Diffusion-map start", "first diffusion-map coordinate of the residual-model coordinates"),
          ("start_latent", "Latent start", "PC1 of the cryoDRGN RAE encoder latents (no decoder)")]
BASE = {"J4001": "rae_base_control", "J207": "rae_base", "J2708": "rae_base", "J399": "rae_base"}
W, H = 13.333, 7.5; DARK = (0x1F, 0x2A, 0x37); GREY = (0x6B, 0x72, 0x80); TEAL = (0x0F, 0x76, 0x6E); RED = (0xB4, 0x3C, 0x3C); LIGHT = (0xF3, 0xF4, 0xF6)
CROP = os.path.join(os.path.dirname(os.path.abspath(OUT)), ".deck_crops"); os.makedirs(CROP, exist_ok=True)
import re; slug = lambda t: re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")   # same as od_compare.py

def crop(png):
    out = os.path.join(CROP, os.path.basename(png))
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(png): return out
    im = Image.open(png).convert("RGB"); g = im.convert("L").point(lambda v: 255 if v < 245 else 0); bb = g.getbbox() or (0, 0, im.width, im.height)
    m = 12; bb = (max(0, bb[0] - m), max(0, bb[1] - m), min(im.width, bb[2] + m), min(im.height, bb[3] + m)); im.crop(bb).save(out); return out

# ---- layout model: list of slides, each a list of items (kind, x, y, w, h, payload); rendered to pptx and to a preview
slides = []
def text(s, x, y, w, h, t, size=14, color=DARK, bold=False, align="l"): s.append(("text", x, y, w, h, dict(t=t, size=size, color=color, bold=bold, align=align)))
def image(s, x, y, w, h, png):
    iw, ih = Image.open(png).size; sc = min(w / iw, h / ih); ww, hh = iw * sc, ih * sc; s.append(("image", x + (w - ww) / 2, y + (h - hh) / 2, ww, hh, dict(png=png)))
def rect(s, x, y, w, h, fill): s.append(("rect", x, y, w, h, dict(fill=fill)))
def title(s, t, sub=None):
    text(s, 0.5, 0.28, W - 1.0, 0.6, t, 26, DARK, True)
    if sub: text(s, 0.5, 0.88, W - 1.0, 0.4, sub, 13, GREY)

data = {ds: json.load(open(f"{DATA}/compare_{ds}.json")) for ds in DSS if os.path.exists(f"{DATA}/compare_{ds}.json")}
def render(ds, name): f = f"{DATA}/renders/solid_{ds}_{slug(name)}.png"; return crop(f) if os.path.exists(f) else None

# title + methods
s = []; title(s, "Label-Free Classes from Five Starting Axes", f"Every class map after ab initio + NU · {', '.join(DSS)} · {time.strftime('%Y-%m-%d')}")
y = 1.7
for run, lab, desc in STARTS:
    text(s, 0.7, y, 3.2, 0.4, lab, 17, TEAL if run != "start_latent" else DARK, True); text(s, 3.9, y + 0.03, 8.9, 0.6, desc, 14, DARK); y += 0.72
text(s, 0.7, y + 0.25, 12.0, 1.4,
     "Each start feeds the same residual model, sharpening, fully conditioned search and map-space classes. K = 2–5 for J4001, J207 and J2708; K = 2–10 for J399 "
     "(every K, no K-scan cut). Each class: cryoSPARC import → ab initio (1 class) → NU refine at 128 px with the same parameters on hudson, ganges or east. "
     "Captions: particles; best-matched heterorefinement class (Hungarian match) and cc = corr(map − consensus, class − consensus) at 10 Å, green when cc ≥ 0.5; nearest class by distance d² = |map − class|² / |most distant class pair|². P7 lies closest to the consensus, so its cc is ill-conditioned and only distance can show a P7-like map.", 13, GREY)
slides.append(s)

for ds, (Ks, dlab) in DSS.items():
    if ds not in data: continue
    d = data[ds]; rows = d["rows"]; sets = d["sets"]
    het = [r for r in rows if r["run"] == "hetero"]
    # dataset summary: reference maps + table of mean matched cc
    s = []; title(s, f"{dlab}", "Heterorefinement classes (reference) and mean matched cc of each refined set")
    n = len(het); cw = min(2.2, (W - 1.0) / max(n, 1))
    for i, r in enumerate(het):
        x = 0.5 + i * cw; p = render(ds, r["name"])
        if p: image(s, x + 0.05, 1.35, cw - 0.1, 1.75, p)
        text(s, x, 3.1, cw, 0.3, r["name"].replace("class ", ""), 12, DARK, True, "c")
    cols = list(Ks); x0, y0 = 0.5, 3.65; c0 = 2.9; cwt = min(1.0, (W - 1.0 - c0) / len(cols)); rh = 0.42
    rect(s, x0, y0, c0 + cwt * len(cols), rh, LIGHT); text(s, x0 + 0.08, y0 + 0.07, c0, rh, "start  \\  K", 12, GREY, True)
    for j, K in enumerate(cols): text(s, x0 + c0 + j * cwt, y0 + 0.07, cwt, rh, str(K), 12, GREY, True, "c")
    tab = ([(BASE[ds], "Base run (decoded volume PC1)")] if any(f"{BASE[ds]} K{K}" in sets for K in cols) else []) + [(r, l) for r, l, _ in STARTS]   # base row only when its sets are in this compare
    for i, (run, lab) in enumerate(tab):
        yy = y0 + (i + 1) * rh; text(s, x0 + 0.08, yy + 0.07, c0, rh, lab, 12, DARK, run != BASE[ds])
        for j, K in enumerate(cols):
            v = sets.get(f"{run} K{K}"); t = f"{v['mean_match_cc']:.2f}" if v else "–"
            text(s, x0 + c0 + j * cwt, yy + 0.07, cwt, rh, t, 12, (TEAL if v and v["mean_match_cc"] >= 0.5 else DARK) if v else GREY, False, "c")
    slides.append(s)
    # one slide per start x K: all class maps
    for run, lab, _ in STARTS:
        for K in Ks:
            v = sets.get(f"{run} K{K}"); maps = sorted([r for r in rows if r["run"] == run and r["K"] == K], key=lambda r: r["k"])
            s = []; sub = (f"mean matched cc {v['mean_match_cc']:.2f} · heterorefinement classes with a counterpart (cc ≥ 0.5): {', '.join(v['covered']) or 'none'}"
                           if v else "not refined yet")
            near = sorted({m['closest'] for m in maps}); sub += (f" · nearest by distance: {', '.join(near)}" if maps else "")
            title(s, f"{ds} · {lab} · K = {K}", sub)
            # reference strip
            rw = min(1.15, 6.0 / max(len(het), 1))
            text(s, 0.5, 1.38, 2.0, 0.3, "heterorefinement", 11, GREY)
            for i, r in enumerate(het):
                p = render(ds, r["name"]); x = 2.3 + i * rw
                if p: image(s, x, 1.25, rw - 0.08, 0.95, p)
                text(s, x, 2.18, rw - 0.08, 0.25, r["name"].replace("class ", ""), 10, GREY, False, "c")
            # class maps: one row up to 5, two rows above
            per = K if K <= 5 else (K + 1) // 2; nrow = 1 if K <= 5 else 2; top = 2.65; areah = H - top - 0.25; cellw = min(2.45, (W - 1.0) / per); cellh = areah / nrow
            x00 = (W - per * cellw) / 2; match = dict(zip(v["match"], v["match_cc"])) if v else {}; mt = v["match"] if v else {}
            for k in range(K):
                r = next((m for m in maps if m["name"].endswith(f": g{k}")), None); row, col = divmod(k, per); x = x00 + col * cellw; yy = top + row * cellh
                p = render(ds, f"{run} K{K}: g{k}") if r else None; imh = cellh - 0.86
                if p: image(s, x + 0.05, yy, cellw - 0.1, imh, p)
                else: rect(s, x + 0.15, yy + 0.1, cellw - 0.3, imh - 0.2, LIGHT); text(s, x, yy + imh / 2 - 0.15, cellw, 0.3, "not refined", 11, GREY, False, "c")
                nm = f"{run} K{K}: g{k}"; n = f"{r['n']:,}" if r and r.get("n") else ""
                text(s, x, yy + imh + 0.02, cellw, 0.28, f"g{k}" + (f" · {n} ptcl" if n else ""), 11, DARK, True, "c")
                if nm in match:
                    c = match[nm]; text(s, x, yy + imh + 0.28, cellw, 0.28, f"→ {mt[nm]}  cc {c:.2f}", 11, TEAL if c >= 0.5 else GREY, False, "c")
                elif r:
                    best = max(r["cc_diff"].items(), key=lambda kv: kv[1]); text(s, x, yy + imh + 0.28, cellw, 0.28, f"unmatched · closest {best[0]} {best[1]:.2f}", 10, GREY, False, "c")
                if r:
                    cl = r["closest"]; text(s, x, yy + imh + 0.54, cellw, 0.28, f"nearest {cl}  d² {r['d2'][cl]:.2f}", 10, TEAL if cl == "P7" else GREY, cl == "P7", "c")
            slides.append(s)

# ---- pptx
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H); blank = prs.slide_layouts[6]
for s in slides:
    sl = prs.slides.add_slide(blank)
    for kind, x, y, w, h, p in s:
        if kind == "image": sl.shapes.add_picture(p["png"], Inches(x), Inches(y), Inches(w), Inches(h))
        elif kind == "rect":
            sh = sl.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h)); sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor(*p["fill"]); sh.line.fill.background()
        else:
            tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
            for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"): setattr(tf, m, 0)
            para = tf.paragraphs[0]; para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER}[p["align"]]
            run = para.add_run(); run.text = p["t"]; run.font.size = Pt(p["size"]); run.font.bold = p["bold"]; run.font.color.rgb = RGBColor(*p["color"]); run.font.name = "Calibri"
prs.save(OUT); print("slides", len(slides), "->", OUT)

# ---- approximate preview (same geometry)
if PREV:
    PD = OUT[:-5] + "_preview"; os.makedirs(PD, exist_ok=True); dpi = 96
    def font(sz, b):
        for f in (["DejaVuSans-Bold.ttf"] if b else ["DejaVuSans.ttf"]):
            try: return ImageFont.truetype(f, int(sz * dpi / 72))
            except Exception: pass
        return ImageFont.load_default()
    for i, s in enumerate(slides):
        im = Image.new("RGB", (int(W * dpi), int(H * dpi)), "white"); dr = ImageDraw.Draw(im)
        for kind, x, y, w, h, p in s:
            X, Y, Wd, Ht = [int(v * dpi) for v in (x, y, w, h)]
            if kind == "image": im.paste(Image.open(p["png"]).convert("RGB").resize((max(Wd, 1), max(Ht, 1))), (X, Y))
            elif kind == "rect": dr.rectangle([X, Y, X + Wd, Y + Ht], fill=p["fill"])
            else:
                f = font(p["size"], p["bold"]); words = p["t"].split(" "); lines, cur = [], ""
                for wd in words:
                    t = (cur + " " + wd).strip()
                    if dr.textlength(t, font=f) <= Wd or not cur: cur = t
                    else: lines.append(cur); cur = wd
                lines.append(cur); yy = Y
                for ln in lines:
                    tx = X if p["align"] == "l" else X + (Wd - dr.textlength(ln, font=f)) / 2; dr.text((tx, yy), ln, font=f, fill=p["color"]); yy += int(p["size"] * dpi / 72 * 1.2)
        im.save(f"{PD}/slide_{i + 1:03d}.png")
    print("preview ->", PD)
