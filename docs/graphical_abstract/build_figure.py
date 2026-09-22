#!/usr/bin/env python3
"""Rebuild the SPATHI graphical abstract from vector primitives and Graphviz.

Requires Python >=3.10, Pillow and the free Graphviz command `dot`.
Inkscape is required only for PDF/PNG exports; SVG generation is independent.
No generative-image service, API call, patient data or SPATHI inference is used.
Global panel positions and miniature graph layouts are calculated by Graphviz.
Within-panel illustrations use bounded, deterministic local drawing components.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Any

from PIL import ImageFont

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "figure_config.json").read_text(encoding="utf-8"))
C = CONFIG["palette"]
FONT = CONFIG["font_family"]
TEXT_BOXES: list[dict[str, Any]] = []
REGIONS: list[dict[str, Any]] = []


def command(args: list[str], *, text: str | None = None) -> str:
    result = subprocess.run(args, input=text, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{result.stderr}")
    return result.stdout


def find_font(bold: bool = False) -> str:
    filename = (
        ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf") if FONT == "DejaVu Sans" else FONT
    )
    try:
        return str(ImageFont.truetype(filename, 18).path)
    except OSError as exc:
        if shutil.which("fc-match"):
            pattern = FONT + (":style=Bold" if bold else "")
            candidate = command(["fc-match", "-f", "%{file}", pattern]).strip()
            if candidate and Path(candidate).exists():
                return candidate
        raise RuntimeError("Install DejaVu Sans, or configure an available font family.") from exc


FONTS = {False: find_font(False), True: find_font(True)}


def text_metrics(value: str, size: float, bold: bool = False) -> tuple[float, float, float, float]:
    # Supersampled font measurements avoid integer-size rounding errors.
    scale = 4
    font = ImageFont.truetype(FONTS[bold], round(size * scale))
    a, b, c, d = font.getbbox(value, anchor="ls")
    return a / scale, b / scale, c / scale, d / scale


def blend(hex_a: str, hex_b: str, fraction: float) -> str:
    fraction = min(1.0, max(0.0, fraction))
    aa = [int(hex_a[k : k + 2], 16) for k in (1, 3, 5)]
    bb = [int(hex_b[k : k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(
        f"{round(a + (b - a) * fraction):02X}" for a, b in zip(aa, bb, strict=True)
    )


def attrs(**kw: Any) -> str:
    return " ".join(
        f'{key.replace("_", "-")}="{escape(str(value), quote=True)}"'
        for key, value in kw.items()
        if value is not None
    )


class SVG:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.ox = 0.0
        self.oy = 0.0
        self.region = "canvas"

    def raw(self, value: str) -> None:
        self.parts.append(value)

    def group(self, name: str, x: float, y: float, width: float, height: float) -> None:
        self.ox, self.oy, self.region = x, y, name
        self.raw(f'<g id="{name}" transform="translate({x:.4f},{y:.4f})">')
        REGIONS.append(dict(name=name, x=x, y=y, width=width, height=height))

    def end(self) -> None:
        self.raw("</g>")
        self.ox = self.oy = 0.0
        self.region = "canvas"

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str | None = None,
        radius: float = 0,
        sw: float = 1,
        **kw: Any,
    ) -> None:
        self.raw(
            "<rect "
            + attrs(
                x=x,
                y=y,
                width=w,
                height=h,
                fill=fill,
                stroke=stroke,
                rx=radius,
                stroke_width=sw if stroke else None,
                **kw,
            )
            + "/>"
        )

    def circle(
        self,
        x: float,
        y: float,
        r: float,
        fill: str = "none",
        stroke: str | None = None,
        sw: float = 1,
        **kw: Any,
    ) -> None:
        self.raw(
            "<circle "
            + attrs(
                cx=x, cy=y, r=r, fill=fill, stroke=stroke, stroke_width=sw if stroke else None, **kw
            )
            + "/>"
        )

    def path(
        self, d: str, stroke: str = C["line"], sw: float = 1.5, fill: str = "none", **kw: Any
    ) -> None:
        self.raw(
            "<path "
            + attrs(
                d=d,
                stroke=stroke,
                stroke_width=sw,
                fill=fill,
                stroke_linecap="round",
                stroke_linejoin="round",
                **kw,
            )
            + "/>"
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = C["line"],
        sw: float = 1.5,
        **kw: Any,
    ) -> None:
        self.path(f"M{x1},{y1} L{x2},{y2}", stroke, sw, **kw)

    def arrow(
        self,
        points: list[tuple[float, float]],
        color: str = C["muted"],
        sw: float = 2.0,
        head: float = 7.0,
    ) -> None:
        d = "M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in points)
        self.path(d, color, sw)
        x, y = points[-1]
        xp, yp = points[-2]
        theta = math.atan2(y - yp, x - xp)
        ux, uy = math.cos(theta), math.sin(theta)
        vx, vy = -uy, ux
        p1 = (x - head * ux + head * 0.48 * vx, y - head * uy + head * 0.48 * vy)
        p2 = (x - head * ux - head * 0.48 * vx, y - head * uy - head * 0.48 * vy)
        self.path(f"M{x},{y} L{p1[0]},{p1[1]} L{p2[0]},{p2[1]} Z", color, 0.8, color)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: float = 18,
        fill: str = C["ink"],
        bold: bool = False,
        anchor: str = "start",
        max_width: float | None = None,
        track: bool = True,
        **kw: Any,
    ) -> None:
        a, b, c, d = text_metrics(value, size, bold)
        width = c - a
        if max_width is not None and width > max_width + 0.5:
            raise ValueError(
                f"Text too wide in {self.region}: {value!r} ({width:.1f} > {max_width})"
            )
        align = {"start": 0.0, "middle": width / 2, "end": width}[anchor]
        if track:
            TEXT_BOXES.append(
                dict(
                    region=self.region,
                    text=value,
                    x0=x + self.ox + a - align,
                    y0=y + self.oy + b,
                    x1=x + self.ox + c - align,
                    y1=y + self.oy + d,
                    font_size=size,
                )
            )
        self.raw(
            "<text "
            + attrs(
                x=x,
                y=y,
                fill=fill,
                font_family=FONT,
                font_size=size,
                font_weight=700 if bold else 400,
                text_anchor=anchor,
                **kw,
            )
            + ">"
            + escape(value)
            + "</text>"
        )

    def lines(
        self,
        x: float,
        y: float,
        lines: list[str],
        size: float = 18,
        leading: float | None = None,
        **kw: Any,
    ) -> None:
        leading = leading or size * 1.35
        for i, value in enumerate(lines):
            self.text(x, y + i * leading, value, size=size, **kw)


def top_layout() -> tuple[dict[str, dict[str, float]], float, float]:
    if not shutil.which("dot"):
        raise RuntimeError(
            "Graphviz is required. Install Graphviz and make `dot` available on PATH."
        )
    data = json.loads(command(["dot", "-Tjson", str(ROOT / "graphical_abstract.dot")]))
    nodes = {}
    bbox = [float(v) for v in data["bb"].split(",")]
    for obj in data["objects"]:
        x, y = [float(v) for v in obj["pos"].split(",")]
        w, h = 72 * float(obj["width"]), 72 * float(obj["height"])
        nodes[obj["name"]] = dict(x=x - w / 2, y=bbox[3] - y - h / 2, w=w, h=h)
    return nodes, bbox[2], bbox[3]


def header(s: SVG, w: float) -> None:
    s.text(30, 58, CONFIG["title"], 48, bold=True)
    s.line(272, 23, 272, 80, C["line"], 2)
    s.lines(295, 42, CONFIG["subtitle"].split("\n"), 25, leading=32, bold=True)
    s.text(30, 108, CONFIG["expansion"], 18, C["muted"], max_width=w - 60)


def panel(
    s: SVG,
    name: str,
    box: dict[str, float],
    index: int,
    title: str,
    subtitle: str,
    highlight: bool = False,
) -> None:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    s.group(name, x, y, w, h)
    s.rect(
        0, 0, w, h, C["white"], C["teal"] if highlight else C["line"], 14, 2 if highlight else 1.2
    )
    s.rect(18, 18, 28, 25, C["teal"] if highlight else C["ink"], radius=6)
    s.text(32, 36, str(index), 16, C["white"], bold=True, anchor="middle")
    s.text(
        58,
        37,
        ["INPUT", "CONTEXT", "LEARNING", "OUTPUT"][index - 1],
        15,
        C["teal"] if highlight else C["muted"],
        bold=True,
    )
    s.text(22, 75, title, 22, bold=True, max_width=w - 40)
    s.text(22, 101, subtitle, 16, C["muted"], max_width=w - 40)


def gene_matrix(
    s: SVG, x: float, y: float, w: float, h: float, rows: int = 8, cols: int = 15, seed: int = 11
) -> None:
    rng = random.Random(seed)
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            group = c // max(1, cols // 3)
            value = 0.15 + 0.75 * ((math.sin((r + 1) * 1.17 + group * 1.6) + 1) / 2)
            value = min(0.96, max(0.06, value + rng.uniform(-0.17, 0.17)))
            s.rect(
                x + c * cw,
                y + r * ch,
                cw - 1.2,
                ch - 1.2,
                blend("#E9F1F3", C["teal"], value),
                radius=1,
            )


def input_panel(s: SVG, b: dict[str, float]) -> None:
    w = b["w"]
    panel(s, "input", b, 1, "Single-cell inputs", "Preprocessed expression matrix")
    s.text(w / 2, 135, "Cells", 17, C["muted"], anchor="middle")
    gx, gy, gw, gh = 39, 178, w - 64, 132
    group_colors = [C["teal"], C["orange"], C["purple"]]
    for i, color in enumerate(group_colors):
        x = gx + gw * i / 3
        s.rect(x, 150, gw / 3 - 3, 20, blend("#FFFFFF", color, 0.15), radius=4)
        s.text(x + gw / 6 - 1.5, 165, chr(65 + i), 14, color, bold=True, anchor="middle")
    gene_matrix(s, gx, gy, gw, gh)
    # A rotated axis label is not used in collision accounting.
    s.raw(
        f'<text x="17" y="246" transform="rotate(-90 17 246)" text-anchor="middle" font-family="{FONT}" font-size="17" fill="{C["muted"]}">Genes</text>'
    )
    s.text(26, 349, "Cell-group labels", 19, bold=True)
    for i, color in enumerate(group_colors):
        x = 26 + i * 84
        s.rect(x, 366, 72, 34, blend("#FFFFFF", color, 0.12), radius=7)
        s.text(x + 36, 389, f"Group {chr(65 + i)}", 14, color, bold=True, anchor="middle")
    s.text(26, 437, "Candidate TF list", 19, bold=True)
    for i in range(3):
        x = 26 + i * 84
        s.rect(x, 453, 72, 33, C["soft"], C["line"], 6)
        s.text(x + 36, 476, f"TF{i + 1}", 16, C["muted"], anchor="middle")
    s.lines(
        26,
        529,
        ["All supplied cells remain", "available for each group."],
        18,
        fill=C["muted"],
        leading=24,
        max_width=w - 45,
    )
    s.end()


def glyph(
    s: SVG, x: float, y: float, r: float, group: int, fill: str, stroke: str | None = None
) -> None:
    if group == 0:
        s.circle(x, y, r, fill, stroke)
    elif group == 1:
        s.path(
            f"M{x},{y - r * 1.1} L{x - r},{y + r * 0.8} L{x + r},{y + r * 0.8} Z",
            stroke or fill,
            0.8,
            fill,
        )
    else:
        s.rect(x - r * 0.86, y - r * 0.86, r * 1.72, r * 1.72, fill, stroke, 2)


def star(s: SVG, x: float, y: float, r: float = 8) -> None:
    pts = []
    for j in range(10):
        a = -math.pi / 2 + j * math.pi / 5
        rr = r if j % 2 == 0 else r * 0.44
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    s.path("M" + " L".join(f"{a},{b}" for a, b in pts) + " Z", C["ink"], 1, C["white"])


def weighting_panel(s: SVG, b: dict[str, float]) -> None:
    w = b["w"]
    panel(s, "weights", b, 2, "Population-aware weights", "Repeat for each target group c", True)
    s.rect(18, 116, w - 36, 172, C["teal_soft"], radius=9)
    s.text(35, 140, "Target: A", 17, C["teal"], bold=True)
    s.text(w - 35, 140, "Related populations", 16, C["muted"], anchor="end")
    # Schematic cell locations, not a PCA plot and not measured patient data.
    centers = [(87, 196), (207, 190), (347, 207)]
    fixed = [
        (-27, -12),
        (-10, -20),
        (10, -24),
        (30, -7),
        (-32, 10),
        (-14, 1),
        (7, 0),
        (29, 17),
        (-15, 23),
        (8, 22),
    ]
    weights = [
        [1.0] * 10,
        [0.87, 0.79, 0.72, 0.65, 0.83, 0.75, 0.69, 0.58, 0.61, 0.64],
        [0.27, 0.20, 0.15, 0.12, 0.24, 0.18, 0.14, 0.09, 0.11, 0.13],
    ]
    colors = [C["teal"], C["orange"], C["purple"]]
    # Unarrowed dashed rays denote distances; they do not denote lineage.
    for (cx, cy), off in [(centers[1], fixed[0]), (centers[2], fixed[5])]:
        s.line(
            87,
            196,
            cx + off[0],
            cy + off[1],
            blend(C["white"], C["muted"], 0.5),
            1.2,
            stroke_dasharray="3 5",
        )
    for g, (cx, cy) in enumerate(centers):
        for k, (dx, dy) in enumerate(fixed):
            r = 3.0 + 4.2 * math.sqrt(weights[g][k])
            glyph(s, cx + dx, cy + dy, r, g, colors[g])
    star(s, 87, 196, 10)
    for i, (cx, _cy) in enumerate(centers):
        s.text(
            cx,
            250,
            ["A: w = 1", "B: w varies", "C: w varies"][i],
            15,
            colors[i],
            bold=True,
            anchor="middle",
        )
    s.text(w / 2, 276, "Star: target centroid · size: cell weight", 14, C["muted"], anchor="middle")
    s.text(w / 2, 317, "Representation → distance → kernel", 18, bold=True, anchor="middle")
    s.text(w / 2, 342, "PCA or expression · cosine or Euclidean", 16, C["muted"], anchor="middle")
    # Mode illustrations show cell-level weights for four cells in each of A/B/C.
    # Cell-distance row is maximum-one normalized; anchored rows fix all A cells.
    patterns = [
        (
            "Cell distance",
            [0.66, 0.94, 1.0, 0.75, 0.88, 0.70, 0.60, 0.52, 0.26, 0.19, 0.12, 0.08],
            False,
        ),
        (
            "Anchored cell",
            [1.0, 1.0, 1.0, 1.0, 0.88, 0.70, 0.60, 0.52, 0.26, 0.19, 0.12, 0.08],
            True,
        ),
        (
            "Group distance",
            [1.0, 1.0, 1.0, 1.0, 0.72, 0.72, 0.72, 0.72, 0.19, 0.19, 0.19, 0.19],
            False,
        ),
    ]
    startx = 226
    s.text(startx + 23, 370, "A", 13, C["teal"], bold=True, anchor="middle")
    s.text(startx + 79, 370, "B", 13, C["orange"], bold=True, anchor="middle")
    s.text(startx + 135, 370, "C", 13, C["purple"], bold=True, anchor="middle")
    for i, (label, values, default) in enumerate(patterns):
        y = 380 + i * 39
        if default:
            s.rect(20, y - 4, w - 40, 35, C["teal_soft"], radius=5)
        s.text(31, y + 18, label, 16, C["teal"] if default else C["ink"], bold=default)
        if default:
            s.rect(158, y + 3, 57, 19, C["teal"], radius=4)
            s.text(186.5, y + 17, "default", 11, C["white"], bold=True, anchor="middle")
        for j, val in enumerate(values):
            x = startx + (j // 4) * 56 + (j % 4) * 12.5
            s.rect(x, y, 10.5, 23, blend("#EDF3F4", C["teal"], val), radius=2)
    s.lines(
        24,
        514,
        [
            "Default: target cells = 1; external cells = K(d)",
            "Optional external-group size correction",
        ],
        16,
        leading=25,
        fill=C["muted"],
        max_width=w - 43,
    )
    s.end()


def tree_layout() -> dict[str, tuple[float, float]]:
    source = """digraph T {
      graph [rankdir=TB, nodesep=0.16, ranksep=0.28, margin=0];
      node [shape=circle, fixedsize=true, width=0.12, height=0.12, label=""];
      r -> a; r -> b; a -> a1; a -> a2; b -> b1; b -> b2;
    }"""
    data = json.loads(command(["dot", "-Tjson"], text=source))
    return {
        o["name"]: tuple(map(float, o["pos"].split(","))) for o in data["objects"] if "pos" in o
    }


def draw_tree(
    s: SVG,
    x: float,
    y: float,
    w: float,
    h: float,
    coords: dict[str, tuple[float, float]],
    color: str,
) -> None:
    xs = [p[0] for p in coords.values()]
    ys = [p[1] for p in coords.values()]
    xx = min(xs)
    yy = max(ys)
    ww = max(xs) - xx
    hh = yy - min(ys)
    points = {
        key: (x + (px - xx) / ww * w, y + (yy - py) / hh * h) for key, (px, py) in coords.items()
    }
    for a, b in [("r", "a"), ("r", "b"), ("a", "a1"), ("a", "a2"), ("b", "b1"), ("b", "b2")]:
        s.line(*points[a], *points[b], color, 1.8)
    for _key, (px, py) in points.items():
        s.circle(px, py, 3.8, C["white"], color, 1.5)


def model_panel(s: SVG, b: dict[str, float], trees: dict[str, tuple[float, float]]) -> None:
    w = b["w"]
    panel(s, "model", b, 3, "Weighted inference", "One model per group × target")
    s.rect(19, 121, w - 38, 107, C["soft"], radius=8)
    s.text(31, 145, "TF expression", 15, bold=True)
    s.text(215, 145, "Target", 13, C["muted"], bold=True, anchor="middle")
    s.text(w - 44, 145, "w(c)", 15, C["teal"], bold=True, anchor="middle")
    gene_matrix(s, 32, 158, 151, 55, rows=5, cols=8, seed=7)
    for i, v in enumerate([0.31, 0.73, 0.62, 0.24, 0.89]):
        s.rect(202, 158 + i * 11, 26, 9, blend("#E9F1F3", C["teal"], v), radius=1)
    for i, v in enumerate([1.0, 1.0, 0.77, 0.42, 0.13]):
        s.rect(w - 59, 158 + i * 11, 30, 9, blend("#E9F1F3", C["teal"], v), radius=1)
    s.arrow([(w / 2, 233), (w / 2, 253)], C["teal"], 2, 6)
    for i in range(3):
        draw_tree(s, 32 + i * 95, 270, 67, 64, trees, C["teal"] if i == 1 else C["muted"])
    s.text(w / 2, 362, "Extra-Trees / Random Forest", 17, bold=True, anchor="middle")
    s.arrow([(w / 2, 374), (w / 2, 396)], C["muted"], 2, 6)
    s.text(28, 422, "TF importance", 18, bold=True)
    for i, value in enumerate([1.0, 0.64, 0.34]):
        yy = 437 + i * 22
        s.text(29, yy + 13, f"TF{i + 1}", 14, C["muted"])
        s.rect(
            72,
            yy,
            195 * value,
            13,
            C["teal"] if i == 0 else blend("#FFFFFF", C["teal"], 0.65 - i * 0.12),
            radius=3,
        )
    s.lines(
        26,
        529,
        ["Weighted impurity reduction", "→ candidate TF–target edges"],
        16,
        leading=24,
        fill=C["muted"],
        max_width=w - 40,
    )
    s.end()


def network_layout() -> dict[str, tuple[float, float]]:
    source = """digraph N {
      graph [rankdir=LR, nodesep=0.18, ranksep=0.95, margin=0];
      node [shape=circle, width=0.20, height=0.20, label="", fixedsize=true];
      {rank=same; T1; T2;} {rank=same; g1; g2; g3;}
      T1 -> g1; T1 -> g2; T1 -> g3;
      T2 -> g1; T2 -> g2; T2 -> g3;
    }"""
    data = json.loads(command(["dot", "-Tjson"], text=source))
    return {
        o["name"]: tuple(map(float, o["pos"].split(","))) for o in data["objects"] if "pos" in o
    }


def draw_network(
    s: SVG,
    x: float,
    y: float,
    w: float,
    h: float,
    coords: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    color: str,
) -> None:
    xs = [p[0] for p in coords.values()]
    ys = [p[1] for p in coords.values()]
    points = {
        key: (
            x + (px - min(xs)) / (max(xs) - min(xs)) * w,
            y + (max(ys) - py) / (max(ys) - min(ys)) * h,
        )
        for key, (px, py) in coords.items()
    }
    for aa, bb in edges:
        a, b = points[aa], points[bb]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        p = (a[0] + 10 * ux, a[1] + 10 * uy)
        q = (b[0] - 10 * ux, b[1] - 10 * uy)
        s.arrow([p, q], color, 1.65, 5)
    for key, (xx, yy) in points.items():
        if key.startswith("T"):
            s.rect(xx - 8, yy - 8, 16, 16, blend("#FFFFFF", color, 0.18), color, 3, 1.5)
        else:
            s.circle(xx, yy, 8, C["white"], color, 1.5)


def output_panel(s: SVG, b: dict[str, float], net: dict[str, tuple[float, float]]) -> None:
    w = b["w"]
    panel(s, "output", b, 4, "Group-specific GRNs", "One network for each group")
    s.text(129, 135, "TFs", 14, C["muted"], anchor="middle")
    s.text(w - 44, 135, "Targets", 14, C["muted"], anchor="middle")
    sets = [
        [("T1", "g1"), ("T1", "g2"), ("T2", "g3")],
        [("T1", "g2"), ("T2", "g2"), ("T2", "g3")],
        [("T1", "g1"), ("T2", "g1"), ("T2", "g2")],
    ]
    for i, edges in enumerate(sets):
        y = 163 + i * 103
        color = [C["teal"], C["orange"], C["purple"]][i]
        s.rect(18, y - 14, w - 36, 95, blend("#FFFFFF", color, 0.035), radius=7)
        s.circle(48, y + 34, 18, blend("#FFFFFF", color, 0.14))
        s.text(48, y + 41, chr(65 + i), 20, color, bold=True, anchor="middle")
        draw_network(s, 129, y, 145, 60, net, edges, color)
    s.line(24, 469, w - 24, 469, C["line"], 1)
    s.text(25, 502, "Weight diagnostics", 18, bold=True)
    s.lines(
        25,
        529,
        ["Effective sample size +", "source-group contributions"],
        16,
        leading=24,
        fill=C["muted"],
        max_width=w - 40,
    )
    s.end()


def validate(width: float, height: float, boxes: dict[str, dict[str, float]]) -> dict[str, Any]:
    problems = []
    keys = list(boxes)
    for i, a in enumerate(keys):
        p = boxes[a]
        for b in keys[i + 1 :]:
            q = boxes[b]
            if (
                min(p["x"] + p["w"], q["x"] + q["w"]) > max(p["x"], q["x"]) + 0.1
                and min(p["y"] + p["h"], q["y"] + q["h"]) > max(p["y"], q["y"]) + 0.1
            ):
                problems.append(f"Panel overlap: {a}, {b}")
    for t in TEXT_BOXES:
        if t["x0"] < -0.5 or t["y0"] < -0.5 or t["x1"] > width + 0.5 or t["y1"] > height + 0.5:
            problems.append("Text outside canvas: " + t["text"])
        if t["region"] in boxes:
            r = boxes[t["region"]]
            if (
                t["x0"] < r["x"] + 6
                or t["x1"] > r["x"] + r["w"] - 6
                or t["y0"] < r["y"] + 6
                or t["y1"] > r["y"] + r["h"] - 6
            ):
                problems.append("Text outside panel padding: " + t["text"])
    for i, a in enumerate(TEXT_BOXES):
        for b in TEXT_BOXES[i + 1 :]:
            dx = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
            dy = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
            if dx > 0.8 and dy > 0.8:
                problems.append(f"Text collision: {a['text']!r} / {b['text']!r}")
    report = dict(
        passed=not problems,
        checks=[
            "panel disjointness",
            "text within canvas",
            "text within panel padding",
            "pairwise text bounding-box intersections",
        ],
        note="Analytic checks cover text and major panels, not all decorative primitives. Inspect the rendered PNG/PDF as well.",
        problems=problems,
        panel_count=len(boxes),
        text_item_count=len(TEXT_BOXES),
        canvas=dict(width=width, height=height),
        panels=boxes,
    )
    if problems:
        raise ValueError("Layout checks failed:\n" + "\n".join(problems))
    return report


def build(output: Path, exports: bool = True) -> None:
    TEXT_BOXES.clear()
    REGIONS.clear()
    output.mkdir(parents=True, exist_ok=True)
    raw, graphw, graphh = top_layout()
    margin = 30.0
    top = 137.0
    boxes = {k: dict(x=v["x"] + margin, y=v["y"] + top, w=v["w"], h=v["h"]) for k, v in raw.items()}
    width = graphw + 2 * margin
    bottom = top + graphh
    height = bottom + 142
    s = SVG()
    s.rect(0, 0, width, height, C["white"])
    header(s, width)
    trees = tree_layout()
    nets = network_layout()
    input_panel(s, boxes["input"])
    weighting_panel(s, boxes["weights"])
    model_panel(s, boxes["model"], trees)
    output_panel(s, boxes["output"], nets)
    # Global arrows use node bounding boxes, not hand-positioned coordinates.
    for left, right in [("input", "weights"), ("weights", "model"), ("model", "output")]:
        a, b = boxes[left], boxes[right]
        y = a["y"] + a["h"] * 0.5
        s.arrow([(a["x"] + a["w"] + 7, y), (b["x"] - 7, y)], C["muted"], 2.6, 7)
    # A separate data lane makes the inference / distance-space distinction explicit.
    a, b = boxes["input"], boxes["model"]
    x1 = a["x"] + a["w"] / 2
    x2 = b["x"] + b["w"] / 2
    lane = bottom + 39
    s.arrow([(x1, bottom + 6), (x1, lane), (x2, lane), (x2, bottom + 6)], C["ink"], 2.0, 7.0)
    s.text(
        (x1 + x2) / 2,
        lane + 29,
        "Original expression + TF predictors feed the models directly",
        18,
        C["ink"],
        bold=True,
        anchor="middle",
    )
    s.text(
        (x1 + x2) / 2,
        lane + 54,
        "The distance representation only determines cell weights.",
        17,
        C["muted"],
        anchor="middle",
    )
    s.line(margin, height - 34, width - margin, height - 34, C["line"], 1)
    s.text(
        width / 2,
        height - 11,
        "Schematic illustrations · Inferred edges are predictive hypotheses, not causal or signed regulatory evidence.",
        16,
        C["muted"],
        anchor="middle",
        max_width=width - 60,
    )
    report = validate(width, height, boxes)
    mm = float(CONFIG["physical_width_mm"])
    metadata = escape(
        json.dumps(
            dict(
                repository=CONFIG["repository"],
                commit=CONFIG["commit"],
                origin="Procedural vector illustration; no image-generation model",
                schematic=True,
            )
        )
    )
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{mm}mm" height="{mm * height / width:.6f}mm"
 viewBox="0 0 {width:.6f} {height:.6f}" role="img" aria-labelledby="title desc">
<title id="title">SPATHI: population-aware gene-regulatory network inference</title>
<desc id="desc">All preprocessed cells contribute to a separate network for each group through transcriptomic-similarity weights. Distance representations define sample weights; models use the supplied expression values. Weighted tree ensembles return directed, unsigned predictive TF-target hypotheses. All matrices, cell positions, weights and network edges shown are schematic, not experimental results.</desc>
<metadata>{metadata}</metadata>
{"".join(s.parts)}
</svg>
'''
    base = output / "SPATHI_graphical_abstract"
    base.with_suffix(".svg").write_text(xml, encoding="utf-8")
    (output / "layout_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if exports:
        if not shutil.which("inkscape"):
            raise RuntimeError(
                f"SVG was created at {base.with_suffix('.svg')}. Install Inkscape for vector PDF and PNG exports, or use --svg-only."
            )
        command(
            [
                "inkscape",
                str(base.with_suffix(".svg")),
                "--export-type=pdf",
                "--export-area-page",
                "--export-filename=" + str(base.with_suffix(".pdf")),
            ]
        )
        command(
            [
                "inkscape",
                str(base.with_suffix(".svg")),
                "--export-type=png",
                "--export-area-page",
                "--export-width=2400",
                "--export-filename=" + str(base.with_suffix(".png")),
            ]
        )
    print(
        f"Built {base.with_suffix('.svg')}\nLayout checks: passed ({len(TEXT_BOXES)} text items)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--svg-only", action="store_true", help="Do not invoke Inkscape.")
    args = parser.parse_args()
    try:
        build(args.output_dir.resolve(), exports=not args.svg_only)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Build error: {exc}", file=sys.stderr)
        sys.exit(1)
