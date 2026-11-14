"""Build the college project report PDF (needs `pip install reportlab` and macOS system fonts).

Run after `python -m report.collect_stats` and `python -m report.make_figures`:
    python -m report.build_report
"""
import ast
import io
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from reportlab import rl_config
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

from ml.config import CHECKPOINT_PATH, CLASS_NAMES, REPORTS_DIR, ROOT
from ml.signs import describe

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "SignSense_Project_Report_Vaibhav_Anand_Singh.pdf"

STUDENT = "Vaibhav Anand Singh"
ROLL = "BTECH/10261/23"
GUIDE = "Dr. Jamimamul Bakas"
DEPT = "Department of Computer Science and Engineering"
INSTITUTE = "Birla Institute of Technology, Mesra"
SESSION = "7th Semester, Session 2026-27 (Monsoon 2026)"
TITLE = "Traffic Sign Recognition for Autonomous Vehicles"
REPO = "https://github.com/Vaibhavanandsingh/sign_sense"

STATS = json.loads((FIG / "stats.json").read_text())
METRICS = json.loads((REPORTS_DIR / "metrics.json").read_text())
HISTORY = json.loads((REPORTS_DIR / "history.json").read_text())

NAVY = colors.HexColor("#1f3864")
ACCENT = colors.HexColor("#1f4e79")
LIGHT = colors.HexColor("#eef3f9")
GRID = colors.HexColor("#9aa5b1")
CODE_BG = colors.HexColor("#f6f8fa")

# ---------------------------------------------------------------- fonts and styles

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
for name, file in [("TNR", "Times New Roman.ttf"), ("TNR-Bold", "Times New Roman Bold.ttf"),
                   ("TNR-Italic", "Times New Roman Italic.ttf"),
                   ("TNR-BoldItalic", "Times New Roman Bold Italic.ttf"), ("Mono", "Courier New.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-Bold", italic="TNR-Italic",
                              boldItalic="TNR-BoldItalic")
rl_config.canvas_basefontname = "TNR"

S = {
    "body": ParagraphStyle("body", fontName="TNR", fontSize=12, leading=18, alignment=TA_JUSTIFY,
                           spaceAfter=8),
    "bullet": ParagraphStyle("bullet", fontName="TNR", fontSize=12, leading=17, alignment=TA_JUSTIFY,
                             leftIndent=22, bulletIndent=8, spaceAfter=4),
    "chapter_no": ParagraphStyle("chapter_no", fontName="TNR-Bold", fontSize=14, leading=18,
                                 alignment=TA_CENTER, textColor=ACCENT, spaceBefore=30),
    "chapter": ParagraphStyle("chapter", fontName="TNR-Bold", fontSize=20, leading=26,
                              alignment=TA_CENTER, textColor=NAVY, spaceBefore=6, spaceAfter=26),
    "front": ParagraphStyle("front", fontName="TNR-Bold", fontSize=18, leading=24,
                            alignment=TA_CENTER, textColor=NAVY, spaceBefore=10, spaceAfter=22),
    "h2": ParagraphStyle("h2", fontName="TNR-Bold", fontSize=14, leading=18, textColor=NAVY,
                         spaceBefore=14, spaceAfter=8, keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName="TNR-Bold", fontSize=12, leading=16, spaceBefore=10,
                         spaceAfter=6, keepWithNext=1),
    "caption": ParagraphStyle("caption", fontName="TNR-Italic", fontSize=10.5, leading=13,
                              alignment=TA_CENTER, spaceBefore=4, spaceAfter=14),
    "table_caption": ParagraphStyle("table_caption", fontName="TNR-Bold", fontSize=10.5, leading=13,
                                    alignment=TA_CENTER, spaceBefore=6, spaceAfter=6, keepWithNext=1),
    "cell": ParagraphStyle("cell", fontName="TNR", fontSize=10, leading=12.5),
    "cell_c": ParagraphStyle("cell_c", fontName="TNR", fontSize=10, leading=12.5, alignment=TA_CENTER),
    "cell_r": ParagraphStyle("cell_r", fontName="TNR", fontSize=10, leading=12.5, alignment=TA_RIGHT),
    "cell_h": ParagraphStyle("cell_h", fontName="TNR-Bold", fontSize=10, leading=12.5,
                             textColor=colors.white),
    "equation": ParagraphStyle("equation", fontName="TNR-Italic", fontSize=12, leading=18,
                               alignment=TA_CENTER, spaceBefore=2, spaceAfter=6),
    "ref": ParagraphStyle("ref", fontName="TNR", fontSize=11, leading=15, leftIndent=26,
                          firstLineIndent=-26, spaceAfter=6, alignment=TA_LEFT),
    "center": ParagraphStyle("center", fontName="TNR", fontSize=12, leading=17, alignment=TA_CENTER),
}


def toc_styles(base: int = 12) -> list[ParagraphStyle]:
    return [
        ParagraphStyle("toc0", fontName="TNR-Bold", fontSize=base, leading=base + 6, leftIndent=0,
                       firstLineIndent=0, spaceBefore=4),
        ParagraphStyle("toc1", fontName="TNR", fontSize=base - 1, leading=base + 4, leftIndent=22,
                       firstLineIndent=0),
    ]


# ---------------------------------------------------------------- document template

def roman(n: int) -> str:
    out = ""
    for value, numeral in [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]:
        while n >= value:
            out += numeral
            n -= value
    return out


def format_page(page: int) -> str:
    return roman(-page) if page < 0 else str(page)


class EntryList(TableOfContents):
    """A TableOfContents that listens to its own notification kind (figures, tables)."""

    def __init__(self, kind: str, **kwargs):
        super().__init__(**kwargs)
        self.kind = kind

    def notify(self, kind, stuff):
        if kind == self.kind:
            self.addEntry(*stuff)


class ReportDoc(BaseDocTemplate):
    def __init__(self, path: Path):
        super().__init__(str(path), pagesize=A4, leftMargin=3.0 * cm, rightMargin=2.2 * cm,
                         topMargin=2.6 * cm, bottomMargin=2.4 * cm, title=TITLE, author=STUDENT,
                         subject=f"B.Tech project report, {DEPT}, {INSTITUTE}",
                         creator="report/build_report.py")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate("title", [frame], onPage=self.draw_title_page),
            PageTemplate("front", [frame], onPage=self.draw_front_page),
            PageTemplate("body", [frame], onPage=self.draw_body_page),
        ])
        self.body_start = None

    def logical_page(self) -> int:
        """Body pages are positive; front-matter pages are negative so format_page renders them in roman."""
        if self.pageTemplate.id == "body":
            return self.page - self.body_start + 1
        return -(self.page - 1)

    def draw_title_page(self, canv, doc):
        self.body_start = None
        w, h = A4
        canv.saveState()
        canv.setStrokeColor(NAVY)
        canv.setLineWidth(2.2)
        canv.rect(1.4 * cm, 1.4 * cm, w - 2.8 * cm, h - 2.8 * cm)
        canv.setLineWidth(0.6)
        canv.rect(1.6 * cm, 1.6 * cm, w - 3.2 * cm, h - 3.2 * cm)
        canv.restoreState()

    def draw_front_page(self, canv, doc):
        canv.saveState()
        canv.setFont("TNR", 11)
        canv.drawCentredString(A4[0] / 2, 1.4 * cm, roman(doc.page - 1))
        canv.restoreState()

    def draw_body_page(self, canv, doc):
        if self.body_start is None:
            self.body_start = doc.page
        w, h = A4
        canv.saveState()
        canv.setFont("TNR-Italic", 9.5)
        canv.setFillColor(colors.HexColor("#555555"))
        top = h - 1.7 * cm
        canv.drawString(self.leftMargin, top, TITLE)
        canv.drawRightString(w - self.rightMargin, top, "BIT Mesra, Dept. of CSE")
        canv.setStrokeColor(GRID)
        canv.setLineWidth(0.5)
        canv.line(self.leftMargin, top - 4, w - self.rightMargin, top - 4)
        canv.line(self.leftMargin, 1.95 * cm, w - self.rightMargin, 1.95 * cm)
        canv.setFont("TNR", 9.5)
        canv.drawString(self.leftMargin, 1.45 * cm, f"{STUDENT} ({ROLL})")
        canv.setFont("TNR", 11)
        canv.setFillColor(colors.black)
        canv.drawRightString(w - self.rightMargin, 1.45 * cm, str(doc.page - self.body_start + 1))
        canv.restoreState()

    def afterFlowable(self, flowable):
        entry = getattr(flowable, "toc_entry", None)
        if entry:
            kind, level, text, key = entry
            self.canv.bookmarkPage(key)
            if kind == "TOCEntry":
                self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify(kind, (level, text, self.logical_page(), key))


# ---------------------------------------------------------------- story helpers

class Story:
    def __init__(self, width: float):
        self.width = width
        self.items: list = []
        self.chapter_label = ""
        self.section = 0
        self.figures = 0
        self.tables = 0
        self.keys = 0

    def add(self, *flowables):
        self.items.extend(flowables)

    def _key(self) -> str:
        self.keys += 1
        return f"k{self.keys}"

    def _mark(self, flowable, kind: str, level: int, text: str):
        flowable.toc_entry = (kind, level, text, self._key())
        return flowable

    def front_heading(self, text: str, in_toc: bool = True):
        p = Paragraph(text, S["front"])
        if in_toc:
            self._mark(p, "TOCEntry", 0, text)
        self.add(p)

    def chapter(self, number, title: str, appendix: bool = False):
        self.chapter_label = str(number)
        self.section = self.figures = self.tables = 0
        label = f"APPENDIX {number}" if appendix else f"CHAPTER {number}"
        toc_text = f"Appendix {number}: {title}" if appendix else f"{number}. {title}"
        self.add(PageBreak(), Paragraph(label, S["chapter_no"]),
                 self._mark(Paragraph(title, S["chapter"]), "TOCEntry", 0, toc_text))

    def h2(self, title: str, space_needed_cm: float = 3):
        self.section += 1
        text = f"{self.chapter_label}.{self.section} {title}"
        self.add(CondPageBreak(space_needed_cm * cm), self._mark(Paragraph(text, S["h2"]), "TOCEntry", 1, text))

    def h3(self, title: str):
        self.add(Paragraph(title, S["h3"]))

    def p(self, text: str):
        self.add(Paragraph(text, S["body"]))

    def bullets(self, items: list[str]):
        for item in items:
            self.add(Paragraph(item, S["bullet"], bulletText="•"))
        self.add(Spacer(1, 4))

    def numbered(self, items: list[str]):
        for i, item in enumerate(items, 1):
            self.add(Paragraph(item, S["bullet"], bulletText=f"{i}."))
        self.add(Spacer(1, 4))

    def equation(self, text: str):
        self.add(Paragraph(text, S["equation"]))

    def figure(self, source, caption: str, width_cm: float = 15.5, max_height_cm: float = 20):
        self.figures += 1
        number = f"{self.chapter_label}.{self.figures}"
        if isinstance(source, Drawing):
            flow = source
        else:
            img = PILImage.open(source) if not isinstance(source, PILImage.Image) else source
            w = min(width_cm * cm, self.width)
            h = w * img.height / img.width
            if h > max_height_cm * cm:
                h = max_height_cm * cm
                w = h * img.width / img.height
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            flow = Image(buffer, width=w, height=h)
        cap = Paragraph(f"<b>Figure {number}:</b> {caption}", S["caption"])
        self._mark(cap, "LOF", 0, f"Figure {number}: {plain(caption)}")
        self.add(KeepTogether([Spacer(1, 4), flow, cap]))
        return number

    def next_figure(self) -> str:
        return f"{self.chapter_label}.{self.figures + 1}"

    def next_table(self) -> str:
        return f"{self.chapter_label}.{self.tables + 1}"

    def table(self, header: list[str], rows: list[list], widths: list[float], caption: str,
              align: list[str] | None = None, font_size: float = 10, bold_rows: tuple = (),
              repeat: bool = True):
        self.tables += 1
        number = f"{self.chapter_label}.{self.tables}"
        align = align or ["l"] * len(header)
        cell_styles = {"l": S["cell"], "c": S["cell_c"], "r": S["cell_r"]}

        def cell(value, a, bold=False):
            style = ParagraphStyle("tmp", parent=cell_styles[a], fontSize=font_size,
                                   leading=font_size * 1.25)
            text = str(value)
            return Paragraph(f"<b>{text}</b>" if bold else text, style)

        head_style = ParagraphStyle("h", parent=S["cell_h"], fontSize=font_size, leading=font_size * 1.25)
        data = [[Paragraph(h, head_style) for h in header]]
        for r, row in enumerate(rows):
            data.append([cell(v, a, r in bold_rows) for v, a in zip(row, align)])
        total = sum(widths)
        t = Table(data, colWidths=[w / total * self.width for w in widths], repeatRows=1 if repeat else 0)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.4, GRID),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        short = len(rows) <= 16
        cap_style = S["table_caption"] if short else ParagraphStyle("cap_long", parent=S["table_caption"],
                                                                      keepWithNext=0)
        cap = Paragraph(f"Table {number}: {caption}", cap_style)
        self._mark(cap, "LOT", 0, f"Table {number}: {plain(caption)}")
        if short:
            self.add(KeepTogether([cap, t, Spacer(1, 10)]))
        else:
            self.add(CondPageBreak(5 * cm), cap, t, Spacer(1, 10))
        return number

    def code(self, lines: list[str], first_line: int | None = 1, font_size: float = 7.6):
        number_w = 24 if first_line else 0
        max_chars = int((self.width - number_w - 12) / (font_size * 0.6))
        rows = []
        for offset, line in enumerate(lines):
            line = line.rstrip("\n").expandtabs(4)
            label = str(first_line + offset) if first_line else ""
            chunks, rest, limit = [], line, max_chars
            while len(rest) > limit:
                cut = max(rest.rfind(" ", 0, limit), rest.rfind(",", 0, limit) + 1)
                if cut < limit // 2:
                    cut = limit
                chunks.append(rest[:cut].rstrip())
                rest = "        " + rest[cut:].lstrip()
            chunks.append(rest)
            for i, chunk in enumerate(chunks):
                rows.append([label if i == 0 else "", chunk] if first_line else [chunk])
        widths = [number_w, self.width - number_w] if first_line else [self.width]
        t = Table(rows, colWidths=widths, splitByRow=1)
        style = [
            ("FONTNAME", (0, 0), (-1, -1), "Mono"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size * 1.22),
            ("TOPPADDING", (0, 0), (-1, -1), 0.35),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0.35),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, GRID),
            ("TOPPADDING", (0, 0), (-1, 0), 4),
            ("BOTTOMPADDING", (0, -1), (-1, -1), 4),
        ]
        if first_line:
            style += [("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#8a8f98")),
                      ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                      ("LINEAFTER", (0, 0), (0, -1), 0.4, GRID)]
        t.setStyle(TableStyle(style))
        if len(rows) <= 30:
            self.add(KeepTogether([t, Spacer(1, 10)]))
        else:
            self.add(t, Spacer(1, 10))

    def listing(self, title: str, relpath: str, function: str | None = None):
        path = ROOT / relpath
        lines = path.read_text().splitlines()
        start = 1
        if function:
            node = next(n for n in ast.walk(ast.parse(path.read_text()))
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function)
            start = node.lineno
            lines = lines[node.lineno - 1:node.end_lineno]
        self.h2(title)
        self.add(Paragraph(f"File: <font face='Mono' size='10'>{relpath}</font>"
                           + (f", function <font face='Mono' size='10'>{function}()</font>" if function else "")
                           + f" ({len(lines)} lines)", S["body"]))
        self.code(lines, first_line=start)


def plain(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def C(text: str) -> str:
    return f"<font face='Mono' size='10.5'>{text}</font>"


def pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"


def crop_screenshot(path: Path) -> PILImage.Image:
    """Trim the empty area below the app content (the sidebar keeps its full height)."""
    img = PILImage.open(path).convert("RGB")
    main = np.asarray(img.convert("L"))[:, int(img.width * 0.3):]
    rows = np.flatnonzero(main.min(axis=1) < 235)
    bottom = min(img.height, int(rows[-1]) + 24 if rows.size else img.height)
    return img.crop((0, 0, img.width, bottom))


# ---------------------------------------------------------------- diagrams

def _arrow(d: Drawing, x1, y1, x2, y2, color=NAVY, width=1.0, head=5):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width))
    angle = math.atan2(y2 - y1, x2 - x1)
    left = (x2 - head * math.cos(angle - 0.4), y2 - head * math.sin(angle - 0.4))
    right = (x2 - head * math.cos(angle + 0.4), y2 - head * math.sin(angle + 0.4))
    d.add(Polygon([x2, y2, *left, *right], fillColor=color, strokeColor=color, strokeWidth=0.5))


def _box(d: Drawing, x, y, w, h, lines: list[str], fill, stroke=NAVY):
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=stroke, strokeWidth=0.9))
    step = 9.6
    top = y + h / 2 + (len(lines) - 1) * step / 2 - 3
    for i, text in enumerate(lines):
        d.add(String(x + w / 2, top - i * step, text, textAnchor="middle",
                     fontName="TNR-Bold" if i == 0 else ("TNR-Italic" if i == 1 else "TNR"),
                     fontSize=8.6 if i == 0 else 7.7, fillColor=NAVY if i == 0 else colors.black))


def _label(d: Drawing, x, y, text, size=7, anchor="middle", font="TNR-Italic", color=colors.HexColor("#333333")):
    d.add(String(x, y, text, textAnchor=anchor, fontName=font, fontSize=size, fillColor=color))


def architecture_diagram(width: float) -> Drawing:
    W, H = 448, 336
    d = Drawing(W, H)
    d.add(Rect(0, 184, W, 150, rx=6, ry=6, fillColor=colors.HexColor("#eef3f9"),
               strokeColor=colors.HexColor("#c5d3e3"), strokeWidth=0.8))
    d.add(Rect(0, 0, W, 172, rx=6, ry=6, fillColor=colors.HexColor("#f7f2e8"),
               strokeColor=colors.HexColor("#e2d4b8"), strokeWidth=0.8))
    _label(d, 8, 320, "OFFLINE: MODEL DEVELOPMENT  (ml/)", 8, "start", "TNR-Bold", NAVY)
    _label(d, 8, 158, "ONLINE: INFERENCE AND DRIVING SIMULATION  (backend/, frontend/)", 8, "start",
           "TNR-Bold", colors.HexColor("#7a5a1c"))
    bw, gap = 92, (W - 8 - 4 * 92) / 3
    xs = [4 + i * (bw + gap) for i in range(4)]
    white = colors.white
    top = [
        ["GTSRB dataset", "torchvision loader", "43 classes", "26,640 train / 12,630 test"],
        ["Data pipeline", "ml/data.py", "resize 48×48, augment,", "normalise, 90/10 split"],
        ["Model training", "ml/train.py", "SignNet CNN, AdamW,", "cosine LR, Apple MPS"],
        ["Evaluation", "ml/evaluate.py", "sign_cnn.pt, metrics.json,", "curves, confusion matrix"],
    ]
    for x, lines in zip(xs, top):
        _box(d, x, 214, bw, 76, lines, white)
    for i in range(3):
        _arrow(d, xs[i] + bw + 2, 252, xs[i + 1] - 2, 252)
    bottom = [
        ["User / camera", "input source", "upload, test sample", "or webcam frame"],
        ["Streamlit frontend", "frontend/app.py", "Classify, Batch, Drive", "simulation, Model report"],
        ["FastAPI backend", "backend/main.py", "validation, size limits,", "Pydantic JSON schemas"],
        ["SignClassifier", "backend/inference.py", "preprocess, CNN forward,", "softmax top-k"],
    ]
    for x, lines in zip(xs, bottom):
        _box(d, x, 74, bw, 72, lines, white, colors.HexColor("#7a5a1c"))
    _arrow(d, xs[0] + bw + 2, 116, xs[1] - 2, 116)
    _label(d, (xs[0] + bw + xs[1]) / 2, 120, "image")
    _arrow(d, xs[1] + bw + 2, 122, xs[2] - 2, 122)
    _label(d, (xs[1] + bw + xs[2]) / 2, 126, "POST")
    _arrow(d, xs[2] - 2, 98, xs[1] + bw + 2, 98)
    _label(d, (xs[1] + bw + xs[2]) / 2, 89, "JSON")
    _arrow(d, xs[2] + bw + 2, 122, xs[3] - 2, 122)
    _label(d, (xs[2] + bw + xs[3]) / 2, 126, "tensor")
    _arrow(d, xs[3] - 2, 98, xs[2] + bw + 2, 98)
    _label(d, (xs[2] + bw + xs[3]) / 2, 89, "top-k")
    _arrow(d, xs[3] + bw / 2, 212, xs[3] + bw / 2, 148, colors.HexColor("#2e7d32"), 1.3)
    _label(d, xs[3] + bw / 2 - 5, 200, "checkpoint loaded", 7, "end")
    _label(d, xs[3] + bw / 2 - 5, 191, "once at start-up", 7, "end")
    _box(d, xs[1], 10, bw, 50, ["Rule-based planner", "plan() in app.py", "speed limit + command"], white,
         colors.HexColor("#7a5a1c"))
    _arrow(d, xs[1] + bw / 2, 72, xs[1] + bw / 2, 62)
    _box(d, xs[2], 10, bw, 50, ["Sign semantics", "ml/signs.py", "category, action, limit"], white,
         colors.HexColor("#7a5a1c"))
    _arrow(d, xs[2] + bw / 2, 62, xs[2] + bw / 2, 72)
    scale = width / W
    d.scale(scale, scale)
    d.width, d.height = W * scale, H * scale
    return d


def signnet_diagram(width: float) -> Drawing:
    W, H = 448, 196
    d = Drawing(W, H)
    cy = 112
    items = [
        ("Input", "3×48×48", 6, 92, "tensor"),
        ("Block 1", "32×24×24", 13, 58, "tensor"),
        ("Block 2", "64×12×12", 20, 40, "tensor"),
        ("Block 3", "128×6×6", 28, 30, "tensor"),
        ("Flatten", "4608", 7, 104, "vector"),
        ("FC 1", "256", 7, 66, "vector"),
        ("Output", "43", 7, 40, "vector"),
    ]
    ops = ["conv block", "conv block", "conv block", "flatten", "FC + ReLU", "FC"]
    slot = (W - 20) / len(items)
    centers = [10 + slot * (i + 0.5) for i in range(len(items))]
    depth = 7
    fills = [colors.HexColor("#dfe8f3"), colors.HexColor("#b9cde4"), colors.HexColor("#8fb0d6"),
             colors.HexColor("#5f8fc4"), colors.HexColor("#f3d9b1"), colors.HexColor("#efc27f"),
             colors.HexColor("#e39b3d")]
    for (name, shape, w, h, kind), cx, fill in zip(items, centers, fills):
        x, y = cx - w / 2, cy - h / 2
        if kind == "tensor":
            d.add(Polygon([x, y + h, x + depth, y + h + depth, x + w + depth, y + h + depth, x + w, y + h],
                          fillColor=fill, strokeColor=NAVY, strokeWidth=0.6))
            d.add(Polygon([x + w, y, x + w + depth, y + depth, x + w + depth, y + h + depth, x + w, y + h],
                          fillColor=fill, strokeColor=NAVY, strokeWidth=0.6))
        d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=NAVY, strokeWidth=0.7))
        _label(d, cx, cy - 66, name, 8.6, "middle", "TNR-Bold", NAVY)
        _label(d, cx, cy - 77, shape, 8, "middle", "TNR", colors.black)
    for i, op in enumerate(ops):
        x1 = centers[i] + items[i][2] / 2 + (depth if items[i][4] == "tensor" else 0) + 4
        x2 = centers[i + 1] - items[i + 1][2] / 2 - 4
        _arrow(d, x1, cy, x2, cy, NAVY, 0.9, 4.5)
        _label(d, (x1 + x2) / 2, cy + 6, op, 7)
    _label(d, centers[-1], cy + 30, "softmax", 7)
    _label(d, W / 2, 12, "conv block = 2 × [Conv 3×3 (padding 1) → BatchNorm → ReLU] → MaxPool 2×2 → Dropout "
           "(0.1, 0.2, 0.3 for blocks 1–3)", 7.6, "middle", "TNR", colors.HexColor("#333333"))
    _label(d, W / 2, 2, "FC 1 is followed by Dropout 0.5; the output layer produces 43 logits", 7.6,
           "middle", "TNR", colors.HexColor("#333333"))
    scale = width / W
    d.scale(scale, scale)
    d.width, d.height = W * scale, H * scale
    return d


# ---------------------------------------------------------------- derived numbers

TRAIN_COUNTS = STATS["class_counts"]["train"]
TEST_COUNTS = STATS["class_counts"]["test"]
N_TRAIN_TOTAL = sum(TRAIN_COUNTS)
N_TEST = METRICS["num_test_images"]
N_VAL = int(N_TRAIN_TOTAL * 0.1)
N_TRAIN = N_TRAIN_TOTAL - N_VAL
TEST_ACC = METRICS["test_accuracy"]
MACRO_F1 = METRICS["macro_f1"]
N_CORRECT = round(TEST_ACC * N_TEST)
N_WRONG = N_TEST - N_CORRECT
BEST_EPOCH = max(range(len(HISTORY["val_acc"])), key=lambda i: (HISTORY["val_acc"][i], -i)) + 1
BEST_VAL = HISTORY["val_acc"][BEST_EPOCH - 1]
N_PARAMS = sum(layer["params"] for layer in STATS["layers"])
MODEL_MB = CHECKPOINT_PATH.stat().st_size / 1e6
API = STATS["api"]
F1_BY_ID = [METRICS["per_class_f1"][n] for n in CLASS_NAMES]
MIN_COUNT, MAX_COUNT = min(TRAIN_COUNTS), max(TRAIN_COUNTS)
MIN_IDS = [i for i, c in enumerate(TRAIN_COUNTS) if c == MIN_COUNT]
MAX_IDS = [i for i, c in enumerate(TRAIN_COUNTS) if c == MAX_COUNT]
WORST = sorted(range(43), key=lambda i: F1_BY_ID[i])[:5]
PERFECT = [i for i in range(43) if F1_BY_ID[i] >= 0.99995]
ABOVE_98 = sum(f >= 0.98 for f in F1_BY_ID)


def block_params(prefix: str) -> int:
    return sum(layer["params"] for layer in STATS["layers"] if layer["name"].startswith(prefix))


def id_list(ids: list[int]) -> str:
    return ", ".join(str(i) for i in ids)


# ---------------------------------------------------------------- content

def title_page(s: Story):
    def para(text, size, font="TNR", lead=None, space=0, color=colors.black):
        return Paragraph(text, ParagraphStyle("t", fontName=font, fontSize=size, leading=lead or size * 1.3,
                                              alignment=TA_CENTER, spaceAfter=space, textColor=color))

    s.add(Spacer(1, 1.2 * cm),
          para("A PROJECT REPORT", 13, "TNR-Bold", space=4, color=ACCENT),
          para("on", 12, "TNR-Italic", space=14),
          para(TITLE.upper(), 21, "TNR-Bold", lead=27, space=8, color=NAVY),
          para("SignSense: A Deep Learning System with a FastAPI Service and a Streamlit Driving Simulation",
               12.5, "TNR-Italic", lead=17, space=26),
          para("Submitted in partial fulfilment of the requirements for the award of the degree of", 12,
               lead=16, space=8),
          para("BACHELOR OF TECHNOLOGY", 14, "TNR-Bold", space=4),
          para("in", 12, "TNR-Italic", space=4),
          para("COMPUTER SCIENCE AND ENGINEERING", 14, "TNR-Bold", space=30),
          para("Submitted by", 12, "TNR-Italic", space=4),
          para(STUDENT, 15, "TNR-Bold", space=3),
          para(f"Roll No.: {ROLL}", 12.5, space=3),
          para(SESSION, 12, space=24),
          para("Under the guidance of", 12, "TNR-Italic", space=4),
          para(GUIDE, 14, "TNR-Bold", space=3),
          para(DEPT, 12, space=70),
          Table([[""]], colWidths=[9 * cm], rowHeights=[2],
                style=[("LINEABOVE", (0, 0), (-1, -1), 1.2, NAVY), ("FONTNAME", (0, 0), (-1, -1), "TNR")]),
          Spacer(1, 12),
          para(DEPT.upper(), 12.5, "TNR-Bold", space=4, color=NAVY),
          para("BIRLA INSTITUTE OF TECHNOLOGY, MESRA", 16, "TNR-Bold", space=4, color=NAVY),
          para("RANCHI, JHARKHAND 835215, INDIA", 12, space=4),
          para("2026", 12, "TNR-Bold"))


def acknowledgement(s: Story):
    s.front_heading("ACKNOWLEDGEMENT")
    s.p(f"I would like to express my sincere gratitude to my project guide, <b>{GUIDE}</b>, {DEPT}, "
        f"{INSTITUTE}, for the guidance, encouragement and valuable feedback provided throughout this project. "
        "The suggestions on how to frame the problem, evaluate the model honestly and present the results "
        "helped shape this work at every stage.")
    s.p(f"I am thankful to the Head of the Department and all the faculty members of the {DEPT} for the "
        "knowledge they have shared over the course of the programme, and for creating an environment in which "
        "students are encouraged to build complete, working systems. I also thank Birla Institute of "
        "Technology, Mesra for providing the academic resources and facilities that made this project possible.")
    s.p("I acknowledge the creators of the German Traffic Sign Recognition Benchmark at the Institut für "
        "Neuroinformatik, Ruhr-Universität Bochum, for making the dataset publicly available, and the "
        "developers of the open-source projects used in this work, in particular PyTorch, torchvision, FastAPI, "
        "Streamlit and scikit-learn.")
    s.p("Finally, I am grateful to my family and friends for their constant support, patience and motivation "
        "during the course of this project.")
    s.add(Spacer(1, 1.6 * cm))
    sign = ParagraphStyle("sign", parent=S["body"], alignment=TA_RIGHT, spaceAfter=2)
    left = ParagraphStyle("left", parent=S["body"], alignment=TA_LEFT, spaceAfter=2)
    s.add(Table([[Paragraph("Place: Mesra, Ranchi", left), Paragraph(f"<b>{STUDENT}</b>", sign)],
                 [Paragraph("Date: ____________________", left), Paragraph(f"Roll No.: {ROLL}", sign)],
                 ["", Paragraph("B.Tech (CSE), 7th Semester", sign)]],
                colWidths=[s.width / 2] * 2,
                style=[("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                       ("FONTNAME", (0, 0), (-1, -1), "TNR")]))


def abstract(s: Story):
    s.add(PageBreak())
    s.front_heading("ABSTRACT")
    s.p("Traffic sign recognition is a core perception task for autonomous and driver-assisted vehicles: a car "
        "that cannot reliably read a speed limit, stop or yield sign cannot plan safe behaviour. This project "
        "presents <b>SignSense</b>, an end-to-end traffic sign recognition system written entirely in Python. "
        f"A compact VGG-style convolutional neural network, SignNet, with {N_PARAMS:,} parameters, is trained "
        "from scratch with PyTorch on the German Traffic Sign Recognition Benchmark (GTSRB), which contains "
        f"43 sign classes and {N_TRAIN_TOTAL + N_TEST:,} real-world images. Images are resized to 48×48 pixels, "
        "normalised and augmented with random affine and colour transformations; horizontal flipping is "
        "deliberately excluded because it changes the meaning of directional signs.")
    s.p("Trained for 15 epochs with the AdamW optimiser and a cosine learning-rate schedule on an Apple M3 GPU "
        f"(about eight minutes), the model reaches <b>{pct(TEST_ACC)} accuracy</b> and a <b>macro-averaged F1 "
        f"score of {MACRO_F1:.4f}</b> on the {N_TEST:,}-image official test set, on par with the 98.84% average "
        "human accuracy reported for the same benchmark. The trained model is served by a FastAPI backend that "
        "validates uploads, returns top-k predictions with confidence scores and maps every class to a driving "
        "action. A Streamlit frontend offers single-image and batch classification, a model report and a "
        "driving simulation in which a rule-based planner turns predictions into speed commands. Predictions "
        "below a confidence threshold of 0.6 are flagged, so the simulated vehicle keeps its current behaviour "
        "instead of acting on an uncertain sign.")
    s.p(f"Single-image inference takes a median of {API['median_ms']:.1f} ms, the service classified "
        f"{API['correct']} of {API['num_samples']} held-out demo images correctly (the single error was flagged "
        "as low confidence), and the code base is covered by 21 automated tests. This report describes the "
        "motivation, related work, dataset, system design, implementation, evaluation, limitations and future "
        "scope of the project.")
    s.add(Spacer(1, 8))
    s.p("<b>Keywords:</b> traffic sign recognition, convolutional neural network, GTSRB, autonomous vehicles, "
        "deep learning, PyTorch, FastAPI, Streamlit.")


def front_lists(s: Story):
    s.add(PageBreak())
    s.front_heading("TABLE OF CONTENTS", in_toc=False)
    s.add(TableOfContents(levelStyles=toc_styles(), dotsMinLevel=0, formatter=format_page))
    s.add(PageBreak())
    s.front_heading("LIST OF FIGURES")
    s.add(EntryList("LOF", levelStyles=[ParagraphStyle("lof", fontName="TNR", fontSize=11, leading=16)],
                    dotsMinLevel=0, formatter=format_page))
    s.add(PageBreak())
    s.front_heading("LIST OF TABLES")
    s.add(EntryList("LOT", levelStyles=[ParagraphStyle("lot", fontName="TNR", fontSize=11, leading=16)],
                    dotsMinLevel=0, formatter=format_page))
    s.add(PageBreak())
    s.front_heading("LIST OF ABBREVIATIONS")
    abbreviations = [
        ("ADAS", "Advanced Driver-Assistance System"), ("API", "Application Programming Interface"),
        ("BN", "Batch Normalisation"), ("CNN", "Convolutional Neural Network"),
        ("CORS", "Cross-Origin Resource Sharing"), ("CUDA", "Compute Unified Device Architecture (NVIDIA)"),
        ("FC", "Fully Connected (layer)"), ("FN / FP / TP", "False Negative / False Positive / True Positive"),
        ("GPU", "Graphics Processing Unit"), ("GTSDB", "German Traffic Sign Detection Benchmark"),
        ("GTSRB", "German Traffic Sign Recognition Benchmark"), ("HOG", "Histogram of Oriented Gradients"),
        ("HTTP", "Hypertext Transfer Protocol"), ("JSON", "JavaScript Object Notation"),
        ("LR", "Learning Rate"), ("MPS", "Metal Performance Shaders (Apple GPU backend for PyTorch)"),
        ("ReLU", "Rectified Linear Unit"), ("REST", "Representational State Transfer"),
        ("SVM", "Support Vector Machine"), ("UI", "User Interface"),
    ]
    rows = [[Paragraph(f"<b>{a}</b>", S["cell"]), Paragraph(b, S["cell"])] for a, b in abbreviations]
    t = Table(rows, colWidths=[3.6 * cm, s.width - 3.6 * cm])
    t.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5dbe2"))]))
    s.add(t)


def chapter_introduction(s: Story):
    s.chapter(1, "Introduction")
    s.h2("Background")
    s.p("Autonomous vehicles and advanced driver-assistance systems (ADAS) must perceive their surroundings, "
        "decide what to do and then act. Road signs are one of the most important inputs to that decision, "
        "because they encode the rules of the road at a particular place: the maximum legal speed, who has "
        "priority at a junction, where entry is forbidden and which hazards lie ahead. Human drivers read these "
        "signs almost without thinking, but a vehicle has to recognise them from camera images that may be "
        "small, blurred, badly lit, partly occluded or seen at an angle.")
    s.p("Traffic sign recognition (TSR) is therefore a standard component of the perception stack of a "
        "self-driving car, alongside lane, vehicle and pedestrian detection. Modern TSR systems are built with "
        "deep convolutional neural networks (CNNs), which learn visual features directly from labelled images "
        "instead of relying on hand-crafted colour and shape rules [1], [3], [13]. Public benchmarks such as the "
        "German Traffic Sign Recognition Benchmark (GTSRB) [1], [2] make it possible to compare methods on the "
        "same data, and they are widely used in research and in self-driving car simulations.")
    s.h2("Problem Statement")
    s.p("Given a cropped colour image of a traffic sign taken from a vehicle-mounted camera, the system must "
        "identify which of 43 sign classes it belongs to, report how confident it is, and translate the result "
        "into an action that a simulated autonomous vehicle can follow. The system must be accurate on unseen "
        "images, fast enough for interactive use, robust to invalid input, and usable by someone who is not a "
        "machine-learning specialist through a simple web interface.")
    s.h2("Objectives")
    s.p("The objectives of this project are:")
    s.bullets([
        "To build a reproducible data pipeline for the GTSRB dataset with a held-out validation split and "
        "augmentation that respects the meaning of traffic signs.",
        "To design and train a compact CNN that achieves at least 97% accuracy on the official GTSRB test set "
        "while remaining small enough to run in real time on a laptop GPU or CPU.",
        "To evaluate the model with accuracy, per-class F1 scores and a confusion matrix, and to understand "
        "where and why it fails.",
        "To expose the model through a documented REST API (FastAPI) that validates input, returns top-k "
        "predictions with confidence and maps each class to a vehicle action.",
        "To build an interactive Streamlit frontend with single-image, batch and camera input, a model report "
        "and a driving simulation that demonstrates how predictions drive vehicle behaviour.",
        "To make the system safe by design by flagging low-confidence predictions, and to cover it with "
        "automated tests.",
    ])
    s.h2("Scope of the Project")
    s.p("The project addresses the <i>classification</i> stage of traffic sign recognition: the input is an "
        "image that already contains a single sign, as produced by a detector or by manual cropping. Locating "
        "signs inside full dash-camera frames (detection) is outside the scope of this minimum viable product "
        "and is discussed as future work. The driving behaviour is simulated with a rule-based planner rather "
        "than a physical vehicle or a full driving simulator. The model is trained on German signs, which follow "
        "the Vienna Convention conventions shared by many countries.")
    s.h2("Organisation of the Report")
    s.p("Chapter 2 reviews classical and deep-learning approaches to traffic sign recognition. Chapter 3 lists "
        "the requirements and the hardware and software used. Chapter 4 describes the GTSRB dataset. Chapter 5 "
        "presents the overall system design, and Chapter 6 explains the methodology and implementation of the "
        "data pipeline, the SignNet model, the training procedure, the backend and the frontend. Chapter 7 "
        "reports and discusses the results, Chapter 8 covers testing, and Chapter 9 concludes with limitations "
        "and future scope. Appendix A contains key source code listings and Appendix B lists all 43 classes.")


def chapter_literature(s: Story):
    s.chapter(2, "Literature Review")
    s.h2("Classical Approaches")
    s.p("Early traffic sign recognition systems relied on the strong visual design of signs. A typical pipeline "
        "segmented candidate regions using colour thresholds (red, blue, yellow) in colour spaces such as HSV, "
        "filtered them by shape (circles, triangles, octagons), and classified them with hand-crafted features. "
        "Histograms of Oriented Gradients (HOG) [14] combined with classifiers such as Support Vector Machines "
        "(SVM) or linear discriminant analysis were among the strongest of these methods and served as baselines "
        "in the GTSRB competition [1]. These approaches are fast and interpretable, but they are sensitive to "
        "lighting, faded paint, motion blur and occlusion, and the features have to be engineered by hand.")
    s.h2("The GTSRB Benchmark")
    s.p("Stallkamp et al. introduced the German Traffic Sign Recognition Benchmark as a multi-class "
        "classification competition at IJCNN 2011 [1]. It contains more than 50,000 images of 43 classes "
        "captured from video on German roads, with large variation in scale, illumination and viewpoint. In a "
        "follow-up study [2] the authors measured human performance on the same test set and reported an "
        "average human accuracy of 98.84%, which has since become a reference point for automatic methods. A "
        "companion benchmark for detection, GTSDB, was released later [20].")
    s.h2("Deep Learning Approaches")
    s.p("Convolutional neural networks, introduced for document recognition by LeCun et al. [13], dominated "
        "the GTSRB competition. Cireşan et al. won with a multi-column deep neural network, a committee of CNNs "
        "trained on differently pre-processed images, reaching 99.46% accuracy and exceeding human performance "
        "[3]. Sermanet and LeCun used a multi-scale CNN in which features from several layers are fed to the "
        "classifier, achieving 98.31% in the competition [4], [2]. Later work added spatial transformer "
        "networks [11], which learn to undo rotations and scale changes inside the network; Arcos-García et al. "
        "reported 99.71% with such a model [10].")
    s.p("Several general-purpose techniques are central to these results and are used in this project. Deep "
        "stacks of small 3×3 convolutions, popularised by the VGG networks [5], give large receptive fields with "
        "few parameters. Batch normalisation [6] stabilises and speeds up training, and dropout [7] reduces "
        "overfitting. AdamW [8] decouples weight decay from the adaptive gradient update, and cosine annealing of "
        "the learning rate [9] provides a smooth schedule that usually improves final accuracy.")
    s.table(["Method", "Reference", "GTSRB test accuracy"],
            [["Average human performance", "Stallkamp et al., 2012 [2]", "98.84%"],
             ["Multi-scale CNN", "Sermanet and LeCun, 2011 [4]", "98.31%"],
             ["Multi-column DNN (committee of CNNs)", "Cireşan et al., 2012 [3]", "99.46%"],
             ["CNN with spatial transformers", "Arcos-García et al., 2018 [10]", "99.71%"],
             [f"SignNet, single model, {N_PARAMS / 1e6:.2f} M parameters", "This work", pct(TEST_ACC)]],
            [5, 4.2, 2.6], "Reported results on the GTSRB test set", ["l", "l", "c"], bold_rows=(4,))
    s.h2("Research Gap and Motivation")
    s.p("Most published work reports a single accuracy number for a model evaluated offline. For a student "
        "project aimed at self-driving car simulations, two further questions matter: how the model behaves when "
        "it is wrong, and how its output can be used by the rest of a vehicle software stack. This project "
        "therefore focuses on a compact model that is trained in minutes on a laptop, on a careful error "
        "analysis, and on wrapping the model in a service with explicit confidence handling and a simple planner, "
        "so the complete path from camera image to vehicle decision can be demonstrated end to end.")


def chapter_requirements(s: Story):
    s.chapter(3, "Requirements and Tools")
    s.h2("Functional Requirements")
    s.bullets([
        "<b>FR1.</b> Download and prepare GTSRB automatically, with a reproducible training/validation split.",
        "<b>FR2.</b> Train a CNN classifier and save the best checkpoint with its preprocessing metadata.",
        "<b>FR3.</b> Evaluate the model on the official test set and produce metrics and plots.",
        "<b>FR4.</b> Classify a single image or a batch of up to 32 images through an HTTP API, returning the "
        "label, confidence, top-k alternatives, sign category and recommended vehicle action.",
        "<b>FR5.</b> Provide a web interface for upload, test samples and camera input, batch results, a driving "
        "simulation and a model report.",
        "<b>FR6.</b> Flag predictions below a configurable confidence threshold as uncertain.",
    ])
    s.h2("Non-Functional Requirements")
    s.bullets([
        "<b>Performance:</b> single-image inference well below 50 ms on a laptop GPU, so the simulation "
        "runs smoothly.",
        "<b>Robustness:</b> invalid, empty or oversized uploads must be rejected with clear error messages "
        "instead of crashing the service.",
        "<b>Portability:</b> the same code must run on CUDA, Apple MPS or CPU, selected automatically or through "
        f"the {C('DEVICE')} environment variable.",
        "<b>Reproducibility:</b> fixed random seeds, pinned dependency ranges and saved training history.",
        "<b>Maintainability:</b> a small, modular code base with separate packages for machine learning, the "
        "backend and the frontend, covered by automated tests.",
    ])
    s.h2("Hardware and Software Environment")
    s.table(["Component", "Details"],
            [["Machine", "Apple MacBook with Apple M3 chip, 8 GB unified memory"],
             ["Accelerator", "Apple GPU through the PyTorch MPS backend (CUDA and CPU also supported)"],
             ["Operating system", "macOS"],
             ["Language", "Python 3.13"],
             ["Deep learning", "PyTorch 2.15, torchvision 0.29"],
             ["Data and metrics", "NumPy 2.5, pandas 3.1, scikit-learn 1.9, Matplotlib 3.11, Pillow 13"],
             ["Backend", "FastAPI 0.162, Uvicorn 0.54, Pydantic 2.14, python-multipart"],
             ["Frontend", "Streamlit 1.67, Requests 2.34"],
             ["Testing", "pytest 9.2 with FastAPI TestClient"],
             ["Version control", f"Git and GitHub ({REPO})"]],
            [3, 9], "Hardware and software used in the project")


def chapter_dataset(s: Story):
    s.chapter(4, "Dataset")
    s.h2("The German Traffic Sign Recognition Benchmark")
    s.p("GTSRB [1] consists of images of 43 classes of German road signs extracted from video recorded while "
        "driving on real roads. Each image contains exactly one sign with a small margin around it, and image "
        "sizes range from about 15×15 to 250×250 pixels. The classes cover speed limits, prohibitions, warnings, "
        "mandatory directions, priority signs and the ends of restrictions. The dataset is loaded with "
        f"{C('torchvision.datasets.GTSRB')}, which downloads and verifies the archives on first use. The "
        f"training archive distributed through torchvision contains {N_TRAIN_TOTAL:,} images, and the official "
        f"test set contains {N_TEST:,} images.")
    s.p(f"Figure {s.next_figure()} shows one test image for each class; the class names are listed in "
        "Appendix B. The images illustrate the main difficulties of the task: low resolution, motion blur, very "
        "dark or over-exposed frames, and pairs of classes that differ only in a small pictogram or a digit.")
    s.figure(FIG / "sample_grid.png", "One test-set sample for each of the 43 GTSRB classes (class IDs above "
             "each image)", 15.5)
    s.h2("Data Splits")
    s.p("The official training set is divided at random into training and validation subsets in a 90:10 "
        "ratio with a fixed seed (42), and the official test set is used only for the final evaluation.")
    s.table(["Split", "Images", "Share", "Purpose"],
            [["Training", f"{N_TRAIN:,}", pct(N_TRAIN / (N_TRAIN_TOTAL + N_TEST), 1), "Fitting model weights"],
             ["Validation", f"{N_VAL:,}", pct(N_VAL / (N_TRAIN_TOTAL + N_TEST), 1),
              "Model selection (best epoch)"],
             ["Test (official)", f"{N_TEST:,}", pct(N_TEST / (N_TRAIN_TOTAL + N_TEST), 1),
              "Final, unbiased evaluation"],
             ["Total", f"{N_TRAIN_TOTAL + N_TEST:,}", "100%", ""]],
            [2.4, 1.6, 1.4, 4.5], "Dataset splits used in this project", ["l", "r", "r", "l"], bold_rows=(3,))
    s.h2("Class Imbalance")
    s.p(f"The classes are strongly imbalanced (Figure {s.next_figure()}). The largest classes have "
        f"{MAX_COUNT:,} training images (classes {id_list(MAX_IDS)}: "
        f"{', '.join(CLASS_NAMES[i] for i in MAX_IDS)}), while the smallest have only {MIN_COUNT} (classes "
        f"{id_list(MIN_IDS)}: {', '.join(CLASS_NAMES[i] for i in MIN_IDS)}), a ratio of "
        f"{MAX_COUNT // MIN_COUNT}:1. Common signs such as speed limits and priority signs dominate, while many "
        "warning signs are rare. The test set has a similar distribution, so overall accuracy is dominated by "
        "the frequent classes; this is why the macro-averaged F1 score, which weights every class equally, is "
        "reported alongside accuracy.")
    s.figure(FIG / "class_distribution.png", "Number of training images per class, coloured by sign category",
             15.5)
    s.h2("A Note on Validation Leakage")
    s.p("GTSRB training images are organised in <i>tracks</i>: about 30 consecutive frames of the same physical "
        "sign as the vehicle approaches it. A random split places frames of the same track in both the training "
        "and the validation subset, so the validation images are near-duplicates of training images. The "
        "validation accuracy is therefore optimistic and is used only to choose the best epoch. All reported "
        "performance figures come from the official test set, which was recorded separately.")


def chapter_design(s: Story):
    s.chapter(5, "System Design")
    s.h2("Overall Architecture")
    s.p(f"The system has two halves (Figure {s.next_figure()}). The <b>offline</b> half, in the {C('ml')} "
        "package, prepares the data, trains SignNet and evaluates it, producing a checkpoint file, metrics and "
        "plots. The <b>online</b> half serves the trained model: a FastAPI backend loads the checkpoint once at "
        "start-up and answers HTTP requests, and a Streamlit frontend sends images to the API and presents the "
        "results. The two halves share only the checkpoint and a small amount of configuration, so the model can "
        "be retrained without touching the service code.")
    s.figure(architecture_diagram(s.width), "System architecture of SignSense", 15.5)
    s.h2("Project Structure", space_needed_cm=10)
    s.p("The repository is organised into three Python packages and supporting folders:")
    s.code([
        "sign_sense/",
        "|-- ml/                    # data, model, training and evaluation",
        "|   |-- config.py          # paths, image size, normalisation, 43 class names",
        "|   |-- data.py            # transforms and data loaders",
        "|   |-- model.py           # SignNet CNN",
        "|   |-- train.py           # training loop, checkpointing, curves",
        "|   |-- evaluate.py        # test metrics and confusion matrix",
        "|   |-- signs.py           # class -> category, action, speed limit",
        "|   `-- export_samples.py  # stratified demo images",
        "|-- backend/",
        "|   |-- main.py            # FastAPI app and endpoints",
        "|   |-- inference.py       # SignClassifier (load, preprocess, predict)",
        "|   `-- schemas.py         # Pydantic response models",
        "|-- frontend/app.py        # Streamlit user interface and planner",
        "|-- tests/                 # pytest suite (API and ML)",
        "|-- models/sign_cnn.pt     # trained checkpoint",
        "|-- reports/               # metrics.json, history.json, plots",
        "`-- data/samples/          # 86 demo images, 2 per class",
    ], first_line=None, font_size=8.6)
    s.h2("Life of a Prediction Request")
    s.numbered([
        "The user uploads an image, picks a test sample or takes a camera frame in the Streamlit app.",
        f"The frontend sends the image as a multipart form to {C('POST /predict')} with the requested number "
        "of alternatives (top-k).",
        "FastAPI checks that the file is not empty, is at most 5 MB and can be decoded as an image; otherwise it "
        "returns HTTP 400 or 413 with an explanation.",
        "The SignClassifier resizes the image to 48×48, normalises it with the training statistics and runs the "
        "CNN in a worker thread, so the event loop is never blocked.",
        "A softmax turns the 43 logits into probabilities; the top-k classes and their confidences are returned.",
        f"The backend adds the sign semantics from {C('ml/signs.py')} (category, action and speed limit) and an "
        f"{C('is_confident')} flag (confidence of at least 0.6), and returns a JSON document.",
        "The frontend displays the result and, in the driving simulation, passes it to the planner, which "
        "updates the speed limit and the commanded speed.",
    ])
    s.h2("From Sign Class to Vehicle Action")
    s.p(f"Each of the 43 classes is mapped to one of nine categories (Table {s.next_table()}). The category "
        "determines the action text returned by the API and how the planner in the simulation reacts. Speed limit "
        "signs set a new limit; the two end-of-limit signs (classes 6 and 32) restore an open-road limit of "
        "100 km/h; stop and no-entry signs bring the vehicle to a halt; and yield and warning signs scale the "
        "speed down.")
    s.table(["Category", "Class IDs", "Action returned by the API", "Planner effect"],
            [["speed_limit", "0–5, 7, 8", "Set target speed to N km/h", "limit = N"],
             ["end_of_restriction", "6, 32, 41, 42", "Restore default behaviour", "6, 32: limit = 100 km/h"],
             ["stop", "14", "Come to a full stop, proceed when clear", "speed = 0"],
             ["no_entry", "15, 17", "Do not enter, plan an alternative route", "speed = 0"],
             ["yield", "13", "Slow down and give way", "speed = 0.5 × limit"],
             ["warning", "18–31", "Hazard ahead, reduce speed", "speed = 0.7 × limit"],
             ["priority", "11, 12", "Maintain speed, you have right of way", "speed = limit"],
             ["prohibitory", "9, 10, 16", "Restriction active", "speed = limit"],
             ["mandatory", "33–40", "Follow mandatory direction", "speed = limit"],
             ["(confidence < 0.6)", "any", "Keep current behaviour until confirmed", "unchanged"]],
            [2.5, 2.0, 4.6, 2.9], "Sign categories, actions and planner rules", ["l", "c", "l", "l"])


def chapter_method(s: Story):
    s.chapter(6, "Methodology and Implementation")
    s.h2("Preprocessing")
    s.p("Every image is converted to RGB, resized to 48×48 pixels and converted to a tensor with values in "
        "[0, 1]. Each channel is then standardised with the per-channel mean (0.3403, 0.3121, 0.3214) and "
        "standard deviation (0.2724, 0.2608, 0.2669) of the GTSRB training set. The same values are stored "
        "inside the checkpoint, so the API always applies exactly the preprocessing the model was trained with. "
        "A resolution of 48×48 keeps digits and pictograms legible while keeping the network small.")
    s.h2("Data Augmentation")
    s.p(f"Augmentation is applied only to training images (Table {s.next_table()}). Validation and test images "
        "receive the deterministic preprocessing only; the training and validation subsets are built from two "
        "separate views of the dataset so that validation images are never augmented.")
    s.table(["Transformation", "Parameters", "Reason"],
            [["RandomAffine", "rotation ±10°, translation ±10%, scale 0.9–1.1, shear 5°",
              "Camera angle, distance and imperfect crops"],
             ["ColorJitter", "brightness 0.4, contrast 0.4, saturation 0.3",
              "Day/night lighting, shadows, faded paint"],
             ["Horizontal flip (not used)", "n/a",
              "Would turn ‘Keep left’ into ‘Keep right’ and ‘Dangerous curve to the left’ into ‘to the right’, "
              "silently corrupting labels"]],
            [3, 4.6, 5], "Training-time data augmentation", ["l", "l", "l"])
    s.h2("SignNet Architecture")
    s.p(f"SignNet is a VGG-style CNN [5] with three convolutional blocks followed by a two-layer classifier "
        f"(Figure {s.next_figure()}). Each block applies two 3×3 convolutions with padding 1, each followed by "
        "batch normalisation [6] and ReLU, then a 2×2 max-pooling layer that halves the spatial resolution and a "
        "dropout layer [7]. The number of filters doubles from block to block (32, 64, 128) while the resolution "
        "halves (48, 24, 12, 6), and the dropout rate increases with depth (0.1, 0.2, 0.3). Convolutions omit the "
        "bias term because the following batch normalisation has its own shift parameter.")
    s.figure(signnet_diagram(s.width), "SignNet architecture: feature maps after each stage", 15.5)
    s.p(f"The layer-by-layer shapes and parameter counts in Table {s.next_table()} were computed directly from "
        f"the PyTorch model. In total the network has {N_PARAMS:,} trainable parameters, and the checkpoint "
        f"occupies {MODEL_MB:.1f} MB. Four fifths of the parameters sit in the first fully connected layer, "
        "which maps the 4,608 flattened features to 256 hidden units; this is why it receives the strongest "
        "dropout (0.5).")
    fc1, fc2 = block_params("classifier.1"), block_params("classifier.4")
    s.table(["Stage", "Layers", "Output shape", "Parameters"],
            [["Input", "RGB image, resized and normalised", "3 × 48 × 48", "0"],
             ["Block 1", "2 × (Conv 3×3, 32 filters, BN, ReLU), MaxPool 2×2, Dropout 0.1", "32 × 24 × 24",
              f"{block_params('features.0.'):,}"],
             ["Block 2", "2 × (Conv 3×3, 64 filters, BN, ReLU), MaxPool 2×2, Dropout 0.2", "64 × 12 × 12",
              f"{block_params('features.1.'):,}"],
             ["Block 3", "2 × (Conv 3×3, 128 filters, BN, ReLU), MaxPool 2×2, Dropout 0.3", "128 × 6 × 6",
              f"{block_params('features.2.'):,}"],
             ["Flatten", "Reshape feature maps to a vector", "4,608", "0"],
             ["FC 1", "Linear 4,608 → 256, ReLU, Dropout 0.5", "256", f"{fc1:,}"],
             ["FC 2", "Linear 256 → 43 (logits); softmax at inference", "43", f"{fc2:,}"],
             ["Total", "", "", f"{N_PARAMS:,}"]],
            [1.6, 6.2, 2.3, 2.0], "SignNet layer summary", ["l", "l", "c", "r"], bold_rows=(7,))
    s.h2("Training Procedure")
    s.p("The network is trained from random initialisation by minimising the cross-entropy loss between the "
        "predicted class distribution and the true label. For a mini-batch of N images with true classes "
        "y<sub>i</sub>, where p<sub>i</sub>(c) is the predicted probability of class c for image i, the loss is")
    s.equation("L = −(1/N) Σ<sub>i</sub> log p<sub>i</sub>(y<sub>i</sub>)")
    s.p("AdamW [8] updates the weights with decoupled weight decay, and the learning rate follows a cosine "
        "curve from its initial value towards zero over the 15 epochs [9]:")
    s.equation("η<sub>t</sub> = η<sub>0</sub> · ½ (1 + cos(π t / T)),  with η<sub>0</sub> = 10<super>−3</super> "
               "and T = 15")
    s.p(f"After every epoch the model is evaluated on the validation subset, and the checkpoint is overwritten "
        "only when validation accuracy improves. The checkpoint stores the weights together with the class "
        "names, image size and normalisation statistics, and it is loaded with "
        f"{C('torch.load(weights_only=True)')} so that no arbitrary Python objects are unpickled. Table "
        f"{s.next_table()} summarises the configuration.")
    s.table(["Setting", "Value"],
            [["Loss function", "Cross-entropy"],
             ["Optimiser", "AdamW, initial learning rate 1 × 10<super>−3</super>, weight decay 1 × 10<super>−4</super>"],
             ["Learning-rate schedule", "Cosine annealing over 15 epochs (T<sub>max</sub> = 15)"],
             ["Batch size / epochs", "128 / 15"],
             ["Random seed", "42 (data split and weight initialisation)"],
             ["Model selection", "Best validation accuracy"],
             ["Hardware", "Apple M3 GPU (MPS), two data-loader worker processes"],
             ["Training time", "About 8 minutes (30–40 s per epoch)"]],
            [4, 8], "Training configuration")
    s.h2("Evaluation Metrics")
    s.p("Besides overall accuracy, the model is evaluated per class. For class c, with TP<sub>c</sub>, "
        "FP<sub>c</sub> and FN<sub>c</sub> the true positives, false positives and false negatives on the test "
        "set:")
    s.equation("Precision<sub>c</sub> = TP<sub>c</sub> / (TP<sub>c</sub> + FP<sub>c</sub>),     "
               "Recall<sub>c</sub> = TP<sub>c</sub> / (TP<sub>c</sub> + FN<sub>c</sub>)")
    s.equation("F1<sub>c</sub> = 2 · Precision<sub>c</sub> · Recall<sub>c</sub> / (Precision<sub>c</sub> + "
               "Recall<sub>c</sub>),     Macro F1 = (1/43) Σ<sub>c</sub> F1<sub>c</sub>")
    s.p("The macro F1 score gives every class the same weight regardless of its size, so it exposes weaknesses "
        "on rare classes that overall accuracy would hide. A row-normalised confusion matrix shows which classes "
        "are mistaken for which.")
    s.h2("Backend Implementation")
    s.p(f"The backend ({C('backend/main.py')}) is a FastAPI application. A lifespan handler creates a single "
        f"{C('SignClassifier')} when the server starts; it selects the device (CUDA, MPS or CPU), loads the "
        "checkpoint, builds the preprocessing transform and runs one warm-up prediction so the first real "
        f"request is not slowed down by GPU initialisation. The checkpoint path can be overridden with the "
        f"{C('MODEL_PATH')} environment variable and the threshold with {C('CONFIDENCE_THRESHOLD')}. Because model "
        f"inference is CPU/GPU-bound, it is executed with {C('run_in_threadpool')}, keeping the asynchronous event "
        f"loop free to accept other requests. All responses are described by Pydantic models, which also "
        f"generate interactive OpenAPI documentation at {C('/docs')}. The endpoints are listed in Table "
        f"{s.next_table()}.")
    s.table(["Method and path", "Purpose", "Response model"],
            [["GET /health", "Liveness check: status, model loaded, device", "HealthResponse"],
             ["GET /classes", "All 43 classes with their category", "list[ClassInfo]"],
             ["GET /model/info", "Architecture, parameter count, image size, validation accuracy, test metrics",
              "ModelInfo"],
             ["POST /predict?top_k=1..10", "Classify one image (default top-5)", "Prediction"],
             ["POST /predict/batch?top_k=1..10", "Classify up to 32 images in one forward pass (default top-3)",
              "BatchPrediction"]],
            [4.2, 6.4, 2.8], "REST API endpoints", ["l", "l", "l"])
    s.p("Invalid requests are rejected early: an empty upload or a file that cannot be decoded returns HTTP 400, "
        "a file larger than 5 MB returns HTTP 413, a batch of more than 32 images returns HTTP 400, and an "
        f"out-of-range {C('top_k')} returns HTTP 422 through FastAPI's query validation. A typical response for "
        "the Stop-sign sample, abridged to the top three classes, is shown below.")
    s.code([
        "{",
        '  "filename": "028_class14.png",',
        '  "class_id": 14,',
        '  "label": "Stop",',
        '  "confidence": 0.9992,',
        '  "is_confident": true,',
        '  "top_k": [',
        '    {"class_id": 14, "label": "Stop", "confidence": 0.9992},',
        '    {"class_id": 18, "label": "General caution", "confidence": 0.0004},',
        '    {"class_id": 17, "label": "No entry", "confidence": 0.0003}',
        "  ],",
        '  "inference_ms": 3.4,',
        '  "category": "stop",',
        '  "action": "Come to a full stop at the line, proceed when clear",',
        '  "speed_limit_kmh": null',
        "}",
    ], first_line=None, font_size=8.6)
    s.h2("Frontend Implementation")
    s.p(f"The frontend ({C('frontend/app.py')}) is a Streamlit application. The sidebar holds the API URL and "
        "the number of alternatives to show, and it checks the API's health on every run; if the API cannot be "
        "reached, the app shows the command needed to start it instead of failing with a stack trace. The main "
        "area has four tabs:")
    s.bullets([
        "<b>Classify</b>: an image from an upload, the bundled test samples or the webcam is sent to the API. The "
        "app shows the predicted sign, the confidence, the vehicle action and a top-k table with confidence bars, "
        "and for test samples it also shows the ground truth. Uncertain predictions are shown as a warning and "
        "the action is replaced with ‘keep current behaviour until the sign is confirmed’.",
        "<b>Batch</b>: up to 32 uploaded images, or 12 random test samples, are classified in one request and "
        "shown as an image grid and a results table with correctness.",
        "<b>Drive simulation</b>: a simulated car passes a random sequence of 5 to 40 signs. Each frame is "
        "classified, the rule-based planner updates the speed limit and commanded speed, and at the end the app "
        "shows route accuracy, a speed/limit chart and a log of every decision.",
        "<b>Model report</b>: architecture, parameter count, validation and test accuracy, training curves, "
        "the confusion matrix and the hardest classes, all fetched from the API.",
    ])
    s.p(f"The planner ({C('plan()')}, Appendix A.6) starts at 50 km/h. A confident speed-limit sign sets a new "
        "limit and the end-of-limit signs restore 100 km/h. The commanded speed is the limit multiplied by a "
        "category factor: 0 for stop and no-entry, 0.5 for yield, 0.7 for warnings and 1 otherwise. An uncertain "
        "prediction leaves both limit and speed unchanged, which is the safe default for a real vehicle.")


def chapter_results(s: Story, screens: dict):
    s.chapter(7, "Results and Discussion")
    s.h2("Training Behaviour")
    s.p(f"Figure {s.next_figure()} shows the loss and accuracy curves over the 15 epochs and Table "
        f"{s.next_table()} lists the values. The model learns quickly: validation accuracy passes 96% after four "
        f"epochs and 99% after seven, and reaches its best value of {pct(BEST_VAL)} at epoch {BEST_EPOCH}, which "
        "is the checkpoint that is deployed. The cosine schedule makes the last epochs smooth, with almost no "
        "fluctuation.")
    s.figure(FIG / "training_curves.png", "Training and validation loss (a) and accuracy (b) per epoch", 15.5)
    rows = []
    for i in range(len(HISTORY["train_loss"])):
        rows.append([str(i + 1), f"{HISTORY['train_loss'][i]:.4f}", pct(HISTORY["train_acc"][i]),
                     f"{HISTORY['val_loss'][i]:.4f}", pct(HISTORY["val_acc"][i])])
    s.table(["Epoch", "Training loss", "Training accuracy", "Validation loss", "Validation accuracy"],
            rows, [1.2, 2, 2.2, 2, 2.3], f"Per-epoch training history (best epoch {BEST_EPOCH} in bold)",
            ["c", "r", "r", "r", "r"], font_size=9.5, bold_rows=(BEST_EPOCH - 1,))
    s.p("Validation accuracy is higher than training accuracy throughout. This is expected for two reasons: "
        "the training images are augmented and pass through active dropout layers, which makes them harder, "
        "while validation images are clean and evaluated with dropout disabled; and the validation subset "
        "shares tracks with the training subset (Section 4.4). The gap is therefore not a sign of a problem, "
        "but it is the reason the test set is used for all final numbers.")
    s.h2("Test-Set Performance")
    s.p(f"On the {N_TEST:,} images of the official GTSRB test set, which the model never saw during training "
        f"or model selection, SignNet classifies {N_CORRECT:,} images correctly (Table {s.next_table()}).")
    s.table(["Metric", "Value"],
            [["Test accuracy", f"{pct(TEST_ACC)} ({N_CORRECT:,} of {N_TEST:,} images)"],
             ["Misclassified test images", f"{N_WRONG}"],
             ["Macro-averaged F1 score", f"{MACRO_F1:.4f}"],
             ["Classes with F1 = 1.000", f"{len(PERFECT)} (classes {id_list(PERFECT)})"],
             ["Classes with F1 ≥ 0.98", f"{ABOVE_98} of 43"],
             ["Best validation accuracy", f"{pct(BEST_VAL)} (epoch {BEST_EPOCH})"],
             ["Parameters / checkpoint size", f"{N_PARAMS:,} / {MODEL_MB:.1f} MB"]],
            [5, 7], "Summary of test-set results", ["l", "l"])
    s.p("The accuracy of 98.80% exceeds the objective of 97% and is on par with the average human accuracy of "
        "98.84% reported on the same test set [2]. It is below the best published results of 99.46% to 99.71% "
        "[3], [10], which rely on committees of networks or spatial transformer modules; SignNet is a single, "
        "plain network trained in about eight minutes on a laptop.")
    s.h2("Per-Class Analysis")
    s.p(f"Figure {s.next_figure()} shows the F1 score of every class. Most classes are above 0.98, and only five "
        f"fall below 0.96. These five classes, listed in Table {s.next_table()}, are all among the rarer classes "
        f"in the training data, with between {min(TRAIN_COUNTS[i] for i in WORST)} and "
        f"{max(TRAIN_COUNTS[i] for i in WORST)} training images each compared with a median of "
        f"{sorted(TRAIN_COUNTS)[21]}. Class imbalance is therefore the most likely cause of the remaining errors.")
    s.figure(FIG / "per_class_f1.png", "Per-class F1 score on the test set (red: F1 below 0.96)", 15.5)
    weak_reasons = {
        19: "Absorbs Double curve errors (same triangle, similar curve)",
        21: "Predicted as Dangerous curve to the left",
        22: "Predicted as Road work (dark pictogram at the bottom)",
        40: "Confused with other blue circular mandatory signs",
        41: "Predicted as No passing (same sign without the stripe)",
    }
    s.table(["ID", "Class", "Test F1", "Training images", "Main confusion"],
            [[str(i), CLASS_NAMES[i], f"{F1_BY_ID[i]:.3f}", str(TRAIN_COUNTS[i]), weak_reasons.get(i, "")]
             for i in WORST],
            [0.7, 3.6, 1.3, 1.6, 4.6], "The five weakest classes", ["c", "l", "c", "c", "l"])
    s.h2("Confusion Analysis")
    s.p(f"The normalised confusion matrix in Figure {s.next_figure()} is almost perfectly diagonal: for every "
        "class, the large majority of test images are predicted correctly, and off-diagonal entries are sparse. "
        f"Table {s.next_table()} lists the most frequent individual confusions.")
    s.figure(REPORTS_DIR / "confusion_matrix.png", "Row-normalised confusion matrix on the GTSRB test set "
             "(rows: true class, columns: predicted class)", 14.5)
    reasons = {
        (8, 5): "Same red circle; at 48×48 the digits ‘120’ and ‘80’ blur together",
        (21, 19): "Same warning triangle; the pictograms differ by a single bend",
        (22, 25): "Small dark pictogram at the bottom of a warning triangle",
        (12, 13): "Very dark or blurred frames in which the shape cue is lost",
        (6, 38): "Both have a diagonal element across a round sign",
        (18, 26): "Both show a vertical pictogram in a warning triangle",
        (38, 40): "Both are blue circular mandatory signs with white arrows",
        (30, 11): "Small, detailed central symbols (snowflake vs. junction)",
    }
    s.table(["True class", "Predicted as", "Images", "Likely reason"],
            [[f"{c['true']} ({c['true_id']})", f"{c['pred']} ({c['pred_id']})", str(c["count"]),
              reasons.get((c["true_id"], c["pred_id"]), "")] for c in STATS["top_confusions"]],
            [3.4, 3.4, 1.2, 4.8], "Most frequent confusions on the test set", ["l", "l", "c", "l"],
            font_size=9.5)
    s.p("The most common error, 120 km/h read as 80 km/h, is also the most safety-relevant, since the vehicle "
        "would drive slower than allowed rather than faster; the reverse error would be more dangerous. Most "
        "of the other confusions are between signs with the same shape and colour that differ only in a small "
        "pictogram, where the 48×48 input leaves very few pixels for the distinguishing detail. A higher input "
        "resolution and class-balanced sampling are the most promising remedies (Section 9.3).")
    s.h2("API and Inference Performance")
    s.p(f"The 86 demo images exported to {C('data/samples')} (two per class, drawn from the test set with a "
        "fixed seed) were sent one at a time through the API with the trained model on the Apple M3 GPU "
        f"(Table {s.next_table()}). The timings are the model time measured inside the endpoint, including "
        "preprocessing, and exclude network transfer.")
    miss = API["misses"][0] if API["misses"] else None
    s.table(["Measurement", "Result"],
            [["Demo images classified correctly", f"{API['correct']} of {API['num_samples']}"],
             ["Misclassified image", f"{miss['file']}: {miss['truth']} predicted as {miss['pred']} at "
              f"{miss['confidence']:.1%} confidence, flagged as uncertain" if miss else "none"],
             ["Single-image latency (median)", f"{API['median_ms']:.2f} ms"],
             ["Single-image latency (95th percentile)", f"{API['p95_ms']:.2f} ms"],
             ["Single-image latency (maximum)", f"{API['max_ms']:.2f} ms"],
             ["Batch of 12 images (total)", f"{API['batch12_total_ms']:.0f} ms"],
             ["Model load and warm-up at start-up", "a few seconds, once per server start"]],
            [5, 7], "API accuracy and latency on the demo images", ["l", "l"])
    s.p("A median latency of a few milliseconds is far below the 50 ms target, so the model could keep up with "
        "a camera running at well over 30 frames per second. The first request after the server has been idle "
        "can take tens to a few hundred milliseconds while the GPU wakes up, which is visible in some of the "
        "screenshots in the next section; subsequent requests return to the figures above. The one "
        "misclassified demo image shows the "
        "confidence threshold working as intended: the model hesitated between ‘No passing’ (53%) and ‘Vehicles "
        "over 3.5 t prohibited’ (41%), the API marked the prediction as not confident, and the frontend told "
        "the vehicle to keep its current behaviour instead of acting on the wrong sign.")
    s.h2("Application Demonstration")
    fig = {name: f"{s.chapter_label}.{s.figures + i}"
           for i, name in enumerate(["classify", "low", "batch", "drive", "report"], 1)}
    s.p(f"Figures {fig['classify']} to {fig['report']} show the running application. In the Classify tab "
        f"(Figure {fig['classify']}) the Stop-sign sample is recognised with 99.9% confidence, and the vehicle "
        f"action and top-5 alternatives are shown. Figure {fig['low']} shows the low-confidence case discussed "
        "above.")
    s.figure(screens["classify"], "Classify tab: a Stop sign recognised with 99.9% confidence", 14.5, 12)
    s.figure(screens["low_confidence"], "Low-confidence prediction: the action is withheld until the sign is "
             "confirmed", 14.5, 12)
    s.p(f"The Batch tab (Figure {fig['batch']}) classifies 12 random samples in a single request. The confidence "
        "under each image makes the harder cases stand out, such as the ‘End of no passing’ sample at 58%.")
    s.figure(screens["batch"], "Batch tab: 12 random test samples classified in one request", 14.5, 14)
    s.p(f"In the drive simulation (Figure {fig['drive']}) the planner reacts to each sign: the speed drops to "
        "35 km/h (0.7 × 50) at a warning sign, to zero at a no-entry sign, and the limit rises to 100 km/h at "
        "step 9. All 12 signs on the route shown were classified correctly; in an earlier recorded run, 11 of "
        "12 were correct (92%).")
    s.figure(screens["drive"], "Drive simulation: the planner's speed and limit over a 12-sign route", 14.5, 14)
    s.figure(screens["model_report"], "Model report tab with headline metrics, curves and the confusion matrix",
             14.5, 14)
    s.h2("Discussion")
    s.p("The results show that a compact, single CNN trained from scratch is sufficient for accurate traffic sign "
        "classification on GTSRB, reaching human-level accuracy with a model that trains in minutes and runs in "
        "milliseconds. The main value of the project, however, lies in the complete pipeline: careful "
        "augmentation choices that respect sign semantics, an evaluation that looks beyond accuracy, and a "
        "service that tells the rest of the system when not to trust the model. The error analysis also makes "
        "clear where the model would need to improve before it could be relied upon: rare warning signs with "
        "small pictograms, and speed limits whose digits are hard to read at low resolution.")


def chapter_testing(s: Story):
    s.chapter(8, "Testing and Validation")
    s.h2("Automated Tests")
    s.p(f"The project has 21 automated tests written with pytest, located in {C('tests/')}. API tests use "
        f"FastAPI's {C('TestClient')} with a freshly initialised, untrained SignNet saved to a temporary "
        "checkpoint and forced onto the CPU, so they run in about three seconds without the real model or a GPU. "
        f"Machine-learning tests check the model and the sign semantics, and two of them load the trained "
        f"checkpoint to confirm that known demo images are still classified correctly. Table {s.next_table()} "
        "lists the tests; all 21 pass.")
    s.table(["Test", "What it verifies", "Cases"],
            [["test_health", "Health endpoint reports status ok, model loaded and the device", "1"],
             ["test_classes", "43 classes are returned; class 14 is ‘Stop’ with category stop", "1"],
             ["test_model_info", "Model metadata: 43 classes, image size, parameter count", "1"],
             ["test_predict_returns_ranked_top_k", "top-k has the requested length, is sorted and matches the label", "1"],
             ["test_untrained_model_is_not_confident", "An untrained model is flagged as not confident", "1"],
             ["test_predict_rejects_non_image", "A text file is rejected with HTTP 400", "1"],
             ["test_predict_rejects_empty_file", "An empty upload is rejected with HTTP 400", "1"],
             ["test_predict_rejects_out_of_range_top_k", "top_k of 0 or 11 is rejected with HTTP 422", "2"],
             ["test_batch", "Batch predictions keep the order of the uploaded files", "1"],
             ["test_batch_rejects_more_than_32_images", "A batch of 33 images is rejected with HTTP 400", "1"],
             ["test_model_output_shape", "SignNet maps a 3×48×48 batch to 43 logits", "1"],
             ["test_every_class_has_a_known_category", "Every class maps to one of the nine categories", "1"],
             ["test_speed_limits", "Speed-limit classes return the correct limit (20, 30, 50, 100, 120)", "5"],
             ["test_only_speed_limit_signs_set_a_speed", "Only speed-limit signs carry a speed value", "1"],
             ["test_trained_model_on_demo_samples", "Trained model predicts Stop and 50 km/h samples with "
              "confidence above 0.9", "2"],
             ["Total", "", "21"]],
            [4.6, 7, 1], "Automated test suite", ["l", "l", "c"], font_size=9.5, bold_rows=(15,))
    s.h2("System-Level Validation")
    s.p(f"In addition to the unit and integration tests, the complete system was exercised manually with the "
        f"backend and frontend running together (Table {s.next_table()}).")
    s.table(["Scenario", "Expected behaviour", "Result"],
            [["Classify a clear sign from the test samples", "Correct label, high confidence, action shown", "Pass"],
             ["Classify an ambiguous sign (082_class41)", "Warning shown, action withheld", "Pass"],
             ["Batch of 12 random samples", "Grid and table, correctness per image", "Pass"],
             ["Drive simulation with 12 signs", "Speed follows the planner rules, summary chart and log", "Pass"],
             ["Model report tab", "Metrics, curves and confusion matrix load from the API", "Pass"],
             ["All 86 demo images through the API", "85 correct, the error flagged as uncertain", "Pass"],
             ["Upload larger than 5 MB", "HTTP 413 with an explanatory message", "Pass"]],
            [4.6, 5.4, 1.2], "System-level validation scenarios", ["l", "l", "c"])


def chapter_conclusion(s: Story):
    s.chapter(9, "Conclusion and Future Scope")
    s.h2("Conclusion")
    s.p("This project set out to build a complete traffic sign recognition system for autonomous vehicle "
        "simulations, from raw data to an interactive demonstration, and all objectives were met. A compact "
        f"VGG-style CNN with {N_PARAMS / 1e6:.2f} million parameters, trained from scratch on GTSRB in about eight "
        f"minutes on a laptop GPU, reaches {pct(TEST_ACC)} accuracy and a macro F1 score of {MACRO_F1:.4f} on the "
        "official test set, on par with reported human performance. The model is served through a validated "
        "FastAPI service with a median inference time of a few milliseconds, and a Streamlit application lets a "
        "user classify images, inspect the model and watch a simulated vehicle react to the signs it sees.")
    s.p("Beyond accuracy, the project shows several practices that matter for safety-related machine learning: "
        "choosing augmentations that do not change the meaning of the data, evaluating on a truly held-out test "
        "set with per-class metrics, analysing the errors that remain, and making the system aware of its own "
        "uncertainty so that it can decline to act on a doubtful prediction.")
    s.h2("Limitations")
    s.bullets([
        "<b>Classification only:</b> the system expects an image containing a single sign and cannot yet find "
        "signs inside full camera frames.",
        "<b>German signs only:</b> GTSRB covers German signs. Indian road signs differ in design and include "
        "classes that GTSRB lacks, so the model would need to be retrained or fine-tuned for Indian roads.",
        "<b>Single-frame decisions:</b> each image is classified independently, although a real vehicle sees the "
        "same sign in many consecutive frames.",
        "<b>Uncalibrated confidence:</b> softmax probabilities are not guaranteed to match true correctness "
        "rates, so the 0.6 threshold was chosen by inspection rather than calibrated.",
        "<b>Class imbalance:</b> no re-weighting or balanced sampling was used, and the weakest classes are among "
        "the rarest.",
        "<b>Simplified planner:</b> the rule-based planner illustrates the decision path but is not a vehicle "
        "controller and ignores other traffic, the road layout and the vehicle's dynamics.",
    ])
    s.h2("Future Scope")
    s.bullets([
        "<b>Detection stage:</b> add a detector such as a YOLO-family model trained on GTSDB [20] to locate signs "
        "in full dash-camera frames, and feed the crops to SignNet.",
        "<b>Temporal smoothing:</b> combine predictions over consecutive frames (for example by voting or "
        "averaging probabilities) before the planner acts.",
        "<b>Accuracy improvements:</b> use a 64×64 input, class-balanced sampling or focal loss, spatial "
        "transformer layers [11] or an ensemble to close the gap to the best published results.",
        "<b>Calibration and explainability:</b> apply temperature scaling so the threshold has a probabilistic "
        "meaning, and show Grad-CAM [19] heat maps in the interface to explain each prediction.",
        "<b>Indian traffic signs:</b> collect or adopt a dataset of Indian signs and fine-tune the model with "
        "transfer learning.",
        "<b>Simulator integration:</b> connect the API to an open driving simulator such as CARLA [16] for "
        "closed-loop testing.",
        "<b>Edge deployment:</b> export the model to TorchScript or ONNX, quantise it and package the services "
        "with Docker for embedded hardware.",
    ])


REFERENCES = [
    "J. Stallkamp, M. Schlipsing, J. Salmen and C. Igel, “The German Traffic Sign Recognition Benchmark: A "
    "multi-class classification competition,” in <i>Proc. International Joint Conference on Neural Networks "
    "(IJCNN)</i>, 2011, pp. 1453–1460.",
    "J. Stallkamp, M. Schlipsing, J. Salmen and C. Igel, “Man vs. computer: Benchmarking machine learning "
    "algorithms for traffic sign recognition,” <i>Neural Networks</i>, vol. 32, pp. 323–332, 2012.",
    "D. Cireşan, U. Meier, J. Masci and J. Schmidhuber, “Multi-column deep neural network for traffic sign "
    "classification,” <i>Neural Networks</i>, vol. 32, pp. 333–338, 2012.",
    "P. Sermanet and Y. LeCun, “Traffic sign recognition with multi-scale convolutional networks,” in <i>Proc. "
    "International Joint Conference on Neural Networks (IJCNN)</i>, 2011, pp. 2809–2813.",
    "K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in "
    "<i>Proc. International Conference on Learning Representations (ICLR)</i>, 2015.",
    "S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal "
    "covariate shift,” in <i>Proc. International Conference on Machine Learning (ICML)</i>, 2015, pp. 448–456.",
    "N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov, “Dropout: A simple way to "
    "prevent neural networks from overfitting,” <i>Journal of Machine Learning Research</i>, vol. 15, "
    "pp. 1929–1958, 2014.",
    "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in <i>Proc. International Conference "
    "on Learning Representations (ICLR)</i>, 2019.",
    "I. Loshchilov and F. Hutter, “SGDR: Stochastic gradient descent with warm restarts,” in <i>Proc. "
    "International Conference on Learning Representations (ICLR)</i>, 2017.",
    "Á. Arcos-García, J. A. Álvarez-García and L. M. Soria-Morillo, “Deep neural network for traffic sign "
    "recognition systems: An analysis of spatial transformers and stochastic optimisation methods,” <i>Neural "
    "Networks</i>, vol. 99, pp. 158–165, 2018.",
    "M. Jaderberg, K. Simonyan, A. Zisserman and K. Kavukcuoglu, “Spatial transformer networks,” in <i>Advances "
    "in Neural Information Processing Systems (NeurIPS)</i>, 2015.",
    "A. Paszke <i>et al.</i>, “PyTorch: An imperative style, high-performance deep learning library,” in "
    "<i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2019.",
    "Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, “Gradient-based learning applied to document recognition,” "
    "<i>Proceedings of the IEEE</i>, vol. 86, no. 11, pp. 2278–2324, 1998.",
    "N. Dalal and B. Triggs, “Histograms of oriented gradients for human detection,” in <i>Proc. IEEE Conference "
    "on Computer Vision and Pattern Recognition (CVPR)</i>, 2005, pp. 886–893.",
    "SAE International, “Taxonomy and definitions for terms related to driving automation systems for on-road "
    "motor vehicles,” Standard J3016, 2021.",
    "A. Dosovitskiy, G. Ros, F. Codevilla, A. López and V. Koltun, “CARLA: An open urban driving simulator,” in "
    "<i>Proc. Conference on Robot Learning (CoRL)</i>, 2017.",
    "S. Ramírez, “FastAPI documentation.” [Online]. Available: https://fastapi.tiangolo.com",
    "Streamlit Inc., “Streamlit documentation.” [Online]. Available: https://docs.streamlit.io",
    "R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, “Grad-CAM: Visual explanations "
    "from deep networks via gradient-based localization,” in <i>Proc. IEEE International Conference on Computer "
    "Vision (ICCV)</i>, 2017.",
    "S. Houben, J. Stallkamp, J. Salmen, M. Schlipsing and C. Igel, “Detection of traffic signs in real-world "
    "images: The German Traffic Sign Detection Benchmark,” in <i>Proc. International Joint Conference on Neural "
    "Networks (IJCNN)</i>, 2013.",
    f"V. A. Singh, “SignSense: Traffic sign recognition for autonomous vehicles,” source code repository, 2026. "
    f"[Online]. Available: {REPO}",
]


def references(s: Story):
    s.add(PageBreak())
    s.front_heading("REFERENCES")
    for i, ref in enumerate(REFERENCES, 1):
        s.add(Paragraph(f"[{i}]&nbsp;&nbsp;{ref}", S["ref"]))


def appendix_code(s: Story):
    s.chapter("A", "Source Code Listings", appendix=True)
    s.p(f"This appendix reproduces the key source files of the project. The complete code, including the "
        f"evaluation script, Pydantic schemas, tests and documentation, is available at {REPO}. Line numbers "
        "match the files in the repository.")
    s.listing("SignNet model", "ml/model.py")
    s.listing("Data pipeline and augmentation", "ml/data.py")
    s.listing("Training loop", "ml/train.py")
    s.listing("Inference wrapper", "backend/inference.py")
    s.listing("FastAPI backend", "backend/main.py")
    s.listing("Rule-based planner", "frontend/app.py", function="plan")
    s.listing("Sign semantics", "ml/signs.py")


def appendix_classes(s: Story):
    s.chapter("B", "GTSRB Class Reference", appendix=True)
    s.p("The 43 GTSRB classes with the category used by the planner, the number of training and test images, "
        "and the F1 score of SignNet on the test set.")
    rows = [[str(i), CLASS_NAMES[i], describe(i)["category"], f"{TRAIN_COUNTS[i]:,}", f"{TEST_COUNTS[i]:,}",
             f"{F1_BY_ID[i]:.3f}"] for i in range(43)]
    s.table(["ID", "Class name", "Category", "Train", "Test", "F1"], rows, [0.7, 5.2, 2.6, 1.1, 1.0, 1.0],
            "GTSRB classes, categories, image counts and test F1 scores", ["c", "l", "l", "r", "r", "c"],
            font_size=9.2)


def build():
    doc = ReportDoc(OUT)
    s = Story(doc.width)
    screens = {name: crop_screenshot(FIG / "screens" / f"{name}.png")
               for name in ["classify", "low_confidence", "batch", "drive", "model_report"]}
    title_page(s)
    s.add(NextPageTemplate("front"), PageBreak())
    acknowledgement(s)
    abstract(s)
    front_lists(s)
    s.add(NextPageTemplate("body"))
    chapter_introduction(s)
    chapter_literature(s)
    chapter_requirements(s)
    chapter_dataset(s)
    chapter_design(s)
    chapter_method(s)
    chapter_results(s, screens)
    chapter_testing(s)
    chapter_conclusion(s)
    references(s)
    appendix_code(s)
    appendix_classes(s)
    doc.multiBuild(s.items)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    build()
