from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

WORKSPACE = Path("/Users/Yuki/TKNT")
SOURCE_PATH = (
    WORKSPACE / "output" / "documents" / "ban_tom_tat_roomflow_nguyen_thanh_minh.md"
)
OUTPUT_PATH = (
    WORKSPACE / "output" / "documents" / "ban_tom_tat_roomflow_nguyen_thanh_minh.docx"
)

FONT_NAME = "Calibri"
BODY_COLOR = RGBColor(26, 34, 44)
HEADING_BLUE = RGBColor(46, 116, 181)
HEADING_DARK_BLUE = RGBColor(31, 77, 120)
INK_BLUE = RGBColor(11, 37, 69)
MUTED = RGBColor(96, 108, 122)


def set_run_font(
    run: object,
    *,
    size: float,
    color: RGBColor,
    bold: bool | None = None,
    italic: bool | None = None,
) -> None:
    run.font.name = FONT_NAME
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT_NAME)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT_NAME)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT_NAME)
    run.font.size = Pt(size)
    run.font.color.rgb = color
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_paragraph_tokens(
    paragraph: object,
    *,
    before: float,
    after: float,
    line_spacing: float,
    keep_with_next: bool = False,
) -> None:
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line_spacing
    fmt.keep_with_next = keep_with_next
    fmt.widow_control = True


def configure_styles(document: Document) -> None:
    styles = document.styles

    normal = styles["Normal"]
    normal.font.name = FONT_NAME
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)
    normal.font.size = Pt(11)
    normal.font.color.rgb = BODY_COLOR
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25
    normal.paragraph_format.widow_control = True

    title = styles.add_style("RoomFlow Title", WD_STYLE_TYPE.PARAGRAPH)
    title.base_style = normal
    title.font.name = FONT_NAME
    title._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
    title._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
    title._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)
    title.font.size = Pt(23)
    title.font.bold = True
    title.font.color.rgb = INK_BLUE
    title.paragraph_format.space_before = Pt(12)
    title.paragraph_format.space_after = Pt(8)
    title.paragraph_format.line_spacing = 1.0
    title.paragraph_format.keep_with_next = True

    heading_1 = styles["Heading 1"]
    heading_1.font.name = FONT_NAME
    heading_1._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
    heading_1._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
    heading_1._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)
    heading_1.font.size = Pt(16)
    heading_1.font.bold = True
    heading_1.font.color.rgb = HEADING_BLUE
    heading_1.paragraph_format.space_before = Pt(18)
    heading_1.paragraph_format.space_after = Pt(10)
    heading_1.paragraph_format.line_spacing = 1.0
    heading_1.paragraph_format.keep_with_next = True
    heading_1.paragraph_format.keep_together = True

    heading_2 = styles["Heading 2"]
    heading_2.font.name = FONT_NAME
    heading_2._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
    heading_2._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
    heading_2._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)
    heading_2.font.size = Pt(13)
    heading_2.font.bold = True
    heading_2.font.color.rgb = HEADING_DARK_BLUE
    heading_2.paragraph_format.space_before = Pt(14)
    heading_2.paragraph_format.space_after = Pt(7)
    heading_2.paragraph_format.line_spacing = 1.0
    heading_2.paragraph_format.keep_with_next = True
    heading_2.paragraph_format.keep_together = True

    heading_3 = styles["Heading 3"]
    heading_3.font.name = FONT_NAME
    heading_3._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
    heading_3._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
    heading_3._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)
    heading_3.font.size = Pt(12)
    heading_3.font.bold = True
    heading_3.font.color.rgb = HEADING_DARK_BLUE
    # Named override: compact experiment subheads keep the brief within four pages.
    heading_3.paragraph_format.space_before = Pt(4)
    heading_3.paragraph_format.space_after = Pt(3)
    heading_3.paragraph_format.line_spacing = 1.0
    heading_3.paragraph_format.keep_with_next = True
    heading_3.paragraph_format.keep_together = True


def next_numbering_id(numbering: object, tag: str, attribute: str) -> int:
    existing = []
    for element in numbering.findall(qn(tag)):
        value = element.get(qn(attribute))
        if value is not None and value.isdigit():
            existing.append(int(value))
    return max(existing, default=0) + 1


def create_compact_bullet_numbering(document: Document) -> int:
    numbering = document.part.numbering_part.element
    abstract_id = next_numbering_id(numbering, "w:abstractNum", "w:abstractNumId")
    num_id = next_numbering_id(numbering, "w:num", "w:numId")

    abstract_num = OxmlElement("w:abstractNum")
    abstract_num.set(qn("w:abstractNumId"), str(abstract_id))

    multi_level = OxmlElement("w:multiLevelType")
    multi_level.set(qn("w:val"), "singleLevel")
    abstract_num.append(multi_level)

    level = OxmlElement("w:lvl")
    level.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    level.append(start)
    num_format = OxmlElement("w:numFmt")
    num_format.set(qn("w:val"), "bullet")
    level.append(num_format)
    level_text = OxmlElement("w:lvlText")
    level_text.set(qn("w:val"), "•")
    level.append(level_text)
    level_justification = OxmlElement("w:lvlJc")
    level_justification.set(qn("w:val"), "left")
    level.append(level_justification)

    paragraph_properties = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "540")
    tabs.append(tab)
    paragraph_properties.append(tabs)
    indent = OxmlElement("w:ind")
    indent.set(qn("w:left"), "540")
    indent.set(qn("w:hanging"), "270")
    paragraph_properties.append(indent)
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), "0")
    spacing.set(qn("w:after"), "80")
    spacing.set(qn("w:line"), "300")
    spacing.set(qn("w:lineRule"), "auto")
    paragraph_properties.append(spacing)
    level.append(paragraph_properties)

    run_properties = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), FONT_NAME)
    fonts.set(qn("w:hAnsi"), FONT_NAME)
    fonts.set(qn("w:eastAsia"), FONT_NAME)
    run_properties.append(fonts)
    level.append(run_properties)

    abstract_num.append(level)
    numbering.append(abstract_num)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abstract_num_id = OxmlElement("w:abstractNumId")
    abstract_num_id.set(qn("w:val"), str(abstract_id))
    num.append(abstract_num_id)
    numbering.append(num)
    return num_id


def apply_bullet_numbering(paragraph: object, num_id: int) -> None:
    properties = paragraph._p.get_or_add_pPr()
    num_properties = OxmlElement("w:numPr")
    level = OxmlElement("w:ilvl")
    level.set(qn("w:val"), "0")
    numbering_id = OxmlElement("w:numId")
    numbering_id.set(qn("w:val"), str(num_id))
    num_properties.append(level)
    num_properties.append(numbering_id)
    properties.append(num_properties)


def add_page_number(paragraph: object) -> None:
    run = paragraph.add_run("Trang ")
    set_run_font(run, size=9, color=MUTED)
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    field_separate = OxmlElement("w:fldChar")
    field_separate.set(qn("w:fldCharType"), "separate")
    field_text = OxmlElement("w:t")
    field_text.text = "1"
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run._r.extend([field_begin, instruction, field_separate, field_text, field_end])


def configure_page_furniture(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.right_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    header_paragraph = section.header.paragraphs[0]
    header_paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_paragraph_tokens(
        header_paragraph,
        before=0,
        after=0,
        line_spacing=1.0,
    )
    header_run = header_paragraph.add_run("ROOMFLOW  |  BẢN TÓM TẮT ĐỒ ÁN")
    set_run_font(header_run, size=8.5, color=MUTED, bold=True)

    footer_paragraph = section.footer.paragraphs[0]
    footer_paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_paragraph_tokens(
        footer_paragraph,
        before=0,
        after=0,
        line_spacing=1.0,
    )
    add_page_number(footer_paragraph)


INLINE_PATTERN = re.compile(r"(\*\*.+?\*\*|\*.+?\*)")


def add_inline_markdown(paragraph: object, text: str, *, size: float = 11) -> None:
    cleaned = text.rstrip().removesuffix("  ")
    cursor = 0
    for match in INLINE_PATTERN.finditer(cleaned):
        if match.start() > cursor:
            run = paragraph.add_run(cleaned[cursor : match.start()])
            set_run_font(run, size=size, color=BODY_COLOR)
        token = match.group(0)
        if token.startswith("**"):
            content = token[2:-2]
            run = paragraph.add_run(content)
            set_run_font(run, size=size, color=INK_BLUE, bold=True)
        else:
            content = token[1:-1]
            run = paragraph.add_run(content)
            set_run_font(run, size=size, color=BODY_COLOR, italic=True)
        cursor = match.end()
    if cursor < len(cleaned):
        run = paragraph.add_run(cleaned[cursor:])
        set_run_font(run, size=size, color=BODY_COLOR)


def add_markdown_content(
    document: Document, source_text: str, bullet_num_id: int
) -> None:
    for raw_line in source_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("# "):
            paragraph = document.add_paragraph(style="RoomFlow Title")
            add_inline_markdown(paragraph, line[2:], size=23)
            continue

        if line.startswith("## "):
            paragraph = document.add_paragraph(style="Heading 1")
            add_inline_markdown(paragraph, line[3:], size=16)
            continue

        if line.startswith("### "):
            paragraph = document.add_paragraph(style="Heading 2")
            add_inline_markdown(paragraph, line[4:], size=13)
            continue

        if re.fullmatch(r"\*\*TN\d+.+\*\*", line):
            paragraph = document.add_paragraph(style="Heading 3")
            add_inline_markdown(paragraph, line, size=12)
            continue

        if line.startswith("- "):
            paragraph = document.add_paragraph(style="Normal")
            set_paragraph_tokens(
                paragraph,
                before=0,
                after=4,
                line_spacing=1.25,
            )
            apply_bullet_numbering(paragraph, bullet_num_id)
            add_inline_markdown(paragraph, line[2:])
            continue

        paragraph = document.add_paragraph(style="Normal")
        set_paragraph_tokens(
            paragraph,
            before=0,
            after=6,
            line_spacing=1.25,
        )
        add_inline_markdown(paragraph, line)


def set_document_properties(document: Document) -> None:
    properties = document.core_properties
    properties.title = "Bản tóm tắt đồ án RoomFlow"
    properties.subject = (
        "Bài toán, đầu vào, đầu ra, đóng góp và thực nghiệm của RoomFlow"
    )
    properties.author = "Nguyễn Thành Minh"
    properties.keywords = "RoomFlow, interior layout, photorealistic rendering"
    properties.comments = (
        "Prepared from the submitted thesis, defense slides, and code."
    )


def main() -> None:
    source_text = SOURCE_PATH.read_text(encoding="utf-8")
    document = Document()
    configure_styles(document)
    configure_page_furniture(document)
    set_document_properties(document)
    bullet_num_id = create_compact_bullet_numbering(document)
    add_markdown_content(document, source_text, bullet_num_id)
    document.save(OUTPUT_PATH)


if __name__ == "__main__":
    main()
