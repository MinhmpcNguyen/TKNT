from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


OUTPUT = Path(
    "/Users/Yuki/TKNT/output/documents/"
    "Tom_tat_bai_toan_dong_gop_thuc_nghiem_RoomFlow.docx"
)

NAVY = RGBColor(11, 37, 69)
BLUE = RGBColor(46, 116, 181)
MUTED = RGBColor(95, 105, 120)
BLACK = RGBColor(0, 0, 0)


def set_run_font(run, size=10.0, bold=False, italic=False, color=BLACK):
    run.font.name = "Calibri"
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Calibri")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Calibri")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = color


def set_cell_margins(cell, top=60, start=80, bottom=60, end=80):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Trang ")
    set_run_font(run, size=8.5, color=MUTED)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)


def add_title(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run("TÓM TẮT BÀI TOÁN, ĐÓNG GÓP VÀ THỰC NGHIỆM")
    set_run_font(run, size=16.5, bold=True, color=NAVY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(
        "An Interactive Generative AI System for Constraint-Aware Interior Layout "
        "and Photorealistic Synthesis"
    )
    set_run_font(run, size=9.2, italic=True, color=MUTED)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    run = p.add_run("Nguyễn Thành Minh | 20225450")
    set_run_font(run, size=9.2, bold=True, color=NAVY)


def add_body(doc, text, after=2.2, bold_label=None):
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.widow_control = True
    if bold_label and text.startswith(bold_label):
        label = p.add_run(bold_label)
        set_run_font(label, size=10.0, bold=True, color=NAVY)
        rest = p.add_run(text[len(bold_label) :])
        set_run_font(rest, size=10.0)
    else:
        run = p.add_run(text)
        set_run_font(run, size=10.0)
    return p


def add_h1(doc, text):
    p = doc.add_paragraph(style="Heading 1")
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2.5)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_run_font(run, size=13.3, bold=True, color=NAVY)
    return p


def add_h2(doc, text):
    p = doc.add_paragraph(style="Heading 2")
    p.paragraph_format.space_before = Pt(2.8)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_run_font(run, size=10.7, bold=True, color=BLUE)
    return p


def add_h3(doc, text):
    p = doc.add_paragraph(style="Heading 3")
    p.paragraph_format.space_before = Pt(2.5)
    p.paragraph_format.space_after = Pt(1.2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_run_font(run, size=10.1, bold=True, color=NAVY)
    return p


def add_exp(doc, label, idea, data, result):
    add_h3(doc, label)
    add_body(doc, f"5a. Ý tưởng: {idea}", bold_label="5a. Ý tưởng:")
    add_body(doc, f"5b. Dữ liệu: {data}", bold_label="5b. Dữ liệu:")
    add_body(doc, f"5c. Kết quả: {result}", bold_label="5c. Kết quả:")


def configure_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(10)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(2.2)
    normal.paragraph_format.line_spacing = 1.0

    for style_name in ("Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.paragraph_format.keep_with_next = True


def configure_section(section):
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.25)
    section.bottom_margin = Cm(1.2)
    section.left_margin = Cm(1.45)
    section.right_margin = Cm(1.45)
    section.header_distance = Cm(0.55)
    section.footer_distance = Cm(0.55)


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    configure_styles(doc)
    configure_section(doc.sections[0])

    for section in doc.sections:
        add_page_number(section.footer.paragraphs[0])

    add_title(doc)
    add_body(
        doc,
        "Đồ án xây dựng RoomFlow, một hệ thống hỗ trợ thiết kế nội thất giai đoạn "
        "đầu. Hệ thống giải quyết hai bài toán liên tiếp: tạo bố trí nội thất có "
        "thể chỉnh sửa; sau đó chuyển bố trí đã chọn thành ảnh chân thực và cho "
        "phép chỉnh sửa cục bộ.",
        after=3.0,
    )

    add_h1(doc, "Bài toán 1. Tự động tạo bố trí nội thất hợp lệ và có thể chỉnh sửa")

    add_h2(doc, "1. Đầu vào")
    add_body(
        doc,
        "Mô tả tự nhiên về nhu cầu, công năng và phong cách; hình học phòng dưới "
        "dạng đa giác; vị trí cửa, cửa sổ; và danh mục nội thất gồm loại đồ, kích "
        "thước, phong cách, vật liệu và mô hình 3D nếu có.",
    )

    add_h2(doc, "2. Đầu ra")
    add_body(
        doc,
        "Một tập phương án bố trí được xếp hạng. Mỗi đồ vật có định danh, kích "
        "thước, vị trí và hướng quay; kết quả có thể kiểm tra trên 2D, xem trước "
        "trong 3D và tiếp tục chỉnh sửa.",
    )

    add_h2(doc, "3. Các đóng góp sáng tạo")
    add_body(
        doc,
        "Đóng góp 1 - Tách hiểu ngữ nghĩa và giải hình học. AI chuyển yêu cầu "
        "thành danh sách đồ vật, mức ưu tiên, quan hệ không gian và phong cách; "
        "tọa độ cuối cùng do bộ máy bố trí hình học tìm kiếm.",
        bold_label="Đóng góp 1 - Tách hiểu ngữ nghĩa và giải hình học.",
    )
    add_body(
        doc,
        "Đóng góp 2 - Bộ giải anchor-first có kiểm tra và sửa lỗi. Hệ thống đặt "
        "đồ chính trước, đặt đồ phụ theo quan hệ, rồi kiểm tra nằm trong phòng, "
        "va chạm, cửa và lối đi; phương án gần hợp lệ được dịch chuyển để sửa.",
        bold_label="Đóng góp 2 - Bộ giải anchor-first có kiểm tra và sửa lỗi.",
    )
    add_body(
        doc,
        "Đóng góp 3 - Bố trí là trạng thái trung tâm. Cùng một dữ liệu phục vụ "
        "chỉnh sửa 2D, xem 3D và sinh ảnh; gallery selection loại các phương án "
        "gần trùng để trả về các lựa chọn thực sự khác nhau.",
        bold_label="Đóng góp 3 - Bố trí là trạng thái trung tâm.",
    )

    add_h2(doc, "4. Điểm hay, đã có ai làm chưa và hơn ở đâu")
    add_body(
        doc,
        "LayoutGPT, I-Design, FlairGPT, Co-Layout và HOLODECK đã nghiên cứu các "
        "hướng liên quan, nên không nên tuyên bố chưa ai làm. Điểm mới của đồ án "
        "là kết hợp hiểu yêu cầu, giải hình học, chỉnh sửa và trực quan hóa trong "
        "một quy trình dùng được. So với việc để AI sinh tọa độ, RoomFlow dùng AI "
        "cho phần ngữ nghĩa và hình học cho phần cần độ chính xác, nhờ đó xử lý "
        "tốt hơn cửa, cửa sổ, va chạm, lối đi và phòng không chữ nhật.",
    )

    add_h2(doc, "5. Các thực nghiệm")
    add_exp(
        doc,
        "Thực nghiệm 1 - So sánh với hệ thống khác",
        "So sánh RoomFlow với I-Design và FlairGPT theo khả thi không gian, đáp "
        "ứng yêu cầu, công năng và số đồ vật sử dụng được.",
        "50 case phòng ngủ/phòng khách: 20 phòng chữ nhật đơn giản, 20 phòng chữ "
        "nhật chi tiết và 10 phòng không chữ nhật; có cả trường hợp có/không có "
        "cửa, cửa sổ; mỗi case chạy 2 lần.",
        "RoomFlow cao hơn I-Design 15,4 điểm tổng, 34,7 điểm khả thi không gian "
        "và 26,1 điểm công năng; cao hơn FlairGPT 58,6 điểm tổng. Với phòng không "
        "chữ nhật có cửa, ba hệ thống lần lượt đạt 99,7; 46,9; và 31,7.",
    )
    add_exp(
        doc,
        "Thực nghiệm 2 - Ablation và robustness",
        "Lần lượt bỏ semantic planning, request priority, style, material, "
        "stylist và gallery selection để xác định vai trò từng thành phần.",
        "Benchmark chính dùng các case bố trí chung. Riêng robustness dùng đúng "
        "10 case holdout; mỗi case chạy 2 lần, tương ứng 20 case-run cho mỗi "
        "biến thể.",
        "Pipeline đầy đủ đạt 95,3 điểm bố trí; bỏ semantic còn 85,0 và minimal "
        "context còn 77,1. Trên 10 case robustness, pipeline đầy đủ đạt 94,5 "
        "điểm tổng hợp, 86,9 điểm worst-case và 100 điểm gallery diversity; một "
        "số biến thể ablation chỉ còn khoảng 7-8 điểm worst-case.",
    )

    doc.add_page_break()

    add_h1(doc, "Bài toán 2. Sinh và chỉnh sửa ảnh chân thực nhưng vẫn giữ bố trí")

    add_h2(doc, "1. Đầu vào")
    add_body(
        doc,
        "Khi sinh ảnh: bố trí đã chọn, ảnh 3D, góc camera, danh sách đồ vật nhìn "
        "thấy và preset phong cách/ánh sáng/bối cảnh. Khi chỉnh sửa: ảnh nguồn, "
        "yêu cầu thêm, xóa, thay thế hoặc đổi màu, vùng/đối tượng mục tiêu và các "
        "thông tin cần giữ nguyên.",
    )

    add_h2(doc, "2. Đầu ra")
    add_body(
        doc,
        "Ảnh phối cảnh chân thực nhưng vẫn giữ hình dạng phòng, cửa, góc camera "
        "và vị trí nội thất chính; hoặc ảnh đã chỉnh sửa trong đó thay đổi được "
        "giới hạn vào đúng đối tượng/vùng được chọn. Ảnh vẫn liên kết với bố trí "
        "hoặc ảnh nguồn để truy vết.",
    )

    add_h2(doc, "3. Các đóng góp sáng tạo")
    add_body(
        doc,
        "Đóng góp 1 - Snapshot with Edge/Silhouette Guidance (SE-ESG). Hệ thống "
        "kết hợp ảnh 3D, đường biên và silhouette làm bằng chứng cấu trúc để mô "
        "hình ảnh biết phần hình học nào phải giữ.",
        bold_label="Đóng góp 1 - Snapshot with Edge/Silhouette Guidance (SE-ESG).",
    )
    add_body(
        doc,
        "Đóng góp 2 - Object-Scope Target Validation. Trước khi chỉnh sửa, hệ "
        "thống xác định rõ đối tượng, vùng được phép thay đổi và các phần phải "
        "khóa, giúp giảm thay đổi ngoài ý muốn.",
        bold_label="Đóng góp 2 - Object-Scope Target Validation.",
    )
    add_body(
        doc,
        "Đóng góp 3 - Quy trình xuyên suốt từ bố trí đến ảnh. Ảnh không phải kết "
        "quả độc lập mà được sinh từ bố trí có thể chỉnh sửa; người dùng có thể "
        "sửa bố trí rồi sinh lại hoặc chỉ chỉnh một vùng trên ảnh.",
        bold_label="Đóng góp 3 - Quy trình xuyên suốt từ bố trí đến ảnh.",
    )

    add_h2(doc, "4. Điểm hay, đã có ai làm chưa và hơn ở đâu")
    add_body(
        doc,
        "Ctrl-Room, SceneCraft và SpatialGen đã dùng điều kiện không gian để sinh "
        "ảnh/cảnh 3D, nhưng thường hướng tới panorama, NeRF hoặc cần pipeline "
        "chuyên biệt. RoomFlow chọn hướng nhẹ hơn: dùng trực tiếp góc nhìn 3D mà "
        "người dùng chọn, kết hợp metadata và ràng buộc bảo toàn để điều khiển mô "
        "hình ảnh có sẵn. Điểm mạnh là tăng tính nhất quán giữa bố trí chỉnh sửa "
        "được và ảnh đầu ra, đồng thời cho phép chỉnh đúng vùng thay vì chỉ dùng "
        "prompt chung.",
    )

    add_h2(doc, "5. Các thực nghiệm")
    add_exp(
        doc,
        "Thực nghiệm 1 - Sinh ảnh chân thực",
        "So sánh 6 mức hướng dẫn để đánh giá khả năng giữ đường biên, chiều sâu, "
        "vùng ảnh và đồ vật.",
        "14 bố trí phòng ngủ/phòng khách; mỗi case chạy 2 lần cho từng biến thể.",
        "SE-ESG đạt macro score cao nhất 0,587, so với 0,577 của Snapshot Prompt "
        "và 0,554 của LVP. Preset fidelity đạt 0,918, cao hơn ngưỡng đạt 0,75. "
        "SE-ESG cho cân bằng bảo toàn bố trí tốt nhất, dù không đứng đầu mọi chỉ "
        "số riêng lẻ.",
    )
    add_exp(
        doc,
        "Thực nghiệm 2 - Chỉnh sửa ảnh cục bộ",
        "So sánh Object-Scope Target Validation với prompt-image-only để đánh "
        "giá thao tác có đúng mục tiêu và giữ được phần còn lại hay không.",
        "32 yêu cầu, chia đều cho thêm, đổi màu, xóa và thay thế; mỗi yêu cầu "
        "chạy 2 lần.",
        "Object-Scope đạt 0,956, cao hơn 0,897 của prompt-only. Cải thiện lớn nhất "
        "ở đổi màu: 1,000 so với 0,775. Thay thế là thao tác khó nhất, đạt 0,887 "
        "và 0,875.",
    )

    add_body(
        doc,
        "Kết luận: RoomFlow kết nối hiểu yêu cầu bằng AI, bố trí hình học có kiểm "
        "tra, chỉnh sửa 2D/3D và sinh/chỉnh sửa ảnh có kiểm soát thành một quy "
        "trình thống nhất.",
        after=0,
        bold_label="Kết luận:",
    )

    doc.core_properties.title = "Tóm tắt bài toán, đóng góp và thực nghiệm RoomFlow"
    doc.core_properties.subject = "Đồ án tốt nghiệp - Nguyễn Thành Minh"
    doc.core_properties.author = "Nguyễn Thành Minh"
    doc.core_properties.keywords = "RoomFlow, bố trí nội thất, sinh ảnh, robustness"
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
