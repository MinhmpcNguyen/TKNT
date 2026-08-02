# BẢN TÓM TẮT ĐỒ ÁN - ROOMFLOW

**Sinh viên:** Nguyễn Thành Minh - **MSSV:** 20225450  
**Đề tài:** *An Interactive Generative AI System for Constraint-Aware Interior Layout and Photorealistic Synthesis*

## Tóm tắt một câu

RoomFlow giải quyết một quy trình liên thông cho thiết kế nội thất giai đoạn sớm: từ yêu cầu ngôn ngữ tự nhiên và hình học phòng, hệ thống tạo bố trí nội thất hợp lệ, có thể chỉnh sửa; sau đó dùng chính bố trí đã chọn để sinh và chỉnh sửa ảnh chân thực mà vẫn cố gắng giữ nguyên cấu trúc không gian.

Đồ án có **hai bài toán chính**. Bài toán 2 gồm hai nhánh: sinh ảnh và chỉnh sửa ảnh. Ba đóng góp cấp đồ án là: (i) quy trình bố trí lai giữa lập kế hoạch ngữ nghĩa và kiểm tra hình học; (ii) quy trình sinh/chỉnh sửa ảnh có điều kiện theo bố trí; (iii) ứng dụng web tích hợp toàn bộ chu trình.

## Bài toán 1 - Sinh bố trí nội thất hợp lệ và có thể chỉnh sửa

### 1. Đầu vào

- Loại phòng (phòng khách hoặc phòng ngủ), đa giác biên phòng, cửa đi, cửa sổ.
- Danh mục đồ nội thất có loại, kích thước, thuộc tính phong cách/vật liệu và tài sản 3D nếu có.
- Yêu cầu bằng ngôn ngữ tự nhiên: đồ bắt buộc/tùy chọn, số lượng, công năng, quan hệ bố trí và sở thích thẩm mỹ.

### 2. Đầu ra

- Một nhóm phương án bố trí 2D được xếp hạng, gồm loại đồ, vị trí, kích thước, hướng quay, điểm số và chẩn đoán vi phạm.
- Phương án được chọn có thể chỉnh sửa trực tiếp và dựng thành cảnh 3D; bố trí này là trạng thái thiết kế trung tâm cho các bước sinh ảnh sau đó.

### 3. Các đóng góp sáng tạo

- **ĐG1.1 - Lập kế hoạch ngữ nghĩa có cấu trúc:** chuyển yêu cầu tự do thành “hợp đồng yêu cầu” và kế hoạch máy có thể kiểm tra: đồ bắt buộc/ưu tiên/tùy chọn/cấm, số lượng, cụm công năng, quan hệ và ưu tiên không gian.
- **ĐG1.2 - Tách hiểu ý định khỏi đặt tọa độ:** mô hình ngôn ngữ quyết định *cần gì và quan hệ ra sao*; bộ giải hình học quyết định *đặt ở đâu*. Bộ giải sinh ứng viên, kiểm tra nằm trong phòng, va chạm, khoảng trống cửa và lối đi, sửa xung đột nhẹ, rồi chấm điểm.
- **ĐG1.3 - Gallery phương án có thể chỉnh sửa:** chỉ giữ các phương án qua cổng hợp lệ, loại gần trùng và trả về các lựa chọn khác biệt có chẩn đoán, thay vì một ảnh hoặc một tập tọa độ “chết”.

### 4. Điểm hay, đã có ai làm chưa, và hơn ở đâu

- LLM lập kế hoạch bố trí **không phải ý tưởng hoàn toàn chưa từng có**: LayoutGPT, I-Design và FlairGPT đã dùng LLM, biểu diễn có cấu trúc hoặc tối ưu bố trí. Điểm mới thực tế của RoomFlow là đóng gói chuỗi *yêu cầu tự nhiên -> kế hoạch có kiểu -> bộ giải xác định -> phương án chỉnh sửa được* trong một ứng dụng thống nhất.
- So với cách để LLM sinh trực tiếp tọa độ, việc tách ngữ nghĩa và hình học làm cho lỗi biên phòng, chồng lấn, chắn cửa và đứt lối đi trở thành các điều kiện kiểm tra được, không phụ thuộc vào “niềm tin” vào câu trả lời của LLM.
- So với các baseline được thử nghiệm, lợi thế không nằm ở bài toán rất dễ. Khi phòng chữ nhật không có cửa, I-Design đạt 99,4 điểm, cao hơn RoomFlow 93,3. Lợi thế của RoomFlow xuất hiện rõ khi có cửa/cửa sổ hoặc biên phòng bất quy tắc, tức đúng phần mà ràng buộc hình học có giá trị.

### 5. Thực nghiệm

**TN1 - So sánh với I-Design và FlairGPT**

- **5a. Ý tưởng:** cho ba hệ thống giải cùng các ca, chuẩn hóa đầu ra về cùng schema, rồi đánh giá bốn mặt: khả thi hình học, bám yêu cầu, chất lượng công năng và số đồ dùng được.
- **5b. Dữ liệu:** 50 ca kiểm soát do đồ án xây dựng, lấy cảm hứng từ bố trí thực tế nhưng không sao chép trực tiếp: 20 phòng chữ nhật yêu cầu tối giản, 20 phòng chữ nhật yêu cầu chi tiết và 10 phòng bất quy tắc (chữ L, vát góc, hõm cạnh, dạng bay); có cả trường hợp có và không có cửa/cửa sổ. Mỗi ca chạy hai lần. Dữ liệu không ép một đáp án duy nhất mà lưu các ràng buộc đầu ra mong đợi.
- **5c. Kết quả:** trên so sánh cặp toàn bộ tập thử, RoomFlow tăng **15,4 điểm** so với I-Design (CI 95%: **[10,5; 20,4]**, Holm-adjusted **p < 0,001**) và **58,6 điểm** so với FlairGPT. Mức tăng tập trung ở khả thi hình học (**+34,7** so với I-Design; **+88,0** so với FlairGPT) và công năng (**+26,1**; **+82,4**). Với phòng bất quy tắc có openings, điểm tổng lần lượt là **99,7 / 46,9 / 31,7** cho RoomFlow / I-Design / FlairGPT.

**TN2 - Ablation các thành phần nội bộ**

- **5a. Ý tưởng:** bỏ hoặc làm yếu từng khâu để xác định phần nào tạo ra chất lượng tức thời và phần nào tạo ra độ ổn định, phong cách và đa dạng phương án.
- **5b. Dữ liệu:** cùng giao thức benchmark; bổ sung một holdout có yêu cầu phong cách, vật liệu, bảng màu và đa dạng gallery.
- **5c. Kết quả:** bản đầy đủ đạt **95,3** điểm; bỏ lập kế hoạch ngữ nghĩa còn **85,0**, chỉ giữ ngữ cảnh tối thiểu còn **77,1**. Một số biến thể bỏ style/gallery/stylist lại nhỉnh hơn trên bốn metric hình học-ngắn hạn, nên không thể nói mọi thành phần đều làm tăng điểm tức thời. Giá trị của các khâu hậu xử lý thể hiện ở robustness: bản đầy đủ đạt **95,0**, worst-case **90,4**, trong khi các biến thể chính giảm rõ rệt ở chất lượng đuôi thấp và đa dạng phương án.

## Bài toán 2 - Sinh và chỉnh sửa ảnh chân thực nhưng giữ bố trí

### 1. Đầu vào

**2A - Sinh ảnh:** bố trí đã chọn, ảnh chụp cảnh 3D, camera cố định, hình học phòng và openings, danh sách vật thể nhìn thấy, preset phong cách/ánh sáng/bối cảnh và các điều kiện cần bảo toàn.

**2B - Chỉnh sửa ảnh:** ảnh render đã duyệt, thao tác add/remove/replace/recolor, mô tả thay đổi, vật thể hoặc hộp vùng đích và ngữ cảnh cần giữ nguyên.

### 2. Đầu ra

- Ảnh phối cảnh chân thực hơn về vật liệu, ánh sáng và không khí, nhưng vẫn bám camera, vỏ phòng, openings và bố trí đồ chính.
- Ảnh sau chỉnh sửa chỉ thay đổi đối tượng/vùng đã chọn; phần còn lại của căn phòng được giữ ổn định ở mức có thể.

### 3. Các đóng góp sáng tạo

- **ĐG2.1 - Gói hướng dẫn sinh ảnh theo bố trí:** biên dịch cảnh 3D thành bằng chứng có cấu trúc gồm ảnh snapshot, vật thể nhìn thấy, quan hệ không gian, điều kiện bảo toàn và đặc biệt là hướng dẫn cạnh/silhouette (SE-ESG).
- **ĐG2.2 - Chỉnh sửa theo phạm vi đối tượng:** kiểm tra trước mục tiêu chỉnh sửa, khóa camera/phòng/đồ không liên quan và tạo mask sao cho vùng đích được phép thay đổi còn vùng khác được bảo vệ.
- **ĐG3 - Ứng dụng web liên thông:** người dùng tạo phòng, sinh và chọn phương án, chỉnh 2D, xem 3D, khóa góc camera, sinh ảnh và chỉnh ảnh trong cùng một trạng thái thiết kế.

### 4. Điểm hay, đã có ai làm chưa, và hơn ở đâu

- Sinh ảnh có điều kiện theo layout **đã có** trong Ctrl-Room, SceneCraft, SpatialGen và các hướng tương tự. Các công trình này thường nhắm tới panorama, multi-view hoặc tái dựng 3D và cần pipeline/dữ liệu chuyên biệt. RoomFlow chọn hướng nhẹ hơn cho ứng dụng: tận dụng chính cảnh 3D chỉnh sửa được và góc camera người dùng chọn để tạo điều kiện cho mô hình ảnh.
- SE-ESG không thắng mọi metric; nó tốt nhất về **đánh đổi tổng thể** giữa cạnh, chiều sâu, đặc trưng vùng và vật thể. Vì vậy đóng góp nên được mô tả là “cải thiện mức giữ bố trí trong thiết lập thử nghiệm”, không nên tuyên bố là phương pháp tốt nhất tuyệt đối.
- Chỉnh sửa cục bộ cũng không phải bài toán mới. Điểm riêng của RoomFlow là nối hộp/vật thể đích với metadata từ bố trí và áp dụng preflight + preservation locks. Thực nghiệm mới chỉ so với prompt-image-only, chưa đủ để khẳng định hơn mọi hệ thống chỉnh ảnh đã công bố.

### 5. Thực nghiệm

**TN3 - So sánh sáu biến thể sinh ảnh**

- **5a. Ý tưởng:** giữ nguyên layout, camera và preset, chỉ thay đổi lượng bằng chứng bố trí đưa vào mô hình; đo mức bảo toàn bằng Edge, Depth, DINO patch và DINO object, đồng thời kiểm tra độ bám preset.
- **5b. Dữ liệu:** 14 ca lấy từ các layout RoomFlow đã lưu, phủ phòng khách/phòng ngủ, nhiều hình phòng, openings và góc camera; mỗi ca chạy hai lần cho sáu biến thể.
- **5c. Kết quả:** SE-ESG có macro cao nhất **0,587**, so với snapshot-prompt **0,577** và LVP **0,554**. SE-ESG cao nhất ở Edge (**0,606**) và DINO patch (**0,570**), nhưng snapshot-prompt nhỉnh hơn ở Depth (**0,731 so với 0,727**) và DINO object (**0,469 so với 0,445**). Preset fidelity của SE-ESG là **0,918**, vượt ngưỡng 0,75.

**TN4 - So sánh chỉnh sửa theo phạm vi đối tượng với prompt-only**

- **5a. Ý tưởng:** cùng ảnh nguồn và yêu cầu chỉnh sửa, so sánh việc có/không có target scope, preflight và preservation context; đánh giá vừa đúng yêu cầu chỉnh sửa vừa giữ phần không liên quan.
- **5b. Dữ liệu:** 32 yêu cầu matched, chia đều cho add, recolor, remove và replace; mỗi yêu cầu chạy hai lần cho hai biến thể. Điểm chỉnh sửa do một người chấm theo rubric riêng cho từng thao tác.
- **5c. Kết quả:** object-scope đạt **0,956**, cao hơn prompt-only **0,897**. Cải thiện lớn nhất ở recolor: **1,000 so với 0,775**. Remove là **0,975 so với 0,963**; replace là thao tác khó nhất (**0,887 so với 0,875**). Với add, prompt-only hơi cao hơn (**0,975 so với 0,963**), nên kết luận đúng là object-scope cải thiện tổng thể và đặc biệt mạnh ở recolor, không phải thắng mọi thao tác.

## Phạm vi kết luận

Các kết quả chứng minh hiệu quả **trong thiết lập đã thử nghiệm**, chủ yếu cho phòng khách và phòng ngủ. Benchmark là tập kiểm soát do đồ án xây dựng, mỗi ca chỉ chạy hai lần; đánh giá chỉnh sửa do một người chấm và chưa có user study mù nhiều người. Vì vậy nên khẳng định RoomFlow “tốt hơn các baseline được thử nghiệm trong các ca có ràng buộc khó”, không nên mở rộng thành tuyên bố vượt trội phổ quát.
