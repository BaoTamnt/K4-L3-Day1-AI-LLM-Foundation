# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature thấp, model trả lời ổn định và lặp lại phong cách tương tự; khi temperature tăng, phản hồi trở nên đa dạng hơn, có thể thay đổi câu chữ và mức độ sáng tạo. Ở mức cao, kết quả dễ lệch khỏi ý chính hoặc mất tính nhất quán hơn, dù có vẻ “sống động” hơn. Nói ngắn gọn: temperature thấp cho độ chắc chắn, temperature cao cho sự ngẫu nhiên và sáng tạo.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt khoảng 0.2–0.5 cho chatbot hỗ trợ khách hàng. Mức này vẫn giữ câu trả lời tự nhiên và đa dạng đủ để phản hồi tốt, nhưng không quá ngẫu nhiên đến mức sai lệch thông tin hoặc lặp lại cách diễn đạt bất ổn. Với ứng dụng hỗ trợ khách hàng, độ chính xác và tính nhất quán quan trọng hơn độ sáng tạo.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với 10.000 × 3 × 350 = 10,5 triệu token đầu ra/ngày, chi phí GPT-4o ước tính là 10,5 triệu / 1.000 × 0,010 = ~105 USD/ngày; GPT-4o-mini là 10,5 triệu / 1.000 × 0,0006 = ~6,3 USD/ngày. Tỷ lệ chi phí là khoảng 16,7 lần, tức GPT-4o đắt hơn mini khoảng 17x. GPT-4o xứng đáng khi cần suy luận phức tạp, trả lời chuyên sâu, hoặc quality cao như hỗ trợ kỹ thuật, phân tích dữ liệu, soạn văn bản chuyên nghiệp. GPT-4o-mini phù hợp cho triage, tóm tắt nhanh, hỗ trợ khách hàng đơn giản, hoặc xử lý khối lượng lớn với chi phí thấp.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi sẽ khác rõ về độ dài, từ vựng và ví dụ: phiên bản giáo viên ngắn, dễ hiểu, dùng từ đơn giản và ví dụ gần gũi như sổ cái hoặc danh sách dữ liệu; phiên bản chuyên gia dài hơn, dùng thuật ngữ như nonce, hash, distributed ledger, và giải thích chặt chẽ hơn. System prompt định hình persona, tông giọng và mức độ chuyên sâu của câu trả lời. Nói cách khác, prompt không chỉ thay đổi nội dung mà còn thay đổi cách model suy nghĩ và trình bày thông tin.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với cùng khoảng 100 từ, `số từ / 0.75` cho ra khoảng 133 token ước lượng, còn tiktoken cho số token thường cao hơn hoặc khác đáng kể tùy đoạn văn. Chênh lệch thường khoảng 20–40% vì tokenizer tiếng Việt không chia từ theo kiểu đơn vị rõ ràng như tiếng Anh; ký tự dấu, từ ghép, và cách tách từ của mô hình tokenization làm tăng số token. Vì vậy cùng độ dài, tiếng Việt thường cần nhiều token hơn để biểu diễn cùng một ý nghĩa.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi người dùng đang chat trực tiếp, chờ phản hồi dài hoặc cần cảm giác “đang suy nghĩ” ngay lập tức. Nó giúp UI phản hồi từng đoạn, giảm cảm giác chờ đợi và nâng trải nghiệm tương tác. Ngược lại, non-streaming phù hợp hơn khi bạn cần toàn bộ output trước khi hiển thị, ví dụ như tạo báo cáo, sinh tài liệu, hoặc xử lý batch mà không cần hiển thị liên tục.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp phân tán thời điểm retry theo cấp số nhân, giảm khả năng nhiều client cùng tấn công API tại một thời điểm. Khi server đang quá tải, delay cố định cùng lúc sẽ tạo ra “thundering herd”: hàng nghìn request đồng loạt retry sau 1 giây, làm tải tăng thêm và khiến hệ thống càng dễ sụp. Với backoff, các client nhảy đà khác nhau, giúp hệ thống phục hồi ổn định hơn và ít bị quá tải do retry đồng loạt.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona là “trợ giảng thân thiện của khóa AI”. System prompt ví dụ: “Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.” Từ “thân thiện” giúp model có giọng điệu hỗ trợ, gần gũi và không khô khan; “trả lời ngắn gọn” hạn chế output quá dài, phù hợp với trải nghiệm chat trong terminal. “bằng tiếng Việt” đảm bảo câu trả lời đồng nhất với người dùng và tránh lẫn lẫn tiếng Anh.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là lịch sử hội thoại ngắn, chỉ giữ 3 lượt gần nhất nên trợ lý không có bộ nhớ dài hạn và dễ quên ngữ cảnh cũ. Một cải thiện cụ thể là thêm “memory layer”: sau mỗi lượt, tóm tắt lại thông tin quan trọng và lưu vào một bản ghi ngắn, rồi khi người dùng hỏi tiếp, truy xuất các ghi nhớ liên quan bằng embedding hoặc từ khóa. Cách triển khai đơn giản là lưu một danh sách `memory` và ghép vào `messages` như một phần “system” hoặc “assistant summary”, cùng với lịch sử gần nhất để giữ cả ngữ cảnh hiện tại và thông tin quan trọng từ trước.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
