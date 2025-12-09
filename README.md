# Subtitle Sync Tool (v2.0)

**Mô tả ngắn:**  
Công cụ đồng bộ subtitle tiếng Việt theo timing của subtitle tiếng Anh. Hỗ trợ `.srt` và `.ass`. Matching chính bằng Sentence Transformers, fallback RapidFuzz. Phiên bản 2.0 bổ sung context-matching, time-aware penalty, bảo toàn thứ tự và phát hiện câu bị tách.

---

# Tính năng chính

- Hỗ trợ `.srt` và `.ass` (SSA/ASS).
- Matching dựa trên embedding (SentenceTransformer) hoặc RapidFuzz nếu không có embedding.
- Context matching (dùng câu trước + câu sau để tăng độ chính xác).
- Time-aware matching: phạt theo khác biệt thời gian.
- Order preservation: ngăn chặn đảo dòng trong cùng câu.
- Detect & fix split-sentences (câu bị tách thành nhiều dòng).
- Tự động detect và bù offset thời gian (global / piecewise đơn giản).

---

# Yêu cầu

- Python 3.8+
- (Khuyến nghị) `pip install sentence-transformers numpy rapidfuzz`
  - Nếu không cài `sentence-transformers`, tool vẫn chạy với RapidFuzz (ít chính xác hơn).
- File phụ đề `.srt` hoặc `.ass` cho cả EN và VI.

---

# Cài đặt nhanh

```bash
# cài dependencies (nếu muốn embedding)
pip install sentence-transformers numpy rapidfuzz
```

---

# Cách dùng (CLI)

```bash
python ghep_sub.py -e english.srt -v vietnamese.srt -o output.srt
# ví dụ ASS
python ghep_sub.py -e english.ass -v vietnamese.ass -o output.ass --mode loose
```

**Options quan trọng**

- `-e, --english` : file subtitle tiếng Anh (timing chuẩn).
- `-v, --vietnamese`: file subtitle tiếng Việt (cần sync).
- `-o, --output` : file output.
- `-m, --mode` : `strict|loose|auto` (mặc định `auto`).
- `--context-weight` : trọng số context (0-1).
- `--time-penalty` : penalty theo giây.
- `--order-penalty` : penalty khi vi phạm thứ tự.

---

# Các tham số cấu hình chính (có thể chỉnh trong mã hoặc qua CLI)

- `MIN_FUZZ_SIMILARITY = 60`
- `MIN_COSINE_SIMILARITY = 0.5`
- `CONTEXT_WEIGHT = 0.7`
- `TIME_PENALTY_FACTOR = 0.05` (penalty / giây)
- `MAX_TIME_PENALTY = 0.3`
- `ORDER_PENALTY = 0.4`
- `SEARCH_WINDOW_SECONDS = 5.0`
- `ENABLE_PRE_SPLITTING = False` (bật nếu muốn tách dòng dài trước khi match)

Chỉnh các giá trị này khi bạn thấy kết quả quá “nhát” (match ít) hoặc quá “thoáng” (match nhầm).

---

# Quy trình hoạt động (tóm tắt)

1. Detect format file (SRT/ASS).
2. Parse EN & VI.
3. (Tùy chọn) Pre-splitting các dòng dài.
4. Detect offset thời gian (global / piecewise) và áp dụng.
5. Precompute context/embeddings (nếu có model).
6. Với mỗi EN entry: tìm best VI candidate trong cửa sổ thời gian, tính similarity, trừ penalties (time/order/index-jump/duration).
7. Post-process: sửa lỗi đảo dòng trong trường hợp split sentences.
8. Export sang SRT hoặc ASS (dùng timing của EN).

---

# Mẹo tối ưu & troubleshooting nhanh

- Nếu không có model `sentence-transformers`, cài thêm để tăng chất lượng: `pip install sentence-transformers`.
- Nếu kết quả nhiều `[MISSING]`, thử:

  - tăng `SEARCH_WINDOW_SECONDS`, hoặc
  - giảm `CONTEXT_WEIGHT`, hoặc
  - chạy detect offset manual nếu sub VI bị trễ/sớm lớn.

- Nếu gặp đảo dòng trong các đoạn nhiều câu, bật `ENABLE_PRE_SPLITTING = True` hoặc điều chỉnh `SPLIT_ON_PUNCTUATION`.
- Sub ASS: nếu muốn giữ style/header, truyền `--output` là `.ass` và dùng EN file làm `template_path`.

---

# Hạn chế hiện tại

- Piecewise offset hiện chỉ đơn giản (dựa trên vài dòng đầu nên cần mồi vài dòng đầu giống nhau); video có drift phức tạp có thể cần xử lý riêng.
- Nếu VI thiếu nhiều dòng so với EN (hoặc ngược lại), index-jump penalty có thể cần điều chỉnh để tránh mất match hợp lệ.
- Không xử lý auto-merge các câu VI ngắn thành câu dài tự động — đôi khi cần hậu xử lý thủ công.
- Không tự dịch các line MISSING, cần thêm quy trình dịch tự động vào sau.

**Đến Version 2.5 thì tỷ lệ chính xác đã lên đến trên 95%, có thể dùng Sub được luôn. Vài chỗ MISSING có thể "nhắm mắt" bỏ qua hoặc dịch nhanh thủ công và tận hưởng.**
