# Untitled

# 📘 Dự án: Mô hình Transformer huấn luyện trên tập Tiny Shakespeare

## 🎯 Giới thiệu

Dự án này xây dựng và huấn luyện một **mô hình ngôn ngữ Transformer (Decoder-only)** trên tập dữ liệu **Tiny Shakespeare** – bộ văn bản gồm khoảng **1 triệu ký tự** trích từ các tác phẩm nổi tiếng của William Shakespeare (như *Hamlet*, *Macbeth*, *Julius Caesar*, *The Tempest*, v.v.).

Mục tiêu chính là **huấn luyện mô hình sinh văn bản** theo phong cách Shakespeare, thông qua việc:

- Token hóa dữ liệu bằng **SentencePiece (Byte Pair Encoding - BPE)**
- Thiết kế mô hình Transformer tự cài đặt bằng **PyTorch**
- Huấn luyện mô hình trên tập dữ liệu văn bản
- Sinh ra các đoạn văn mới có cấu trúc và ngữ điệu tương tự Shakespeare

---

## 📂 Dữ liệu sử dụng

### 🧾 Nguồn dữ liệu

Dữ liệu được tải trực tiếp từ:

```
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

```

Tập dữ liệu gồm:

- Khoảng **1,118,394 ký tự**
- Toàn bộ văn bản là tiếng Anh cổ, không có nhãn phân loại
- Được xử lý dưới dạng chuỗi ký tự liên tục (character-level text)

### 🔍 Đặc điểm dữ liệu

- Dữ liệu có **phong cách văn học cổ điển**, giàu cấu trúc ngôn ngữ và dấu câu.
- Không có phân chia theo câu chuyện – chỉ là chuỗi ký tự ghép nối.
- Được token hóa bằng **SentencePiece (BPE)** với `vocab_size = 8000`.
- Tệp mô hình tokenizer được lưu dưới dạng `tinyshakespeare.model` và `tinyshakespeare.vocab`.

---

## ⚙️ Quy trình xử lý và mô hình

### 1. **Tiền xử lý và Tokenization**

- Toàn bộ văn bản được nối lại thành một chuỗi duy nhất.
- Dùng `SentencePieceTrainer` để huấn luyện tokenizer với tham số:
    
    ```python
    vocab_size = 8000
    character_coverage = 1.0
    model_type = "bpe"
    
    ```
    
- Tạo lớp `TinyShakespeareData` kế thừa `torch.utils.data.Dataset`,
    
    có chức năng:
    
    - Mã hóa văn bản thành ID (`encode_as_ids`)
    - Thêm token BOS/EOS
    - Tự động padding batch bằng `collate_function`

### 2. **Mô hình Transformer**

- Kiến trúc **Decoder-only Transformer**, tương tự GPT nhỏ gọn.
- Các thành phần chính:
    - `PositionalEncoding` – mã hóa vị trí bằng hàm sin và cos.
    - `MultiHeadAttention` – tính toán attention giữa các token.
    - `FeedForward` – lớp tuyến tính và kích hoạt phi tuyến.
    - `TransformerBlock` – tổ hợp của attention + feedforward + residual connection.
    - `TransformerModel` – nhiều khối ghép nối + embedding đầu vào + lớp đầu ra softmax.

### 3. **Huấn luyện**

- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Huấn luyện trên tập dữ liệu được tokenize, batch-size nhỏ (do tính chất ngôn ngữ tự nhiên).
- Mục tiêu: mô hình học cách **dự đoán token tiếp theo** dựa trên chuỗi token trước đó.

---

## 🧠 Điểm nổi bật trong triển khai

- **Tự cài đặt mô hình Transformer từ đầu** (không dùng `nn.Transformer` của PyTorch).
- **Huấn luyện tokenizer BPE** riêng cho dữ liệu văn học Shakespeare.
- Thiết kế **Dataset + DataLoader** tùy chỉnh giúp đọc dữ liệu hiệu quả.
- Có thể **sinh văn bản mới** bằng cách:
    
    ```python
    model.generate(prompt="ROMEO:", max_length=100)
    
    ```
    
- Mã nguồn rõ ràng, tách biệt từng phần: tiền xử lý – mô hình – huấn luyện – sinh văn bản.

---

## 📊 Kết quả đạt được

- Mô hình sau khi huấn luyện có khả năng sinh ra các đoạn văn có cú pháp, dấu câu, và nhịp điệu **giống phong cách Shakespeare**.
- Các câu đầu ra có dạng:
    
    ```
    ROMEO:
    O, speak again, bright angel! for thou art
    As glorious to this night, being o’er my head...
    
    ```
    
- Tokenizer hoạt động hiệu quả, tái tạo chính xác cấu trúc ngôn ngữ.
- Transformer hội tụ ổn định sau một số epoch huấn luyện, loss giảm đều.

---

## 🚀 Cách chạy notebook

1. Cài đặt thư viện cần thiết:
    
    ```bash
    pip install torch datasets sentencepiece tqdm
    
    ```
    
2. Mở file notebook trong Jupyter hoặc Colab.
3. Chạy toàn bộ cell để:
    - Tải dữ liệu và huấn luyện tokenizer.
    - Xây dựng mô hình Transformer.
    - Huấn luyện mô hình.
    - Sinh thử văn bản mới.

---

## 🧾 Tổng kết

Dự án minh họa một ví dụ **điển hình về mô hình ngôn ngữ tự triển khai (custom Transformer)** trên một tập dữ liệu cổ điển – Tiny Shakespeare.

Toàn bộ quy trình từ **tokenization → modeling → training → text generation** đều được thực hiện thủ công, giúp hiểu rõ cơ chế hoạt động bên trong của mô hình Transformer.

Kết quả cho thấy, với cấu hình phù hợp và dữ liệu nhỏ, mô hình vẫn có thể **học được phong cách ngôn ngữ tự nhiên** và tạo ra văn bản có **tính văn học và mạch lạc cao**.