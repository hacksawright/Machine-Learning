# 🦙 Fine-tuning Decoder on Alpaca Dataset

## 📘 Tổng quan

Dự án này thực hiện **fine-tuning lại một mô hình ngôn ngữ (decoder)** đã được **huấn luyện ban đầu trên tập dữ liệu Tiny Shakespeare**, nhằm giúp mô hình học được **khả năng phản hồi theo kiểu hội thoại (instruction-following)** từ **tập dữ liệu Alpaca**.

Kết quả mong đợi là mô hình có thể trả lời câu hỏi, làm theo yêu cầu, và phản hồi tự nhiên như một chatbot huấn luyện theo hướng dẫn.

---

## ⚙️ Cấu trúc Notebook

Quy trình trong file `finetuning-decoder.ipynb` bao gồm các bước chính:

### 1️⃣ Chuẩn bị dữ liệu Alpaca

```python
from datasets import load_dataset

alpaca_dataset = load_dataset("tatsu-lab/alpaca")
print(alpaca_dataset)

```

- Tập dữ liệu gồm ~52,000 mẫu.
- Mỗi mẫu chứa:
    - `instruction`: yêu cầu từ người dùng
    - `input`: ngữ cảnh (có thể rỗng)
    - `output`: phản hồi tương ứng
- Dữ liệu được định dạng lại theo cấu trúc chuẩn phục vụ huấn luyện chatbot.

Ví dụ:

```
### User Instruction:
Give three tips for staying healthy.

### Assistant Response:
1. Eat a balanced diet...
2. Exercise regularly...
3. Get enough sleep...

```

---

### 2️⃣ Tiền xử lý và định dạng

Hàm định dạng dữ liệu:

```python
def format_alpaca_to_chatbot_text(example):
    instruction = example['instruction']
    input_text = example['input']
    output_text = example['output']

    # --- KHUÔN MẪU TẠO CHUỖI ĐỂ FINETUNE LLM ---
    # Sử dụng định dạng tiêu chuẩn (Standard Prompting Format)
    
    if input_text:
        # Trường hợp có Input/Ngữ cảnh
        text = (
            f"### User Instruction:\n{instruction}\n\n"
            f"### Context Input:\n{input_text}\n\n"
            f"### Assistant Response:\n{output_text}"
        )
    else:
        # Trường hợp chỉ có Instruction (thường thấy ở Alpaca)
        text = (
            f"### User Instruction:\n{instruction}\n\n"
            f"### Assistant Response:\n{output_text}"
        )
        
    return {"text": text}
```

- Chuẩn hóa mỗi mẫu thành đoạn văn bản hoàn chỉnh theo format chuẩn của LLM instruction-tuning.
- Kết quả được lưu lại trong trường `"text"` để dùng cho tokenizer.

---

### 3️⃣ Sử dụng tokenizer của mô hình gốc (Tiny Shakespeare)

- Tokenizer gốc được giữ nguyên để đảm bảo tính tương thích giữa mô hình cũ và mới.
    
    ```python
    from transformers import LlamaTokenizer
    
    tokenizer = LlamaTokenizer(
        vocab_file="/kaggle/input/final2-data/tinyshakespeare.model",
        sp_model_file="/kaggle/input/final2-data/tinyshakespeare.model",
        unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="</s>"
    )
    
    ```
    
- Điều này giúp fine-tuning chỉ tập trung điều chỉnh **trọng số của mô hình**, không làm thay đổi từ vựng ban đầu.

---

### 4️⃣ Fine-tuning mô hình

- Dữ liệu đã được token hóa và đóng gói vào `DataLoader`.
- Mô hình decoder (đã pretrain trên Tiny Shakespeare) được load lên CPU hoặc GPU (tuỳ môi trường).
- Thực hiện fine-tuning với hyperparameter cơ bản:
    - Learning rate nhỏ (vì đây là fine-tune, không phải train từ đầu).
    - Batch size nhỏ vừa để tránh quá tải GPU.
    - Epochs từ 3.

---

### 5️⃣ Đánh giá & sinh văn bản thử nghiệm

Sau khi huấn luyện xong, mô hình được dùng để sinh thử phản hồi với một prompt mới:

```
### Instruction:
Explain what Artificial Intelligence is.

### Input:
None

### Response:
AI is its keyworks, and translation. AI uses in automate data and the user service that automate data, providing for targets to accuracy, and decisions. AI can help users with other handwords, allowing accuracy, and more accuracy, and to their data, and analyze and data and ac
```

Kết quả cho thấy mô hình đã học cách sinh văn bản tự nhiên hơn, không bị lặp, và có khả năng trả lời theo dạng “Assistant Response”.

---

## 🧠 Ý nghĩa của dự án

- **Model gốc:** học ngữ pháp, cấu trúc câu, và phong cách ngôn ngữ tự nhiên từ *Tiny Shakespeare*.
- **Fine-tuning với Alpaca:** giúp mô hình chuyển từ viết văn bản theo phong cách thơ kịch của tập dữ liệu tinyshakespeare sang “trả lời theo hướng dẫn” (instruction-following).
- **Kết quả:** mô hình sau fine-tuning vừa giữ khả năng viết mạch lạc, vừa có thể phản hồi như chatbot thông minh.