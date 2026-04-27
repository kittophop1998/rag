# Company RAG (FastAPI + LangChain + OpenAI + FAISS + LINE)

ระบบถาม-ตอบเอกสารภายในบริษัทแบบ Retrieval-Augmented Generation พร้อม **หน้าเว็บแชท** และ **Webhook LINE** ในตัว

```
ผู้ใช้  ───▶  Chat Web UI / LINE OA
                 │
                 ▼
       FastAPI  ──▶  RAG Chain (LangChain + OpenAI gpt-4o)
                          │
                          ▼
                 FAISS Vector DB ◀── Indexer (PDF จาก ./documents)
```

---

## Flow การทำงาน (เรียงตามลำดับ)

1. **Setup & Indexing**
   วางไฟล์ PDF ลงในโฟลเดอร์ `./documents` → ระบบจะ
   - โหลด PDF ทั้งหมดด้วย `PyPDFLoader`
   - แตกเป็นชิ้น (chunk) ด้วย `RecursiveCharacterTextSplitter`
   - แปลงเป็น vector ด้วย `OpenAIEmbeddings`
   - บันทึกลง **FAISS** ที่ `./faiss_index` เพื่อไม่ต้อง index ซ้ำทุกครั้ง

2. **RAG Logic**
   - รับคำถามจากผู้ใช้
   - ดึง top-3 chunks ที่เกี่ยวข้องที่สุดจาก FAISS
   - ส่งให้ `gpt-4o` ตอบ **โดยอ้างอิงเฉพาะ context ที่ให้เท่านั้น**
   - ถ้าตอบไม่ได้จาก context → ตอบ `"ไม่พบข้อมูลในเอกสารครับ"`

3. **LINE Webhook**
   - `POST /webhook` รับ event จาก LINE Messaging API
   - ตรวจ `X-Line-Signature` ด้วย HMAC-SHA256 (ใช้ `LINE_CHANNEL_SECRET`)
   - ดึงข้อความ → ส่งเข้า RAG Chain → ตอบกลับผ่าน LINE Reply API

---

## โครงสร้างโปรเจกต์

```
RAG/
├── app/
│   ├── config.py          # โหลด env vars / .env (pydantic-settings)
│   ├── indexer.py         # โหลด PDF + สร้าง / โหลด FAISS index
│   ├── rag.py             # RAG chain (retrieve + prompt + LLM)
│   ├── line_webhook.py    # /webhook LINE + ตรวจ signature
│   └── main.py            # FastAPI entrypoint, /api/chat, /api/reindex
├── static/                # หน้าเว็บแชท (HTML/CSS/JS)
├── documents/             # วางไฟล์ PDF ที่ต้องการให้ค้น
├── faiss_index/           # FAISS index ที่ระบบสร้างให้ (auto)
├── requirements.txt
├── .env.example
└── README.md
```

---

## ติดตั้ง & ใช้งาน

### 1) เตรียม environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# จากนั้นเปิด .env แล้วใส่ค่า OPENAI_API_KEY, LINE_CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN
```

### 2) วางไฟล์ PDF

วางไฟล์ PDF ทั้งหมดของบริษัทใน `./documents/` (มีโฟลเดอร์ย่อยได้)

### 3) รันเซิร์ฟเวอร์

```bash
uvicorn app.main:app --reload --port 8000
```

ครั้งแรกระบบจะ index เอกสารให้อัตโนมัติ จากนั้นเปิดเบราว์เซอร์ไปที่
**http://localhost:8000** เพื่อใช้หน้าเว็บแชท

### 4) สร้าง index ใหม่เมื่อมีเอกสารเพิ่ม

- กดปุ่ม **“สร้าง Index ใหม่”** ในหน้าเว็บ หรือ
- เรียก API: `POST /api/reindex`

---

## API ที่เปิดให้ใช้

| Method | Path             | คำอธิบาย                                               |
|--------|------------------|--------------------------------------------------------|
| GET    | `/`              | หน้าเว็บแชท                                            |
| POST   | `/api/chat`      | `{"question": "..."}` → `{"answer": "...", "sources":[]}` |
| POST   | `/api/reindex`   | สร้าง FAISS index ใหม่จาก `./documents`                |
| POST   | `/webhook`       | LINE Messaging API webhook (ตรวจ signature อัตโนมัติ)  |
| GET    | `/healthz`       | Health check                                           |
| GET    | `/docs`          | Swagger UI (จาก FastAPI)                               |

ตัวอย่าง:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "นโยบายลาพักร้อนของบริษัทเป็นอย่างไร?"}'
```

---

## เชื่อมต่อ LINE Messaging API

1. สร้าง **Messaging API channel** ใน [LINE Developers Console](https://developers.line.biz/console/)
2. คัดลอก **Channel secret** และออก **Channel access token (long-lived)** ใส่ลง `.env`
3. เปิดเซิร์ฟเวอร์ออกอินเทอร์เน็ต (เช่น `ngrok http 8000`)
4. ตั้ง **Webhook URL** ในคอนโซลเป็น `https://<your-domain>/webhook`
5. กด **Verify** ในคอนโซล แล้วเปิด **Use webhook** = ON
6. ปิด *Auto-reply messages* ใน LINE Official Account Manager

ทดสอบ: ส่งข้อความใน LINE OA → บอทจะตอบจากเอกสารใน `./documents`

---

## ปรับแต่งเพิ่ม

| ENV               | ค่าเริ่มต้น                 | คำอธิบาย                                  |
|-------------------|------------------------------|--------------------------------------------|
| `OPENAI_CHAT_MODEL` | `gpt-4o`                   | โมเดลสำหรับตอบ                              |
| `OPENAI_EMBED_MODEL`| `text-embedding-3-small`   | โมเดลสำหรับสร้าง embedding                  |
| `CHUNK_SIZE`        | `1000`                     | ขนาด chunk (ตัวอักษร)                       |
| `CHUNK_OVERLAP`     | `150`                      | overlap ระหว่าง chunk                       |
| `TOP_K`             | `3`                        | จำนวน chunk ที่ดึงมาเป็น context            |

---

## License

ใช้ภายในบริษัทเท่านั้น – Internal use only.
