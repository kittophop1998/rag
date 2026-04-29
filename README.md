# Ruangthong RAG (FastAPI + LangChain + OpenAI + ChromaDB)

ระบบถาม-ตอบเอกสารภายในบริษัทแบบ Retrieval-Augmented Generation พร้อม **หน้าเว็บแชท**, **Text-to-SQL**, และ **การจัดการผู้ใช้**

```
ผู้ใช้  ───▶  Chat Web UI
                 │
                 ▼
       FastAPI  ──▶  RAG Chain (LangChain + OpenAI gpt-4o)
                          │
                          ▼
                 ChromaDB ◀── Indexer (PDF + URL sources จาก ./documents)
```

---

## Flow การทำงาน

1. **Setup & Indexing**
   วางไฟล์ PDF ลงในโฟลเดอร์ `./documents` → ระบบจะ
   - โหลด PDF ทั้งหมดด้วย `PyPDFLoader`
   - แตกเป็นชิ้น (chunk) ด้วย `RecursiveCharacterTextSplitter`
   - แปลงเป็น vector ด้วย `OpenAIEmbeddings`
   - บันทึกลง **ChromaDB** ที่ `./chroma`

2. **RAG Logic**
   - รับคำถามจากผู้ใช้
   - ดึง top-K chunks ที่เกี่ยวข้องที่สุดจาก ChromaDB
   - ส่งให้ `gpt-4o` ตอบโดยอ้างอิงเฉพาะ context ที่ให้เท่านั้น
   - ถ้าตอบไม่ได้จาก context → ตอบ `"ไม่พบข้อมูลในเอกสารครับ"`

3. **Text-to-SQL**
   - เชื่อมต่อฐานข้อมูล (MySQL, PostgreSQL, MongoDB) ผ่านหน้า Admin
   - แปลงคำถามภาษาไทย/อังกฤษเป็น SQL/Query โดยใช้ LLM + Vanna.ai
   - แสดงผลลัพธ์และสรุปคำตอบให้อ่านง่าย

---

## โครงสร้างโปรเจกต์

```
RAG/
├── app/
│   ├── main.py            # FastAPI entrypoint
│   ├── config.py          # โหลด env vars (pydantic-settings)
│   ├── constants.py       # ค่าคงที่ที่ใช้ทั่วโปรเจกต์
│   ├── auth.py            # Bearer token session auth
│   ├── user_store.py      # จัดการผู้ใช้ (users.json)
│   ├── indexer.py         # โหลด PDF/URL + สร้าง ChromaDB index
│   ├── rag.py             # RAG chain (retrieve + prompt + LLM streaming)
│   ├── text_to_sql.py     # Text-to-SQL engine
│   ├── db_inspector.py    # ดึง schema จากฐานข้อมูล
│   ├── db_query.py        # Execute SQL/MongoDB queries
│   ├── db_settings.py     # จัดการการเชื่อมต่อ DB (db_connections.json)
│   ├── url_sources.py     # จัดการ URL data sources
│   ├── vanna_engine.py    # Vanna.ai SQL training & generation
│   └── web_crawler.py     # Crawl URL sources สำหรับ indexing
├── static/                # หน้าเว็บ (HTML/CSS/JS)
├── documents/             # วางไฟล์ PDF ที่ต้องการให้ค้น
├── chroma/                # ChromaDB index (auto-generated)
├── users.json             # ข้อมูลผู้ใช้
├── db_connections.json    # การเชื่อมต่อฐานข้อมูล
├── url_sources.json       # URL data sources
├── vanna_trained.json     # สถานะการ train Vanna
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## ติดตั้ง & ใช้งาน

> ต้องการ **Python 3.11+**

### 1) เตรียม environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# เปิด .env แล้วใส่ OPENAI_API_KEY
```

#### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

> ถ้า PowerShell ขึ้น error เรื่อง execution policy:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

#### Windows (Command Prompt)

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
```

### 2) วางไฟล์ PDF

วางไฟล์ PDF ทั้งหมดของบริษัทใน `./documents/` (มีโฟลเดอร์ย่อยได้)

### 3) รันเซิร์ฟเวอร์

```bash
# macOS / Linux
./run.sh

# Windows (cmd)
run.bat

# Windows (PowerShell)
.\run.ps1

# หรือโดยตรง
uvicorn app.main:app --reload --port 8000
```

เปิดเบราว์เซอร์ไปที่ **http://localhost:8000**

### 4) รันด้วย Docker

```bash
docker-compose up --build
```

---

## API ที่เปิดให้ใช้

| Method | Path | คำอธิบาย |
|--------|------|-----------|
| GET | `/` | หน้าเว็บแชท |
| POST | `/api/chat` | RAG chat (SSE streaming) |
| POST | `/api/reindex` | สร้าง ChromaDB index ใหม่ |
| GET/POST | `/api/db-connections` | จัดการการเชื่อมต่อ DB |
| POST | `/api/text-to-sql` | แปลงคำถามเป็น SQL/Query |
| GET/POST | `/api/url-sources` | จัดการ URL data sources |
| GET/POST | `/api/users` | จัดการผู้ใช้ (admin) |
| GET | `/healthz` | Health check |
| GET | `/docs` | Swagger UI |

---

## ตัวแปร Environment

| ENV | ค่าเริ่มต้น | คำอธิบาย |
|-----|------------|-----------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | โมเดลสำหรับตอบ |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | โมเดลสำหรับ embedding |
| `DOCUMENTS_DIR` | `./documents` | โฟลเดอร์ PDF |
| `CHUNK_SIZE` | `1000` | ขนาด chunk (ตัวอักษร) |
| `CHUNK_OVERLAP` | `150` | overlap ระหว่าง chunk |
| `TOP_K` | `3` | จำนวน chunk ที่ดึงมาเป็น context |

---

## License

ใช้ภายในบริษัทเท่านั้น – Internal use only.
