# Backend: private RAG API with free-AI fallback

### Windows (PowerShell):
```powershell
# 1. Virtual environment create karein
python -m venv .venv

# 2. Virtual environment activate karein
.\.venv\Scripts\Activate.ps1
# (Agar script restriction error aaye, toh seedha .venv ka python use karein:)
# .\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Dependencies install karein
pip install -r requirements.txt

# 4. Backend run karein
python -m uvicorn app.main:app --reload --port 8000
```

### Linux / Mac (Bash):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs (click **Authorize**, paste a key).

## How a question is answered

1. A follow-up is rewritten into a standalone question (only when there is history).
2. **Saved answer** (cache): only if the question means the same (≥ 0.97 similar *and* differs by filler words
   only, so "phone of Ravi" never returns Rahul's answer). Scoped per data group.
3. **Your data**: hybrid search (vectors + keywords, fused with RRF), optional re-ranker. IDs, phone numbers and
   emails in the question count as an exact match. A Hindi/Hinglish miss is retried once with an English rewrite.
4. If it matches well enough, an AI writes the answer **from those passages only**, streamed. It replies
   `NOT_FOUND` when the passages are irrelevant (never shown to the user) and we go to step 5.
5. **General AI** (if allowed): only the question is sent. If no AI is reachable at all, the best matching passages
   are returned instead of an error.

The match level is **measured automatically** from your data (`min_score_mode=auto`): unrelated questions in
English/Hindi/Hinglish set the noise floor, real passages set the signal, and the cut-off sits between. It is
re-measured when the data grows by 20%.

## Who can call what

| Caller | How | Can |
|---|---|---|
| Admin (this UI) | `ADMIN_TOKEN` | everything |
| App key `rk_...` | created in Connect apps / `POST /keys` | its scopes, its data groups (namespaces), its rate limit |

Scopes: `chat` (chat + history), `search`, `ingest` (add/update/delete data). Wrong-key attempts are blocked per IP
after `AUTH_FAIL_LIMIT`/minute. Limits and provider cool-downs live in MongoDB, so they hold across workers.

## Endpoints

| Purpose | Endpoint | Who |
|---|---|---|
| Chat (JSON / SSE stream) | `POST /chat`, `POST /chat/stream` | chat |
| Chat history | `GET /conversations`, `GET/PATCH/DELETE /conversations/{id}` | chat |
| Search only | `POST /search` | search |
| Push records from your systems | `POST /records`, `POST /records/delete`, `DELETE /records/{collection}/{id}` | ingest |
| Text / file (pdf docx txt md html csv tsv xlsx json jsonl) | `POST /ingest/text`, `POST /ingest/file` | ingest |
| Job progress | `GET /jobs/{id}` | ingest |
| Browse / delete | `GET /documents`, `DELETE /documents/{id}`, `POST /documents/delete-source` | ingest |
| Data groups | `GET /namespaces` | search |
| Instant sync ping | `POST /sync/sources/{name}/notify` | ingest |
