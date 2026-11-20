# Async Embedding Pipeline

> 🇬🇧 English version: [README.md](README.md)

Pipeline cấp production, **có thể chạy tiếp sau khi bị gián đoạn (resumable)**. Pipeline biến
các file CSV dung lượng tuỳ ý thành vector embedding của OpenAI, lưu dưới dạng Apache Parquet:

```
CSV (đọc dạng stream) ─▶ làm sạch + khử trùng lặp SHA-256 ─▶ chia chunk theo token (tiktoken)
   ─▶ bỏ qua phần đã xong ─▶ gom batch theo ngân sách token
   ─▶ gọi OpenAI bất đồng bộ (semaphore + rate limit + retry)
   ─▶ checkpoint SQLite (ghi nguyên tử từng batch) ─▶ xuất Parquet
```

Mục tiêu thiết kế:
- Bộ nhớ có giới hạn, không phụ thuộc kích thước file.
- Không bao giờ trả tiền API hai lần cho cùng một nội dung.
- Sập giữa chừng vẫn chạy tiếp được an toàn.
- Gặp lỗi thì xử lý từ từ, không làm sập cả lượt chạy kéo dài hàng giờ.

---

## Bắt đầu nhanh

```bash
uv sync                                            # tạo .venv từ uv.lock
cp .env.example .env                               # sau đó điền OPENAI_API_KEY
uv run python data/generate_sample_csv.py --rows 100000   # dữ liệu mẫu "bẩn" -> data/raw/sample.csv

uv run embed-pipeline run --dry-run                # ước tính khối lượng/token, không gọi API
uv run embed-pipeline run                          # chạy đầy đủ (Ctrl+C bất cứ lúc nào; chạy lại để resume)
uv run embed-pipeline stats                        # số chunk trong checkpoint + số mục trong dead-letter
uv run embed-pipeline export                       # xuất lại Parquet từ checkpoint
```

Chạy test và linter:

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

### Tham chiếu CLI

```
embed-pipeline <lệnh> [--config PATH] [--log-level LEVEL] [--log-format text|json] [tuỳ chọn]

run     --input CSV --output PARQUET --checkpoint DB --model NAME --dimensions N
        --max-concurrency N --batch-size N --max-tokens N --overlap-tokens N
        --csv-chunksize N --dry-run --no-export --no-progress
export  --output PARQUET --checkpoint DB
stats   --checkpoint DB
```

Có thể đặt `--config`, `--log-level` và `--log-format` trước hoặc sau tên lệnh con.

| Mã thoát | Ý nghĩa |
|---|---|
| `0` | thành công |
| `1` | lỗi cấu hình / đầu vào (YAML sai, thiếu cột, thiếu file) |
| `2` | chạy xong nhưng có batch phải đưa vào dead-letter: sửa nguyên nhân rồi chạy lại |
| `3` | lỗi nghiêm trọng: sai API key, không đủ quyền, hết quota, hoặc circuit breaker ngắt |
| `130` | bị ngắt (Ctrl+C / `docker stop`): tiến độ đã lưu, chạy lại để tiếp tục |

### Docker

```bash
docker build -t async-embedding-pipeline .

# Tuỳ chọn: sinh dữ liệu mẫu vào ./data
docker run --rm -v "$PWD/data:/app/data" --entrypoint python \
  async-embedding-pipeline data/generate_sample_csv.py --rows 100000

docker run --rm -v "$PWD/data:/app/data" async-embedding-pipeline run --dry-run
docker run --rm --env-file .env -v "$PWD/data:/app/data" async-embedding-pipeline   # chạy đầy đủ
docker run --rm -v "$PWD/data:/app/data" async-embedding-pipeline stats
```

- **Mount thư mục `data/` làm volume**, vì checkpoint nằm ở đó. Khởi động lại container với
  cùng volume thì pipeline chạy tiếp từ chỗ đã dừng.
- **`docker stop` dừng êm.** Image đặt `STOPSIGNAL SIGINT`. Python chạy với vai trò PID 1 sẽ
  bỏ qua tín hiệu SIGTERM mặc định, dẫn tới bị SIGKILL sau 10 giây. Với SIGINT, lượt chạy được
  huỷ gọn gàng: các batch đã commit được giữ lại và container thoát với mã `130`.
- **Tái lập được và tự chứa.** Dependency lấy từ `uv.lock` (`--frozen`). Bảng mã `cl100k_base`
  được đóng sẵn vào image nên lúc chạy không phải tải. Tiến trình chạy bằng user không phải
  root, và log ở dạng JSON (`LOG_FORMAT=json`).
- **WSL2 chưa bật Docker Desktop integration.** Gọi trực tiếp file thực thi của Windows, ví dụ
  `/mnt/c/.../DockerDesktop/resources/bin/docker.exe`, hoặc bật integration trong Docker
  Desktop → Settings → Resources → WSL integration.

**Kịch bản đã kiểm chứng (20.000 dòng dữ liệu bẩn, 21.198 chunk, endpoint OpenAI giả lập
chèn khoảng 13% lỗi 429/500):**
- `docker stop` giữa chừng: thoát với mã `130` trong vòng 1 giây, đã lưu checkpoint 7.936 chunk.
- Chạy lại: bỏ qua đúng 7.936 chunk đó, tự retry 17 lần, xuất đủ 21.198 dòng.
- Chạy lần ba: 0 lời gọi API.
- API sập hoàn toàn: circuit breaker dừng lượt chạy sau 76 giây (mã `3`) với 10 mục DLQ,
  thay vì 166.

---

## Cấu trúc dự án

```
configs/pipeline_config.yaml   Tham số vận hành (model, batching, giới hạn, chính sách retry)
data/generate_sample_csv.py    Sinh CSV lớn, cố tình "bẩn", kết quả tái lập được
data/raw/                      CSV đầu vào
data/processed/                embeddings.parquet
data/checkpoints/              checkpoint.db (SQLite) + dead_letter.jsonl
src/schemas.py                 Pydantic: cấu hình + cấu trúc bản ghi, kiểm tra theo giới hạn API
src/cleaner.py                 Đọc CSV dạng stream, chuẩn hoá, khử trùng lặp bằng content hash
src/chunker.py                 Cửa sổ token có overlap; gom batch giới hạn theo số phần tử + token
src/checkpoint.py              Kho SQLite: các chunk đã hoàn thành kèm vector
src/embedder.py                AsyncOpenAI + Semaphore + token bucket + tenacity + DLQ + breaker
src/pipeline.py                Điều phối producer/consumer, thanh tiến trình tqdm, xuất Parquet
src/cli.py                     CLI, structured logging, mã thoát
tests/                         pytest + pytest-asyncio (giả lập OpenAI, test sập/resume)
```

Package được import dưới tên `src` (ví dụ `from src.pipeline import build_pipeline`), đúng
với cấu trúc thư mục yêu cầu.

---

## Kiến trúc

### Luồng dữ liệu và xử lý đồng thời

```
┌──────────────── worker thread ────────────────┐         ┌───────────── event loop ─────────────┐
│ pandas.read_csv(chunksize)                     │         │  N consumer task (N = concurrency)   │
│   → chuẩn hoá → khử trùng lặp SHA-256          │ asyncio │    embed_batch()                     │
│   → TokenChunker (512 token, overlap 50)       │  Queue  │      Semaphore → RateLimiter → API   │
│   → tra checkpoint (bỏ qua id đã xong)         │ ──────▶ │      tenacity retry / DLQ / breaker  │
│   → BatchBuilder (≤ phần tử, ≤ token)          │ có giới │    save_results() (SQLite, thread)   │
└────────────────────────────────────────────────┘   hạn   │    cập nhật tqdm                     │
                                                           └───────────────────────────────────────┘
```

- **Bộ nhớ có giới hạn.** CSV được đọc theo từng khối pandas, và producer là một generator
  chỉ sinh từng batch một. Hàng đợi giữa hai bên có giới hạn
  (`max_concurrency × queue_size_per_worker`), nên việc đọc không bao giờ chạy quá xa so với
  việc gọi API.
- **Event loop không bao giờ bị chặn.** Parse CSV và tokenise tốn CPU, nên producer chạy trong
  worker thread qua `asyncio.to_thread`. Các lần commit SQLite cũng vậy. Event loop chỉ lo HTTP.
- **Theo dõi tiến độ.** `tqdm.asyncio` hiển thị tốc độ chunk/giây và thời gian ước tính còn lại,
  cùng số token đã dùng, số lần retry, số batch lỗi và số chunk bỏ qua (đã xong từ trước). Tổng
  số tăng dần khi đọc stream và chính xác khi đọc xong. Log được chuyển qua tqdm để thanh tiến
  trình không bị vỡ. Thanh tiến trình chỉ hiện khi stderr là TTY, nên Docker và CI nhận log JSON
  sạch.

### Làm sạch (`cleaner.py`)

1. Ghép các cột trong `text_columns`. Thiếu cột thì báo lỗi ngay khi bắt đầu.
2. Mỗi giá trị được chuẩn hoá Unicode NFKC. Ký tự điều khiển C0/C1, ký tự zero-width, BOM, ký tự
   phân dòng và `U+FFFD` bị loại bỏ, khoảng trắng thừa được gộp lại.
3. Loại bỏ dòng rỗng hoặc ngắn hơn `min_text_chars`. Dòng thiếu id được gán id `row-<n>`.
4. Khử trùng lặp bằng **SHA-256 của văn bản đã chuẩn hoá**: bản xuất hiện đầu tiên được giữ, nên
   id không đổi giữa các lần chạy. Bộ chỉ mục lưu digest 32 byte, khoảng 1 GB cho mỗi 10 triệu
   dòng duy nhất.

### Chia chunk và gom batch (`chunker.py`)

- `cl100k_base` cho số token chính xác. Văn bản không quá `max_tokens` được giữ nguyên. Văn bản
  dài hơn được cắt thành các cửa sổ `max_tokens` token, bắt đầu cách nhau
  `max_tokens − overlap_tokens` token, nên hai chunk liên tiếp có chung `overlap_tokens` token
  ngữ cảnh.
- **An toàn với ký tự nhiều byte.** Mép cửa sổ được dịch tối đa 3 token để rơi đúng ranh giới
  ký tự UTF-8, nên tiếng Việt, chữ CJK và emoji không bao giờ bị giải mã thành `U+FFFD`. Mỗi
  chunk được đếm lại token sau khi giải mã để bảo đảm không vượt `max_tokens`.
- Batch bị giới hạn **đồng thời** bởi `max_items` và `max_tokens_per_batch`. Phần kiểm tra cấu
  hình áp các giới hạn cứng của API: 8.191 token mỗi input, 2.048 input và 300k token mỗi
  request.

---

## Khả năng chịu lỗi

| Sự cố | Cơ chế | Kết quả |
|---|---|---|
| Sập, `kill`, Ctrl+C, `docker stop`, mất mạng | Mỗi batch thành công được commit **kèm vector** trong một transaction SQLite (WAL). Khi khởi động, các chunk có id đã nằm trong checkpoint sẽ được bỏ qua. | Tối đa chỉ mất các batch đang chạy dở, và chúng được gửi lại. Không bao giờ trả tiền hai lần cho phần đã xong. |
| Batch bị gửi lại (sập giữa lúc nhận phản hồi API và lúc commit) | `INSERT OR IGNORE` theo `(chunk_id, model_key)` | Idempotent: không sinh bản trùng. |
| 429 / timeout / lỗi kết nối / 408 / 409 / 5xx | `tenacity`: exponential backoff với **full jitter**, tối đa `max_wait_seconds`, không bao giờ ngắn hơn `Retry-After` / `retry-after-ms` | Lỗi tạm thời được tự xử lý. Jitter giúp các worker không retry dồn cùng một lúc. |
| Lưu lượng dồn dập dễ gây 429 | `asyncio.Semaphore(max_concurrency)` cộng **token bucket** phía client cho số request/phút và số token/phút | Hiếm khi chạm rate limit ngay từ đầu. |
| Batch vẫn lỗi sau khi hết số lần retry, hoặc lỗi không đáng retry (400, phản hồi sai cấu trúc) | Ghi thêm (có fsync) vào **`dead_letter.jsonl`**, gồm lỗi, mã trạng thái, số lần thử, run id và văn bản của các chunk | Pipeline chạy tiếp và thoát với mã `2`. Chunk lỗi không được ghi vào checkpoint, nên **chạy lại là tự retry**. |
| Sự cố diện rộng (API hoặc mạng sập lâu) | **Circuit breaker**: khi `retry.max_consecutive_failures` (mặc định 10) batch *liên tiếp* bị đưa vào dead-letter thì dừng lượt chạy. Một batch thành công sẽ đặt lại bộ đếm. | Báo lỗi trong khoảng một phút (mã `3`), thay vì cày hết dữ liệu rồi đẩy toàn bộ vào DLQ. Chạy lại khi API hoạt động trở lại. |
| Sai API key / không đủ quyền / `insufficient_quota` | `FatalEmbeddingError` | Dừng ngay (mã `3`) thay vì đưa từng batch vào dead-letter. |
| Sập trong lúc xuất Parquet | Ghi vào file tạm rồi `os.replace` | Người đọc không bao giờ thấy file dở dang. |

Cơ chế retry có sẵn của SDK bị tắt (`max_retries=0`), nên tenacity là chính sách retry duy nhất.

### Khoá idempotency

- `content_hash = sha256(văn_bản_đã_chuẩn_hoá)`
- `chunk_id = sha256(content_hash : chunk_index : sha256(văn_bản_chunk))[:32]`
- Checkpoint được tách theo `model_key = "<model>@<dimensions|native>"`.

Cùng đầu vào và cùng tham số thì luôn cho cùng id. Đổi `max_tokens` hoặc `overlap_tokens` sẽ
sinh id mới, còn đổi model sẽ dùng một không gian checkpoint mới, nên checkpoint cũ không bao
giờ bị áp nhầm vào kết quả.

### Đánh đổi về độ bền dữ liệu

Checkpoint dùng `synchronous=NORMAL` ở chế độ WAL. Các batch đã commit vẫn còn nguyên khi tiến
trình bị sập. Nếu mất điện đột ngột, vài commit cuối có thể bị mất, và các chunk đó chỉ đơn giản
được embed lại ở lần chạy sau.

---

## Cấu hình

`configs/pipeline_config.yaml` là nguồn cấu hình chính. Giá trị được áp theo thứ tự sau, mục
sau ghi đè mục trước:

1. YAML
2. Biến môi trường / `.env`: `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `MAX_CONCURRENCY`,
   `REQUESTS_PER_MINUTE`, `TOKENS_PER_MINUTE`, `MAX_RETRY_ATTEMPTS`, `BATCH_MAX_ITEMS`
3. Cờ CLI

Thông tin xác thực (`OPENAI_API_KEY`, tuỳ chọn `OPENAI_BASE_URL`) chỉ được đọc từ biến môi
trường. Chúng không bao giờ nằm trong đối tượng cấu hình và không bao giờ bị ghi ra log. Khoá
YAML lạ bị từ chối, nên gõ sai tên khoá sẽ báo lỗi ngay khi khởi động thay vì bị bỏ qua âm thầm.

**Tinh chỉnh:**
- Đặt `requests_per_minute` và `tokens_per_minute` thấp hơn một chút so với giới hạn tier của
  tổ chức bạn.
- Tăng `max_concurrency` cho tới khi token bucket, chứ không phải độ trễ, là điểm nghẽn.
- Chạy `--dry-run` trước: `estimated_tokens` × giá mỗi token của model cho ra chi phí.

---

## Kết quả đầu ra

`data/processed/embeddings.parquet` (nén zstd, ghi theo row group):

| cột | kiểu |
|---|---|
| `record_id` | string |
| `chunk_id` | string |
| `chunk_index` | int32 |
| `text` | string |
| `token_count` | int32 |
| `embedding` | list<float32> |

Metadata của file có chứa `model_key`.

```python
import pyarrow.parquet as pq
import numpy as np

table = pq.read_table("data/processed/embeddings.parquet")
matrix = np.stack(table.column("embedding").to_numpy(zero_copy_only=False))
```

---

## Kiểm thử

| File | Nội dung kiểm tra |
|---|---|
| `test_cleaner_chunker.py` | Chuẩn hoá, loại NaN/rỗng, khử trùng lặp SHA-256 qua nhiều khối CSV, thiếu cột; giới hạn token, độ chính xác của overlap, an toàn ký tự nhiều byte, id tất định; gom batch theo số phần tử và token; kiểm tra cấu hình theo giới hạn API |
| `test_checkpoint.py` | Dữ liệu còn sau khi mở lại, insert idempotent, tách biệt theo model, rollback khi ghi lỗi; **phục hồi end-to-end sau khi sập** (sập đột ngột giữa chừng, lần sau chỉ gửi phần còn lại, lần thứ ba không gọi API), phục hồi sau **Ctrl+C**, batch trong dead-letter được retry ở lần sau, **sự cố API → circuit breaker → resume** |
| `test_embedder.py` | Client OpenAI giả lập bằng `AsyncMock`: retry với 429/500/502/503/timeout/lỗi kết nối, DLQ khi hết số lần retry, không retry với 400, dừng với 401/403/hết quota, xử lý `Retry-After`, kiểm tra thứ tự phản hồi, giới hạn đồng thời bằng semaphore, cô lập lỗi, circuit breaker (ngắt, đặt lại, tắt được), hành vi token bucket |
| `test_cli.py` | Tuỳ chọn chung đặt trước và sau lệnh con (dạng `CMD` của Docker), mã thoát cho dry run, thiếu API key và cấu hình sai |

Các mock dùng lớp exception thật của `openai`, dựng từ phản hồi HTTP thật, và model
`CreateEmbeddingResponse` thật, nên khi giao diện SDK thay đổi thì test sẽ báo lỗi.

---

## Giới hạn và hướng mở rộng

- Bộ chỉ mục khử trùng lặp nằm trong RAM. Với hàng tỷ dòng, hãy thay bằng tập lưu trên đĩa
  (ví dụ một bảng SQLite) hoặc Bloom filter.
- Khi các bản ghi trùng nhau, đầu ra chỉ có `record_id` của bản xuất hiện đầu tiên. Nếu hệ
  thống phía sau cần đủ mọi id, hãy lưu thêm một bảng alias.
- Checkpoint là một file SQLite duy nhất, giả định mỗi checkpoint chỉ có một tiến trình
  pipeline. Muốn mở rộng theo chiều ngang thì chia nhỏ đầu vào và dùng một checkpoint cho mỗi
  phần.
- Checkpoint lưu cả văn bản của chunk nên có dung lượng xấp xỉ file Parquet. Đó là cái giá để
  luôn dựng lại được file xuất từ checkpoint.
