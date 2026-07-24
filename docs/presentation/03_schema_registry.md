# Layer 3 — Schema Registry + Avro

## What This Layer Does

Enforces a typed contract between every producer and consumer. Ensures that no malformed, renamed, or missing-field message can enter the pipeline — the error is caught at the source, before the broker accepts the message.

---

## The Problem Without a Schema Registry

```
WITHOUT SCHEMA ENFORCEMENT

news_producer.py             Spark consumer
                             (running fine)
   │
   │  Developer renames
   │  "timestamp" → "ts"
   │
   ▼
publishes {title, source, ts}   ────────────────►  consumer reads
                                                   "timestamp" field
                                                   → KeyError / null
                                                   → crashes hours later

                                    ⚠ Bad data already in the topic.
                                    ⚠ Crash happens downstream, not at source.
                                    ⚠ Potentially hours of bad messages to replay.
```

---

## The Solution: Fail at the Source

```
WITH SCHEMA REGISTRY

news_producer.py              Schema Registry       Spark consumer
      │                            │
      │  publish {title, ts}       │
      │                            │
      ▼                            │
  Avro Serializer                  │
      │                            │
      ├── validates against ───────►
      │   registered schema        │
      │                            │
      │   "timestamp" field        │
      │   not in schema            │
      ▼                            │
  SerializationError ◄─────────────┘
  raised HERE, before
  broker accepts message

  Bad data never enters the pipeline.
```

---

## Avro Schema Format

Schemas live in `schemas/` as `.avsc` files (JSON format).

**news_event_v1.avsc:**
```json
{
  "type": "record",
  "name": "NewsEvent",
  "namespace": "com.finplatform",
  "fields": [
    {"name": "title",  "type": "string"},
    {"name": "source", "type": "string"},
    {"name": "ts",     "type": {"type": "long", "logicalType": "timestamp-millis"}},
    {"name": "url",    "type": ["null", "string"], "default": null}
  ]
}
```

**price_tick_v1.avsc:**
```json
{
  "type": "record",
  "name": "PriceTick",
  "namespace": "com.finplatform",
  "fields": [
    {"name": "ticker", "type": "string"},
    {"name": "open",   "type": "float"},
    {"name": "high",   "type": "float"},
    {"name": "low",    "type": "float"},
    {"name": "close",  "type": "float"},
    {"name": "volume", "type": "long"},
    {"name": "ts",     "type": {"type": "long", "logicalType": "timestamp-millis"}}
  ]
}
```

---

## The Confluent Wire Format (5-byte prefix)

Every Avro-encoded Kafka message starts with 5 bytes that identify the schema:

```
┌───────────────────────────────────────────────────────────────────┐
│                     KAFKA MESSAGE BYTES                           │
│                                                                   │
│  Byte 0    │  Bytes 1–4      │  Bytes 5–N                        │
│  ──────────┼─────────────────┼──────────────────────────────     │
│  0x00      │  Schema ID      │  Avro-encoded payload             │
│  (magic)   │  (int32, big-   │                                   │
│            │   endian)       │                                   │
└───────────────────────────────────────────────────────────────────┘

Consumer reads bytes 1–4, calls:
  GET /subjects/news-raw-value/versions/{schema_id}
→ gets schema back
→ deserializes bytes 5–N automatically

No hardcoded StructType in the consumer. Schema is the single source of truth.
```

---

## Schema Evolution: v1 → v2

When the NLP pipeline is ready, `sentiment_score` is added to the news schema.

**Step 1 — Write the new schema (news_event_v2.avsc):**

```json
{
  "type": "record",
  "name": "NewsEvent",
  "fields": [
    {"name": "title",           "type": "string"},
    {"name": "source",          "type": "string"},
    {"name": "ts",              "type": {"type": "long", "logicalType": "timestamp-millis"}},
    {"name": "url",             "type": ["null", "string"], "default": null},
    {"name": "sentiment_score", "type": ["null", "float"],  "default": null}
  ]
}
```

**Step 2 — Pre-check compatibility (dry run):**

```bash
curl -X POST http://localhost:8081/compatibility/subjects/news-raw-value/versions/latest \
  -H "Content-Type: application/json" \
  -d "{\"schema\": $(cat schemas/news_event_v2.avsc | jq -Rs .)}"

# → {"is_compatible": true}
```

**Step 3 — Register if compatible:**

```bash
curl -X POST http://localhost:8081/subjects/news-raw-value/versions \
  -H "Content-Type: application/json" \
  -d "{\"schema\": $(cat schemas/news_event_v2.avsc | jq -Rs .)}"
```

---

## Compatibility Modes

```
BACKWARD (default — used here)
─────────────────────────────
New schema readable by old consumers.
→ New field added with default: old consumer ignores it, still works.
→ Safe to upgrade producers first.

         Old consumer                 New consumer
              │                            │
              │  reads v2 message          │  reads v2 message
              │  sentiment_score → null    │  sentiment_score → 0.82
              │  (default)                 │  (populated)
              ▼                            ▼
           WORKS                        WORKS


FORWARD
───────
Old schema readable by new consumers.
→ Safe to upgrade consumers first.

FULL
────
Both directions. Most restrictive.
Only allow adding/removing optional fields with defaults.
```

---

## What the Registry Rejects

| Change | Rejected? | Why |
|--------|-----------|-----|
| Remove a required field | YES | Old messages have it; new consumer can't deserialize |
| Change `string` → `int` | YES | Type mismatch on existing messages |
| Rename a field without alias | YES | Consumer looks up by name |
| Add optional field with default | NO | Old consumers ignore it; backward-compatible |
| Add required field (no default) | YES | Old messages missing it; breaks deserialization |

---

## Registry REST API Reference

```bash
# List all registered subjects
curl http://localhost:8081/subjects

# List versions for a subject
curl http://localhost:8081/subjects/news-raw-value/versions

# Get a specific version
curl http://localhost:8081/subjects/news-raw-value/versions/1

# Check compatibility before registering
curl -X POST http://localhost:8081/compatibility/subjects/news-raw-value/versions/latest \
  -d "{\"schema\": ...}"

# Register a new version
curl -X POST http://localhost:8081/subjects/news-raw-value/versions \
  -d "{\"schema\": ...}"
```

---

## Avro vs Alternatives

| Format | Schema enforced | Binary | Registry support | gRPC compatible |
|--------|----------------|--------|-----------------|----------------|
| **Avro** | YES (at serialization) | YES | YES (Confluent, Redpanda) | No |
| JSON | No | No | Optional (JSON Schema) | No |
| Protobuf | YES (at compile time) | YES | YES (Buf, Confluent) | YES |
| Parquet | YES (file-level) | YES | No (file format, not stream) | No |

**Why Avro here:** Best Kafka ecosystem integration. Compact binary. Schema ID embedded automatically. Protobuf is in the roadmap for gRPC-compatible serialization.

---

## Files in This Layer

| File | Role |
|------|------|
| [schemas/news_event_v1.avsc](../../schemas/news_event_v1.avsc) | News headline schema |
| [schemas/news_event_v2.avsc](../../schemas/news_event_v2.avsc) | Extended schema with sentiment_score |
| [schemas/price_tick_v1.avsc](../../schemas/price_tick_v1.avsc) | Price tick schema |
| [scripts/register_schemas.py](../../scripts/register_schemas.py) | Registers all schemas on startup |
