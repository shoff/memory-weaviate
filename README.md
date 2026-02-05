# memory-weaviate

Weaviate-backed long-term vector memory plugin for Clawdbot.

## Features

- **Hybrid search** - combines vector similarity + keyword (BM25) for best recall
- **Auto-recall** - automatically injects relevant memories before each conversation
- **Auto-capture** - mines conversations for important info and stores it
- **Sensitivity filtering** - sensitive memories are hidden in group/shared contexts, only shown in DMs
- **Deduplication** - won't store near-identical memories
- **GDPR-friendly** - delete any memory by ID or search
- **LLM-powered extraction** - uses a fast model to intelligently decide what's worth remembering, with proper categorization, importance scoring, and sensitivity detection
- **Two embedding modes:**
  - `openai` - you provide embeddings via OpenAI API (recommended)
  - `weaviate` - use Weaviate's built-in text2vec-openai module

## Quick Start

### 1. Start Weaviate

```bash
cd extensions/memory-weaviate
docker compose up -d
```

This starts Weaviate with a local `text2vec-transformers` sidecar (sentence-transformers/all-MiniLM-L6-v2). No API keys needed for embeddings.

### 2. Install dependencies

```bash
cd extensions/memory-weaviate
npm install
```

### 3. Configure Clawdbot

#### Fully Local (Weaviate embeddings + Ollama extraction)

Zero API costs. All inference runs on your machine.

```json
{
  "plugins": {
    "slots": {
      "memory": "memory-weaviate"
    },
    "load": {
      "paths": ["./extensions/memory-weaviate"]
    },
    "entries": {
      "memory-weaviate": {
        "enabled": true,
        "config": {
          "weaviate": {
            "url": "http://localhost:8080"
          },
          "embedding": {
            "provider": "weaviate"
          },
          "extraction": {
            "baseUrl": "http://localhost:11434/v1",
            "model": "llama3.2"
          },
          "autoCapture": true,
          "autoRecall": true
        }
      }
    }
  }
}
```

#### Fully Local (Weaviate embeddings + LM Studio extraction)

```json
{
  "extraction": {
    "baseUrl": "http://localhost:1234/v1",
    "model": "qwen2.5-coder-7b"
  }
}
```

#### Cloud (OpenAI embeddings + OpenAI extraction)

```json
{
  "plugins": {
    "slots": {
      "memory": "memory-weaviate"
    },
    "load": {
      "paths": ["./extensions/memory-weaviate"]
    },
    "entries": {
      "memory-weaviate": {
        "enabled": true,
        "config": {
          "weaviate": {
            "url": "http://localhost:8080"
          },
          "embedding": {
            "provider": "openai",
            "apiKey": "${OPENAI_API_KEY}",
            "model": "text-embedding-3-small"
          },
          "extraction": {
            "model": "gpt-5-nano"
          },
          "autoCapture": true,
          "autoRecall": true
        }
      }
    }
  }
}
```

When using OpenAI for both embedding and extraction, the extraction API key falls back to `embedding.apiKey` automatically.

### 4. Restart Clawdbot

```bash
clawdbot gateway restart
```

## Sensitivity Filtering

Memories can be marked as **sensitive** to control where they appear. This prevents private information from being surfaced in group chats or shared contexts.

### How it works

1. **Auto-capture** detects sensitive content using LLM-based classification during extraction. Topics like health, politics, family drama, financial details, and personal vulnerabilities are automatically flagged.

2. **Auto-recall** checks the session context:
   - **DM / main session**: All memories are surfaced (no filtering)
   - **Group chat / channel**: Only non-sensitive memories are surfaced

3. **Manual storage** via `memory_store` accepts an optional `sensitive` parameter.

### What's considered sensitive?

| Sensitive (hidden in groups) | Not sensitive (shown everywhere) |
|---|---|
| Health/medical conditions | Technical discussions |
| Political views/affiliations | Project decisions |
| Family conflicts/drama | General preferences |
| Financial details (income, debt) | Work context |
| Personal fears/vulnerabilities | Schedules/deadlines |
| Legal issues | Fun facts |
| Addiction/substance use | Hobbies/interests |

### Migration

If you have existing memories without the `sensitive` field, run the migration:

```bash
# Default: set all existing memories to sensitive=false
openclaw wmem migrate-sensitive

# With keyword scan: attempt to detect and flag sensitive memories
openclaw wmem migrate-sensitive --scan
```

The `--scan` option uses keyword patterns to detect likely-sensitive content and flag it automatically. Review flagged memories afterward.

### Context detection

The plugin determines context from OpenClaw's session key:
- Session keys containing `:group:` or `:channel:` → group/shared context → filter sensitive
- All other keys (e.g., `main`, `discord:dm:...`) → direct/private context → no filtering

## CLI Commands

```bash
openclaw wmem stats                    # Show memory count
openclaw wmem search "query"           # Search memories
openclaw wmem search "query" --safe    # Search only non-sensitive memories
openclaw wmem store "text"             # Manually store a memory
openclaw wmem store "text" --sensitive # Store as sensitive
openclaw wmem forget <uuid>            # Delete a memory
openclaw wmem migrate-sensitive        # Backfill sensitive field
openclaw wmem migrate-sensitive --scan # Backfill with keyword detection
```

## Agent Tools

The plugin exposes these tools to the AI agent:

| Tool | Description |
|------|-------------|
| `memory_recall` | Semantic + hybrid search across all memories |
| `memory_store` | Save new information with category, importance, and sensitivity |
| `memory_forget` | Delete memories by ID or search query |
| `memory_stats` | Show database statistics |

### memory_store parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Information to remember |
| `importance` | number | 0.7 | Importance 0.0-1.0 |
| `category` | string | "other" | preference, fact, decision, entity, conversation, other |
| `sensitive` | boolean | false | Hide in group/shared contexts |

## Search Modes

- **hybrid** (default) - Best of both worlds. Uses Weaviate's hybrid search combining vector similarity with BM25 keyword matching. Great for recall.
- **vector** - Pure semantic search. Good when the query is conceptually related but uses different words.
- **keyword** - Falls back to BM25 text matching when needed.

## Architecture

```
Agent ──> memory_recall ──> WeaviateMemoryStore ──> Weaviate (Docker)
      ──> memory_store  ──>                     ──>
      
Lifecycle:
  before_agent_start ──> auto-recall (inject relevant memories, filter by context)
  agent_end          ──> LLM extraction (with sensitivity detection) ──> deduplicate ──> store

Context-aware filtering:
  DM/main session ──> all memories (including sensitive)
  Group/channel   ──> only non-sensitive memories
```

## Auto-Capture: LLM Extraction

Instead of brittle regex pattern matching, auto-capture sends each conversation turn through an LLM that intelligently decides what's worth remembering.

The extraction model:
- Understands context and intent ("I prefer tabs" vs "I'd prefer we move on")
- Condenses information into clean, self-contained memory statements
- Assigns accurate categories and importance scores
- **Detects sensitive content** and flags it accordingly
- Filters out noise, greetings, and transient chatter

### Extraction Providers

| Provider | `baseUrl` | `apiKey` | Cost |
|----------|-----------|----------|------|
| **Ollama** | `http://localhost:11434/v1` | not needed | Free |
| **LM Studio** | `http://localhost:1234/v1` | not needed | Free |
| **OpenAI** | (omit) | required | ~$0.001/turn |
| **Any OpenAI-compatible** | your endpoint | if needed | varies |

Configure via `extraction.baseUrl` and `extraction.model`. Any model that can follow structured JSON output instructions will work.

## Categories

Memories are categorized by the extraction LLM:
- `preference` - User likes, dislikes, wants
- `decision` - Agreed-upon choices
- `entity` - People, places, contacts
- `fact` - General knowledge/assertions
- `conversation` - Notable conversation context
- `other` - Everything else

## Schema

Each memory object in Weaviate has these properties:

| Property | Type | Description |
|----------|------|-------------|
| `text` | text | The memory content |
| `importance` | number | 0.0-1.0 importance score |
| `category` | text | Memory category |
| `source` | text | "manual", "auto-capture", or "agent" |
| `sensitive` | boolean | Whether the memory is sensitive |
| `sessionKey` | text | Session that created the memory |
| `createdAt` | int | Unix timestamp (ms) |
