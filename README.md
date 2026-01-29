# memory-weaviate

Weaviate-backed long-term vector memory plugin for Clawdbot.

## Features

- **Hybrid search** - combines vector similarity + keyword (BM25) for best recall
- **Auto-recall** - automatically injects relevant memories before each conversation
- **Auto-capture** - mines conversations for important info and stores it
- **Deduplication** - won't store near-identical memories
- **GDPR-friendly** - delete any memory by ID or search
- **Two embedding modes:**
  - `openai` - you provide embeddings via OpenAI API (recommended)
  - `weaviate` - use Weaviate's built-in text2vec-openai module

## Quick Start

### 1. Start Weaviate

```bash
cd extensions/memory-weaviate
docker compose up -d
```

### 2. Install dependencies

```bash
cd extensions/memory-weaviate
npm install
```

### 3. Configure Clawdbot

Add to your `clawdbot.json`:

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
          "autoCapture": true,
          "autoRecall": true
        }
      }
    }
  }
}
```

### 4. Restart Clawdbot

```bash
clawdbot gateway restart
```

## CLI Commands

```bash
clawdbot wmem stats          # Show memory count
clawdbot wmem search "query" # Search memories
clawdbot wmem store "text"   # Manually store a memory
clawdbot wmem forget <uuid>  # Delete a memory
```

## Agent Tools

The plugin exposes these tools to the AI agent:

| Tool | Description |
|------|-------------|
| `memory_recall` | Semantic + hybrid search across all memories |
| `memory_store` | Save new information with category and importance |
| `memory_forget` | Delete memories by ID or search query |
| `memory_stats` | Show database statistics |

## Search Modes

- **hybrid** (default) - Best of both worlds. Uses Weaviate's hybrid search combining vector similarity with BM25 keyword matching. Great for recall.
- **vector** - Pure semantic search. Good when the query is conceptually related but uses different words.
- **keyword** - Falls back to BM25 text matching when needed.

## Architecture

```
Agent ──> memory_recall ──> WeaviateMemoryStore ──> Weaviate (Docker)
      ──> memory_store  ──>                     ──>
      
Lifecycle:
  before_agent_start ──> auto-recall (inject context)
  agent_end          ──> auto-capture (mine & store)
```

## Categories

Memories are categorized automatically during auto-capture:
- `preference` - User likes, dislikes, wants
- `decision` - Agreed-upon choices
- `entity` - People, places, contacts
- `fact` - General knowledge/assertions
- `conversation` - Notable conversation context
- `other` - Everything else
