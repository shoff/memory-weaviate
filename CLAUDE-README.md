# Claude Code Integration Guide

This is the `memory-weaviate` plugin for OpenClaw — a Weaviate-backed long-term memory system with auto-recall and auto-capture.

## Quick Setup

### 1. Install Dependencies & Build

```bash
cd /path/to/memory-weaviate
npm install
npm run build
```

### 2. Ensure Weaviate is Running

The plugin needs a Weaviate instance. If not already running:

```bash
docker run -d \
  --name weaviate \
  -p 8081:8080 \
  -p 50052:50051 \
  cr.weaviate.io/semitechnologies/weaviate:1.28.4 \
  --host 0.0.0.0 \
  --port 8080 \
  --scheme http
```

### 3. Configure OpenClaw

Add to `~/.openclaw/openclaw.json` under the `plugins` section:

```json
{
  "plugins": {
    "memory-weaviate": {
      "path": "/absolute/path/to/memory-weaviate",
      "weaviate": {
        "url": "http://localhost:8081"
      },
      "collectionName": "ClawdbotMemory",
      "autoCapture": true,
      "autoRecall": true,
      "embedding": {
        "provider": "openai",
        "apiKey": "sk-your-openai-key",
        "model": "text-embedding-3-small"
      },
      "extraction": {
        "apiKey": "sk-your-openai-key",
        "model": "gpt-5-nano"
      }
    }
  }
}
```

### 4. Restart Gateway

```bash
openclaw gateway restart
```

---

## Configuration Reference

### Required

| Key | Description |
|-----|-------------|
| `weaviate.url` | Weaviate instance URL (e.g., `http://localhost:8081`) |

### Optional

| Key | Default | Description |
|-----|---------|-------------|
| `collectionName` | `ClawdbotMemory` | Weaviate collection name |
| `autoCapture` | `true` | Auto-extract memories from conversations |
| `autoRecall` | `true` | Auto-inject relevant memories into context |
| `weaviate.grpcPort` | auto | gRPC port (usually 50051 or URL port + 1) |
| `weaviate.apiKey` | - | API key for Weaviate Cloud instances |

### Embedding Config

```json
"embedding": {
  "provider": "openai",        // "openai" or "weaviate"
  "apiKey": "sk-...",          // OpenAI API key (if provider is openai)
  "model": "text-embedding-3-small"  // or "text-embedding-3-large"
}
```

For Weaviate's built-in vectorizer (no OpenAI needed):
```json
"embedding": {
  "provider": "weaviate"
}
```

### Extraction Config (for auto-capture)

```json
"extraction": {
  "apiKey": "sk-...",           // OpenAI API key
  "model": "gpt-5-nano",        // or gpt-4o-mini, gpt-4o, etc.
  "baseUrl": "...",             // Optional: for Azure, Ollama, LM Studio
  "maxTokens": 1024             // Optional: max response tokens
}
```

**For Azure OpenAI:**
```json
"extraction": {
  "baseUrl": "https://your-resource.openai.azure.com/openai/deployments/your-deployment",
  "apiKey": "your-azure-key",
  "model": "gpt-4o-mini"
}
```

**For Local LLMs (Ollama/LM Studio):**
```json
"extraction": {
  "baseUrl": "http://localhost:11434/v1",
  "model": "llama3.2"
}
```

---

## Troubleshooting

### "Memory unavailable" in status

1. Check Weaviate is running: `curl http://localhost:8081/v1/.well-known/ready`
2. Verify the URL in config matches your Weaviate port
3. Check logs: `openclaw gateway logs`

### Auto-capture not working

Ensure both are set:
- `autoCapture: true`
- `extraction.apiKey` or `extraction.baseUrl` configured

Check logs for `memory-weaviate: auto-captured X memories` messages.

### gRPC connection errors

If Weaviate runs on non-standard ports, explicitly set:
```json
"weaviate": {
  "url": "http://localhost:8081",
  "grpcPort": 50052
}
```

---

## Memory Tools Available

Once configured, these tools are available to the agent:

| Tool | Description |
|------|-------------|
| `memory_store` | Save information to long-term memory (supports `sensitive` flag) |
| `memory_recall` | Search memories (hybrid: vector + keyword) |
| `memory_forget` | Delete memories by ID or search |
| `memory_stats` | Show memory database statistics |

## Sensitivity Filtering

The plugin supports **sensitivity-aware recall**:

- **`sensitive: true`** memories are hidden in group/shared contexts (session keys containing `:group:` or `:channel:`)
- **`sensitive: false`** memories are shown everywhere
- Auto-capture uses LLM to detect sensitive content (health, politics, family drama, finances, vulnerabilities)
- Use `memory_store` with `sensitive: true` to manually flag sensitive memories
- Migration: run `openclaw wmem migrate-sensitive --scan` to backfill existing memories

---

## Updating the Plugin

```bash
cd /path/to/memory-weaviate
git pull origin main
npm install
npm run build
openclaw gateway restart
```

---

## Support

- **Repo:** https://github.com/shoff/memory-weaviate
- **Issues:** Open a GitHub issue or ask Mei 😉
