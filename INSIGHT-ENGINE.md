# Memory Insight Engine - Design Doc

## Concept
Async background process that mimics human memory consolidation.
Runs periodically (every 4-6 hours) via cron job, using GPT-5 Nano.

## What It Does

### 1. Pattern Recognition
- Sample 10-15 random memories from Weaviate
- Ask LLM: "What patterns, connections, or insights emerge from these memories?"
- Store genuine insights as new memories (category: "other", source: "insight-engine")

### 2. Memory Consolidation
- Find clusters of related memories (semantic similarity > 0.85)
- Merge fragmented memories into stronger, distilled versions
- Example: 3 memories about Steve's gym → 1 comprehensive memory

### 3. Staleness Detection
- Flag memories that contradict newer ones
- Identify time-sensitive facts that may be outdated
- Mark for review or auto-archive

### 4. Emotional Context Mapping
- Track emotional threads across conversations
- "Steve was frustrated about X on date Y, resolved by Z"
- Helps with empathy and context in future conversations

### 5. Knowledge Graph Edges
- Identify relationships between entities
- "Travis → kendo → BudoHub → Steve's project"
- "Riza → BPD → communication challenges → loneliness"

## Implementation Options

### Option A: Cron Job (Simplest)
- Isolated session via cron, runs every 4h
- Uses memory_recall to sample, GPT-5 Nano to analyze, memory_store to save insights
- Pro: No plugin changes needed. Con: Limited to tool interface.

### Option B: Plugin Hook (Most Integrated)  
- Add `insightEngine` section to plugin config
- Runs on a timer inside the plugin process
- Direct Weaviate access for sampling and batch operations
- Pro: Full control, efficient. Con: Requires plugin code changes.

### Option C: Hybrid
- Cron triggers a lightweight analysis
- Plugin provides a `memory_consolidate` tool the cron agent can call
- Best of both worlds

## Recommended: Option A first, evolve to B
Start with a cron job. It's zero code changes, immediately testable.
Once we validate the insight patterns, bake it into the plugin.

## Insight Memory Format
```json
{
  "text": "[INSIGHT] Steve's discipline across gym, kendo, and software architecture all stem from the same drive: control through structure and earned mastery.",
  "category": "other",
  "importance": 0.8,
  "source": "insight-engine"
}
```

## Staleness Flag Format
```json
{
  "text": "[STALE] Memory 'Discord integration is broken' contradicted by newer memory 'Discord integration fix applied'. Original should be deleted.",
  "category": "other", 
  "importance": 0.6,
  "source": "insight-engine"
}
```
