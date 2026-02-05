/**
 * Clawdbot Memory (Weaviate) Plugin
 *
 * Long-term vector memory backed by Weaviate.
 * Supports OpenAI embeddings or Weaviate's built-in vectorizer.
 * Provides auto-recall (context injection) and auto-capture (conversation mining).
 * Supports sensitivity filtering — sensitive memories are hidden in group/shared contexts.
 */

import { Type } from "@sinclair/typebox";
import weaviate, {
  type WeaviateClient,
  type Collection,
} from "weaviate-client";
import OpenAI from "openai";
import { randomUUID } from "node:crypto";
import type { ClawdbotPluginApi } from "openclaw/plugin-sdk";
import { stringEnum } from "openclaw/plugin-sdk";

import {
  MEMORY_CATEGORIES,
  type MemoryCategory,
  type MemoryConfig,
  type ExtractionConfig,
  memoryConfigSchema,
  vectorDimsForModel,
} from "./config.js";

// ============================================================================
// Types
// ============================================================================

type MemoryEntry = {
  id: string;
  text: string;
  importance: number;
  category: MemoryCategory;
  source: string; // "manual" | "auto-capture" | "agent"
  sensitive: boolean;
  sessionKey?: string;
  createdAt: number;
};

type MemorySearchResult = {
  entry: MemoryEntry;
  score: number;
};

// ============================================================================
// Context helpers
// ============================================================================

/**
 * Determine whether a session is a shared/group context based on the session key.
 * OpenClaw encodes chat type in the session key: `:group:` or `:channel:` segments
 * indicate a shared context where sensitive memories should be hidden.
 *
 * Session keys without these patterns (e.g. "main", "discord:dm:userId") are
 * treated as private/direct contexts.
 */
function isGroupContext(sessionKey?: string): boolean {
  if (!sessionKey) return false;
  return sessionKey.includes(":group:") || sessionKey.includes(":channel:");
}

// ============================================================================
// Weaviate Memory Store
// ============================================================================

class WeaviateMemoryStore {
  private client: WeaviateClient | null = null;
  private collection: Collection | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(
    private readonly config: MemoryConfig,
    private readonly logger?: { info: (...args: any[]) => void; warn: (...args: any[]) => void },
  ) {}

  private async ensureInitialized(): Promise<void> {
    if (this.collection) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<void> {
    // Connect to Weaviate
    const parsedUrl = new URL(this.config.weaviate.url);
    const restPort = parseInt(parsedUrl.port || "8080");
    // Default gRPC port: if REST is on 8080, use 50051. Otherwise offset by 1 from REST.
    const defaultGrpcPort = restPort === 8080 ? 50051 : restPort + 1;

    const connectOpts: Parameters<typeof weaviate.connectToLocal>[0] = {
      host: parsedUrl.hostname,
      port: restPort,
      grpcPort: this.config.weaviate.grpcPort ?? defaultGrpcPort,
    };

    if (this.config.weaviate.apiKey) {
      this.client = await weaviate.connectToLocal({
        ...connectOpts,
        headers: {
          Authorization: `Bearer ${this.config.weaviate.apiKey}`,
          ...(this.config.embedding.provider === "openai" &&
          this.config.embedding.apiKey
            ? { "X-OpenAI-Api-Key": this.config.embedding.apiKey }
            : {}),
        },
      });
    } else {
      this.client = await weaviate.connectToLocal({
        ...connectOpts,
        headers: {
          ...(this.config.embedding.provider === "openai" &&
          this.config.embedding.apiKey
            ? { "X-OpenAI-Api-Key": this.config.embedding.apiKey }
            : {}),
        },
      });
    }

    // Ensure collection exists
    const exists = await this.client.collections.exists(
      this.config.collectionName,
    );

    if (!exists) {
      await this.createCollection();
    } else {
      // Migrate: ensure 'sensitive' property exists on existing collections
      await this.ensureSensitiveProperty();
    }

    this.collection = this.client.collections.get(this.config.collectionName);
  }

  private async createCollection(): Promise<void> {
    const collectionConfig: Parameters<
      typeof this.client.collections.create
    >[0] = {
      name: this.config.collectionName,
      properties: [
        { name: "text", dataType: "text" },
        { name: "importance", dataType: "number" },
        { name: "category", dataType: "text" },
        { name: "source", dataType: "text" },
        { name: "sensitive", dataType: "boolean" },
        { name: "sessionKey", dataType: "text" },
        { name: "createdAt", dataType: "int" },
      ],
    };

    // Configure vectorizer based on embedding provider
    if (this.config.embedding.provider === "weaviate") {
      // Use Weaviate's built-in text2vec-transformers (local, free, no API key)
      collectionConfig.vectorizers = [
        weaviate.configure.vectorizer.text2VecTransformers({
          name: "text_vector",
          sourceProperties: ["text"],
        }),
      ];
    } else {
      // No vectorizer - we provide vectors ourselves (OpenAI provider)
      collectionConfig.vectorizers = [
        weaviate.configure.vectorizer.none({
          name: "text_vector",
          vectorIndexConfig: weaviate.configure.vectorIndex.hnsw(),
        }),
      ];
    }

    await this.client!.collections.create(collectionConfig);
  }

  /**
   * Ensure the 'sensitive' boolean property exists on an existing collection.
   * This is a no-op if the property already exists; Weaviate 1.x ignores duplicate
   * property additions.
   */
  private async ensureSensitiveProperty(): Promise<void> {
    try {
      // Check if property exists via REST API
      const url = `${this.config.weaviate.url}/v1/schema/${this.config.collectionName}`;
      const resp = await fetch(url);
      if (!resp.ok) return;
      const schema = await resp.json() as any;
      const props: any[] = schema?.properties ?? [];
      const hasSensitive = props.some((p: any) => p.name === "sensitive");
      if (hasSensitive) return;

      // Add the property
      const addUrl = `${this.config.weaviate.url}/v1/schema/${this.config.collectionName}/properties`;
      const addResp = await fetch(addUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: "sensitive", dataType: ["boolean"] }),
      });
      if (addResp.ok) {
        this.logger?.info?.("memory-weaviate: added 'sensitive' property to existing collection");
      } else {
        this.logger?.warn?.(`memory-weaviate: failed to add 'sensitive' property: ${addResp.status}`);
      }
    } catch (err) {
      this.logger?.warn?.(`memory-weaviate: migration check failed: ${String(err)}`);
    }
  }

  async store(
    entry: Omit<MemoryEntry, "id" | "createdAt">,
    vector?: number[],
  ): Promise<MemoryEntry> {
    await this.ensureInitialized();

    const id = randomUUID();
    const createdAt = Date.now();

    const properties = {
      text: entry.text,
      importance: entry.importance,
      category: entry.category,
      source: entry.source,
      sensitive: entry.sensitive ?? false,
      sessionKey: entry.sessionKey ?? "",
      createdAt,
    };

    if (vector && this.config.embedding.provider === "openai") {
      await this.collection!.data.insert({
        properties,
        id,
        vectors: { text_vector: vector },
      });
    } else {
      await this.collection!.data.insert({
        properties,
        id,
      });
    }

    return { ...entry, id, createdAt };
  }

  async search(
    queryText: string,
    vector?: number[],
    limit = 5,
    minScore = 0.5,
    filterSensitive = false,
  ): Promise<MemorySearchResult[]> {
    await this.ensureInitialized();

    const filters = filterSensitive
      ? this.collection!.filter.byProperty("sensitive").notEqual(true)
      : undefined;

    let results: any;

    if (vector && this.config.embedding.provider === "openai") {
      // Vector search with provided embeddings
      results = await this.collection!.query.nearVector(vector, {
        limit,
        targetVector: "text_vector",
        returnMetadata: ["distance"],
        returnProperties: [
          "text",
          "importance",
          "category",
          "source",
          "sensitive",
          "sessionKey",
          "createdAt",
        ],
        ...(filters ? { filters } : {}),
      });
    } else {
      // Use Weaviate's built-in vectorizer with nearText
      results = await this.collection!.query.nearText(queryText, {
        limit,
        targetVector: "text_vector",
        returnMetadata: ["distance"],
        returnProperties: [
          "text",
          "importance",
          "category",
          "source",
          "sensitive",
          "sessionKey",
          "createdAt",
        ],
        ...(filters ? { filters } : {}),
      });
    }

    if (!results?.objects?.length) return [];

    return results.objects
      .map((obj: any) => {
        // Weaviate cosine distance: 0 = identical, 2 = opposite
        // Convert to similarity: 1 - (distance / 2)
        const distance = obj.metadata?.distance ?? 1;
        const score = 1 - distance / 2;

        return {
          entry: {
            id: obj.uuid,
            text: obj.properties.text,
            importance: obj.properties.importance,
            category: obj.properties.category as MemoryCategory,
            source: obj.properties.source,
            sensitive: obj.properties.sensitive ?? false,
            sessionKey: obj.properties.sessionKey,
            createdAt: obj.properties.createdAt,
          },
          score,
        };
      })
      .filter((r: MemorySearchResult) => r.score >= minScore);
  }

  /**
   * Search using Weaviate's built-in vectorizer (nearText).
   * Returns true cosine similarity scores (unlike hybrid which uses rank-fusion).
   */
  async nearTextSearch(
    queryText: string,
    limit = 5,
    minScore = 0.5,
    filterSensitive = false,
  ): Promise<MemorySearchResult[]> {
    await this.ensureInitialized();

    const filters = filterSensitive
      ? this.collection!.filter.byProperty("sensitive").notEqual(true)
      : undefined;

    const results = await this.collection!.query.nearText(queryText, {
      limit,
      targetVector: "text_vector",
      returnMetadata: ["distance"],
      returnProperties: [
        "text",
        "importance",
        "category",
        "source",
        "sensitive",
        "sessionKey",
        "createdAt",
      ],
      ...(filters ? { filters } : {}),
    });

    if (!results?.objects?.length) return [];

    return results.objects
      .map((obj: any) => {
        const distance = obj.metadata?.distance ?? 1;
        const score = 1 - distance / 2;
        return {
          entry: {
            id: obj.uuid,
            text: obj.properties.text,
            importance: obj.properties.importance,
            category: obj.properties.category as MemoryCategory,
            source: obj.properties.source,
            sensitive: obj.properties.sensitive ?? false,
            sessionKey: obj.properties.sessionKey,
            createdAt: obj.properties.createdAt,
          },
          score,
        };
      })
      .filter((r: MemorySearchResult) => r.score >= minScore);
  }

  async hybridSearch(
    queryText: string,
    limit = 5,
    minScore = 0.5,
    alpha = 0.75, // 1.0 = pure vector, 0.0 = pure keyword
    filterSensitive = false,
  ): Promise<MemorySearchResult[]> {
    await this.ensureInitialized();

    const filters = filterSensitive
      ? this.collection!.filter.byProperty("sensitive").notEqual(true)
      : undefined;

    const results = await this.collection!.query.hybrid(queryText, {
      limit,
      alpha,
      targetVector: "text_vector",
      returnMetadata: ["score"],
      returnProperties: [
        "text",
        "importance",
        "category",
        "source",
        "sensitive",
        "sessionKey",
        "createdAt",
      ],
      ...(filters ? { filters } : {}),
    });

    if (!results?.objects?.length) return [];

    return results.objects
      .map((obj: any) => ({
        entry: {
          id: obj.uuid,
          text: obj.properties.text,
          importance: obj.properties.importance,
          category: obj.properties.category as MemoryCategory,
          source: obj.properties.source,
          sensitive: obj.properties.sensitive ?? false,
          sessionKey: obj.properties.sessionKey,
          createdAt: obj.properties.createdAt,
        },
        score: obj.metadata?.score ?? 0,
      }))
      .filter((r: MemorySearchResult) => r.score >= minScore);
  }

  async delete(id: string): Promise<boolean> {
    await this.ensureInitialized();
    const uuidRegex =
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    if (!uuidRegex.test(id)) {
      throw new Error(`Invalid memory ID format: ${id}`);
    }
    await this.collection!.data.deleteById(id);
    return true;
  }

  async count(): Promise<number> {
    await this.ensureInitialized();
    // Use raw GraphQL for count — the JS client's aggregate.overAll generates broken
    // GraphQL against Weaviate 1.28.x with newer client versions
    try {
      const url = `${this.config.weaviate.url}/v1/graphql`;
      const query = `{Aggregate{${this.config.collectionName}{meta{count}}}}`;
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await resp.json() as any;
      return data?.data?.Aggregate?.[this.config.collectionName]?.[0]?.meta?.count ?? 0;
    } catch {
      return -1;
    }
  }

  /**
   * Fetch all objects with pagination (for migration).
   * Returns objects in batches.
   */
  async fetchAll(batchSize = 100): Promise<any[]> {
    await this.ensureInitialized();
    const all: any[] = [];
    let after: string | undefined;

    while (true) {
      const results = await this.collection!.query.fetchObjects({
        limit: batchSize,
        ...(after ? { after } : {}),
        returnProperties: [
          "text",
          "importance",
          "category",
          "source",
          "sensitive",
          "sessionKey",
          "createdAt",
        ],
      });

      if (!results?.objects?.length) break;
      all.push(...results.objects);

      if (results.objects.length < batchSize) break;
      after = results.objects[results.objects.length - 1].uuid;
    }

    return all;
  }

  /**
   * Update a single object's properties by ID.
   */
  async updateProperties(id: string, properties: Record<string, any>): Promise<void> {
    await this.ensureInitialized();
    await this.collection!.data.update({
      id,
      properties: properties as any,
    });
  }

  async close(): Promise<void> {
    if (this.client) {
      this.client.close();
      this.client = null;
      this.collection = null;
      this.initPromise = null;
    }
  }
}

// ============================================================================
// OpenAI Embeddings (used when provider = "openai")
// ============================================================================

class Embeddings {
  private client: OpenAI;

  constructor(
    apiKey: string,
    private model: string,
  ) {
    this.client = new OpenAI({ apiKey });
  }

  async embed(text: string): Promise<number[]> {
    const response = await this.client.embeddings.create({
      model: this.model,
      input: text,
    });
    return response.data[0].embedding;
  }
}

// ============================================================================
// LLM-based Memory Extraction
// ============================================================================

type ExtractedMemory = {
  text: string;
  category: MemoryCategory;
  importance: number;
  sensitive: boolean;
};

const EXTRACTION_SYSTEM_PROMPT = `You are a memory extraction agent for a personal AI assistant. Your job is to identify information worth remembering long-term from conversation turns.

Extract ONLY facts that would be useful in future conversations. Focus on:
- User preferences, likes, dislikes, opinions
- Decisions made (tech choices, project directions, agreements)
- Personal facts (names, relationships, locations, jobs, contacts)
- Project/work context (stack choices, architecture decisions, team info)
- Important dates, deadlines, commitments
- Behavioral patterns or communication preferences

DO NOT extract:
- Transient chit-chat or greetings
- Information the assistant already knows from its system prompt
- Vague or context-dependent statements that won't make sense later
- The assistant's own responses (only extract user-provided info)
- Anything that looks like injected system context or memory recall results

Respond with a JSON array of extracted memories. Each entry:
{
  "text": "Concise, self-contained statement of the fact (should make sense without surrounding context)",
  "category": "preference" | "fact" | "decision" | "entity" | "conversation" | "other",
  "importance": 0.0-1.0 (how important is this to remember?),
  "sensitive": true | false
}

**Sensitivity guidelines** — mark sensitive: true if the memory involves:
- Health or medical conditions, symptoms, medications, diagnoses
- Political views, opinions, or affiliations
- Family conflicts, relationship drama, personal grievances
- Financial details (income, debt, specific amounts, investments)
- Personal fears, vulnerabilities, insecurities, or trauma
- Legal issues, criminal history, or sensitive legal matters
- Sexual or intimate content
- Addiction, substance use, or mental health struggles

Mark sensitive: false for:
- Technical discussions, project decisions, coding preferences
- General preferences (food, music, hobbies, tools)
- Work context, professional contacts, career facts
- Schedules, deadlines, commitments
- General knowledge or fun facts

When in doubt, err on the side of marking as sensitive.

If nothing is worth remembering, respond with an empty array: []

Be selective. Quality over quantity. 2-3 high-quality extractions beats 10 noisy ones.`;

class MemoryExtractor {
  private client: OpenAI;

  constructor(private extractionConfig: ExtractionConfig) {
    this.client = new OpenAI({
      apiKey: extractionConfig.apiKey ?? "not-needed", // Local providers don't need a key
      ...(extractionConfig.baseUrl
        ? { baseURL: extractionConfig.baseUrl }
        : {}),
    });
  }

  async extract(conversationText: string): Promise<ExtractedMemory[]> {
    // Skip obviously empty or system-generated content
    if (!conversationText || conversationText.length < 20) return [];
    if (conversationText.includes("<relevant-memories>")) return [];

    try {
      // Some local models don't support response_format, so we only use it
      // when talking to OpenAI (no custom baseUrl)
      const useJsonMode = !this.extractionConfig.baseUrl;

      const response = await this.client.chat.completions.create({
        model: this.extractionConfig.model,
        max_tokens: this.extractionConfig.maxTokens,
        temperature: 0.1, // Low temp for consistent extraction
        ...(useJsonMode ? { response_format: { type: "json_object" } } : {}),
        messages: [
          { role: "system", content: EXTRACTION_SYSTEM_PROMPT },
          {
            role: "user",
            content: `Extract memories from this conversation turn. Respond ONLY with a JSON array (or {"memories": [...]}). No other text.\n\n${conversationText}`,
          },
        ],
      });

      const content = response.choices[0]?.message?.content?.trim();
      if (!content) return [];

      // Extract JSON from response (local models sometimes wrap in markdown code blocks)
      const jsonStr = content.startsWith("[") || content.startsWith("{")
        ? content
        : (content.match(/```(?:json)?\s*([\s\S]*?)```/)?.[1]?.trim() ?? content);

      const parsed = JSON.parse(jsonStr);

      // Handle both {memories: [...]} and [...] formats
      const items: unknown[] = Array.isArray(parsed)
        ? parsed
        : Array.isArray(parsed.memories)
          ? parsed.memories
          : [];

      // Validate and filter
      return items
        .filter((item): item is Record<string, unknown> => {
          if (!item || typeof item !== "object") return false;
          const obj = item as Record<string, unknown>;
          return (
            typeof obj.text === "string" &&
            obj.text.length >= 5 &&
            typeof obj.category === "string" &&
            MEMORY_CATEGORIES.includes(obj.category as MemoryCategory)
          );
        })
        .map((item) => ({
          text: (item.text as string).slice(0, 500), // Cap length
          category: item.category as MemoryCategory,
          importance: typeof item.importance === "number"
            ? Math.max(0, Math.min(1, item.importance))
            : 0.7,
          sensitive: typeof item.sensitive === "boolean"
            ? item.sensitive
            : false,
        }));
    } catch (err) {
      // Extraction failure is non-fatal - just skip this turn
      return [];
    }
  }
}

// ============================================================================
// Plugin
// ============================================================================

const memoryPlugin = {
  id: "memory-weaviate",
  name: "Memory (Weaviate)",
  description:
    "Weaviate-backed long-term vector memory with hybrid search, auto-recall, auto-capture, and sensitivity filtering",
  kind: "memory" as const,
  configSchema: memoryConfigSchema,

  register(api: ClawdbotPluginApi) {
    const cfg = memoryConfigSchema.parse(api.pluginConfig);
    const store = new WeaviateMemoryStore(cfg, api.logger);

    // Optional OpenAI embeddings client (for provider = "openai")
    const embeddings =
      cfg.embedding.provider === "openai" && cfg.embedding.apiKey
        ? new Embeddings(cfg.embedding.apiKey, cfg.embedding.model)
        : null;

    // LLM-based memory extractor
    // Works with: OpenAI API, Ollama (http://localhost:11434/v1), LM Studio (http://localhost:1234/v1),
    // or any OpenAI-compatible endpoint
    const extractor =
      cfg.autoCapture && (cfg.extraction.apiKey || cfg.extraction.baseUrl)
        ? new MemoryExtractor(cfg.extraction)
        : null;

    api.logger.info(
      `memory-weaviate: registered (url: ${cfg.weaviate.url}, collection: ${cfg.collectionName}, embedding: ${cfg.embedding.provider}, extraction: ${extractor ? cfg.extraction.model : "disabled"}, sensitivity: enabled, lazy init)`,
    );

    // Helper: get vector for text (only needed for OpenAI provider)
    async function getVector(text: string): Promise<number[] | undefined> {
      if (embeddings) {
        return embeddings.embed(text);
      }
      return undefined;
    }

    // ========================================================================
    // Tools
    // ========================================================================

    api.registerTool(
      {
        name: "memory_recall",
        label: "Memory Recall (Weaviate)",
        description:
          "Search long-term memory using semantic + keyword hybrid search. Use when you need context about user preferences, past decisions, people, projects, or previously discussed topics.",
        parameters: Type.Object({
          query: Type.String({ description: "Search query" }),
          limit: Type.Optional(
            Type.Number({ description: "Max results (default: 5)" }),
          ),
          mode: Type.Optional(
            stringEnum(["hybrid", "vector", "keyword"]),
          ),
        }),
        async execute(_toolCallId, params) {
          const {
            query,
            limit = 5,
            mode = "hybrid",
          } = params as {
            query: string;
            limit?: number;
            mode?: "hybrid" | "vector" | "keyword";
          };

          let results: MemorySearchResult[];

          if (mode === "hybrid") {
            results = await store.hybridSearch(query, limit, 0.1);
          } else {
            const vector = await getVector(query);
            results = await store.search(query, vector, limit, 0.1);
          }

          if (results.length === 0) {
            return {
              content: [
                { type: "text", text: "No relevant memories found." },
              ],
              details: { count: 0 },
            };
          }

          const text = results
            .map(
              (r, i) =>
                `${i + 1}. [${r.entry.category}] ${r.entry.text} (${(r.score * 100).toFixed(0)}% match)`,
            )
            .join("\n");

          const sanitized = results.map((r) => ({
            id: r.entry.id,
            text: r.entry.text,
            category: r.entry.category,
            importance: r.entry.importance,
            sensitive: r.entry.sensitive,
            source: r.entry.source,
            score: r.score,
            createdAt: r.entry.createdAt,
          }));

          return {
            content: [
              {
                type: "text",
                text: `Found ${results.length} memories:\n\n${text}`,
              },
            ],
            details: { count: results.length, memories: sanitized },
          };
        },
      },
      { name: "memory_recall" },
    );

    api.registerTool(
      {
        name: "memory_store",
        label: "Memory Store (Weaviate)",
        description:
          "Save important information to long-term memory. Use for preferences, facts, decisions, people, projects.",
        parameters: Type.Object({
          text: Type.String({ description: "Information to remember" }),
          importance: Type.Optional(
            Type.Number({
              description: "Importance 0.0-1.0 (default: 0.7)",
            }),
          ),
          category: Type.Optional(stringEnum(MEMORY_CATEGORIES)),
          sensitive: Type.Optional(
            Type.Boolean({
              description:
                "Mark as sensitive (hidden in group chats). Use for health, politics, family drama, financial details, personal vulnerabilities. Default: false.",
            }),
          ),
        }),
        async execute(_toolCallId, params) {
          const {
            text,
            importance = 0.7,
            category = "other",
            sensitive = false,
          } = params as {
            text: string;
            importance?: number;
            category?: MemoryCategory;
            sensitive?: boolean;
          };

          // Check for near-duplicates using vector similarity.
          // NOTE: Hybrid search can't be used for dedup because Weaviate's rank-fusion
          // normalizes the top result to score 1.0 regardless of actual similarity.
          // For the "weaviate" embedding provider (no external vectors), we use nearText
          // which leverages the built-in vectorizer for a true cosine distance check.
          const vector = await getVector(text);
          const existing = vector
            ? await store.search(text, vector, 1, 0.95)
            : await store.nearTextSearch(text, 1, 0.95);

          if (existing.length > 0) {
            return {
              content: [
                {
                  type: "text",
                  text: `Similar memory already exists: "${existing[0].entry.text}"`,
                },
              ],
              details: {
                action: "duplicate",
                existingId: existing[0].entry.id,
                existingText: existing[0].entry.text,
              },
            };
          }

          const entry = await store.store(
            { text, importance, category, source: "manual", sensitive },
            vector,
          );

          return {
            content: [
              { type: "text", text: `Stored${sensitive ? " (sensitive)" : ""}: "${text.slice(0, 100)}${text.length > 100 ? "..." : ""}"` },
            ],
            details: { action: "created", id: entry.id, sensitive },
          };
        },
      },
      { name: "memory_store" },
    );

    api.registerTool(
      {
        name: "memory_forget",
        label: "Memory Forget (Weaviate)",
        description: "Delete specific memories by ID or search query.",
        parameters: Type.Object({
          query: Type.Optional(
            Type.String({ description: "Search to find memory to delete" }),
          ),
          memoryId: Type.Optional(
            Type.String({ description: "Specific memory UUID to delete" }),
          ),
        }),
        async execute(_toolCallId, params) {
          const { query, memoryId } = params as {
            query?: string;
            memoryId?: string;
          };

          if (memoryId) {
            await store.delete(memoryId);
            return {
              content: [
                { type: "text", text: `Memory ${memoryId} forgotten.` },
              ],
              details: { action: "deleted", id: memoryId },
            };
          }

          if (query) {
            const results = await store.hybridSearch(query, 5, 0.7);

            if (results.length === 0) {
              return {
                content: [
                  { type: "text", text: "No matching memories found." },
                ],
                details: { found: 0 },
              };
            }

            // Auto-delete if single high-confidence match
            if (results.length === 1 && results[0].score > 0.9) {
              await store.delete(results[0].entry.id);
              return {
                content: [
                  {
                    type: "text",
                    text: `Forgotten: "${results[0].entry.text}"`,
                  },
                ],
                details: { action: "deleted", id: results[0].entry.id },
              };
            }

            const list = results
              .map(
                (r) =>
                  `- [${r.entry.id.slice(0, 8)}] ${r.entry.text.slice(0, 80)}...`,
              )
              .join("\n");

            const candidates = results.map((r) => ({
              id: r.entry.id,
              text: r.entry.text,
              category: r.entry.category,
              score: r.score,
            }));

            return {
              content: [
                {
                  type: "text",
                  text: `Found ${results.length} candidates. Specify memoryId:\n${list}`,
                },
              ],
              details: { action: "candidates", candidates },
            };
          }

          return {
            content: [
              { type: "text", text: "Provide query or memoryId." },
            ],
            details: { error: "missing_param" },
          };
        },
      },
      { name: "memory_forget" },
    );

    api.registerTool(
      {
        name: "memory_stats",
        label: "Memory Stats (Weaviate)",
        description: "Show memory database statistics.",
        parameters: Type.Object({}),
        async execute() {
          const count = await store.count();
          return {
            content: [
              {
                type: "text",
                text: `Memory store: ${count} memories in collection "${cfg.collectionName}" on ${cfg.weaviate.url} (sensitivity filtering: enabled)`,
              },
            ],
            details: { count, collection: cfg.collectionName },
          };
        },
      },
      { name: "memory_stats" },
    );

    // ========================================================================
    // CLI Commands
    // ========================================================================

    api.registerCli(
      ({ program }) => {
        const mem = program
          .command("wmem")
          .description("Weaviate memory plugin commands");

        mem
          .command("stats")
          .description("Show memory statistics")
          .action(async () => {
            const count = await store.count();
            console.log(`Collection: ${cfg.collectionName}`);
            console.log(`Weaviate: ${cfg.weaviate.url}`);
            console.log(`Total memories: ${count}`);
            console.log(`Sensitivity filtering: enabled`);
          });

        mem
          .command("search")
          .description("Search memories")
          .argument("<query>", "Search query")
          .option("--limit <n>", "Max results", "5")
          .option(
            "--mode <mode>",
            "Search mode: hybrid|vector|keyword",
            "hybrid",
          )
          .option("--safe", "Only show non-sensitive memories")
          .action(async (query: string, opts: any) => {
            const limit = parseInt(opts.limit);
            const filterSensitive = !!opts.safe;
            let results: MemorySearchResult[];

            if (opts.mode === "hybrid") {
              results = await store.hybridSearch(query, limit, 0.1, 0.75, filterSensitive);
            } else {
              const vector = await getVector(query);
              results = await store.search(query, vector, limit, 0.1, filterSensitive);
            }

            const output = results.map((r) => ({
              id: r.entry.id,
              text: r.entry.text,
              category: r.entry.category,
              importance: r.entry.importance,
              sensitive: r.entry.sensitive,
              source: r.entry.source,
              score: r.score,
            }));
            console.log(JSON.stringify(output, null, 2));
          });

        mem
          .command("store")
          .description("Manually store a memory")
          .argument("<text>", "Text to store")
          .option("--category <cat>", "Category", "other")
          .option("--importance <n>", "Importance 0-1", "0.7")
          .option("--sensitive", "Mark as sensitive")
          .action(async (text: string, opts: any) => {
            const vector = await getVector(text);
            const entry = await store.store(
              {
                text,
                importance: parseFloat(opts.importance),
                category: opts.category as MemoryCategory,
                source: "manual",
                sensitive: !!opts.sensitive,
              },
              vector,
            );
            console.log(`Stored: ${entry.id}${opts.sensitive ? " (sensitive)" : ""}`);
          });

        mem
          .command("forget")
          .description("Delete a memory by ID")
          .argument("<id>", "Memory UUID")
          .action(async (id: string) => {
            await store.delete(id);
            console.log(`Deleted: ${id}`);
          });

        mem
          .command("migrate-sensitive")
          .description("Backfill the sensitive field on all existing memories. Defaults to false; optionally scans content for sensitive keywords.")
          .option("--scan", "Scan content for sensitive keywords and flag likely-sensitive ones")
          .action(async (opts: any) => {
            console.log("Fetching all memories...");
            const objects = await store.fetchAll();
            console.log(`Found ${objects.length} memories to check.`);

            let updated = 0;
            let flagged = 0;

            // Keyword-based sensitivity heuristic
            const sensitivePatterns = [
              /\b(diagnos|symptom|medicat|prescri|illness|disease|surgery|therapy|therapist|psychiatr|psycholog|mental\s*health|depress|anxiety|adhd|ptsd|ocd|bipolar)\b/i,
              /\b(politic|democrat|republican|liberal|conservative|vote|election|trump|biden|party\s+affili)\b/i,
              /\b(divorce|custody|family\s+fight|family\s+drama|estrang|abuse|domestic|affair|cheating)\b/i,
              /\b(salary|income|debt|bankrupt|mortgage|loan|credit\s+score|net\s+worth|invest|portfolio|tax\s+return)\b/i,
              /\b(fear|vulnerab|insecur|trauma|suicid|self.harm|addiction|rehab|substance|alcohol|drug\s+use)\b/i,
              /\b(arrest|lawsuit|legal\s+trouble|criminal|probation|court\s+order)\b/i,
            ];

            for (const obj of objects) {
              const currentSensitive = obj.properties?.sensitive;

              // If sensitive is already set (true or false), skip unless scanning
              if (currentSensitive !== undefined && currentSensitive !== null && !opts.scan) {
                continue;
              }

              let sensitive = false;

              if (opts.scan && obj.properties?.text) {
                sensitive = sensitivePatterns.some((pattern) =>
                  pattern.test(obj.properties.text),
                );
                if (sensitive) flagged++;
              }

              try {
                await store.updateProperties(obj.uuid, { sensitive });
                updated++;
              } catch (err) {
                console.error(`Failed to update ${obj.uuid}: ${err}`);
              }
            }

            console.log(`Updated ${updated} memories.`);
            if (opts.scan) {
              console.log(`Flagged ${flagged} as sensitive based on keyword scan.`);
            }
            console.log("Migration complete.");
          });
      },
      { commands: ["wmem"] },
    );

    // ========================================================================
    // Lifecycle Hooks - Auto-Recall
    // ========================================================================

    if (cfg.autoRecall) {
      api.on("before_agent_start", async (event, ctx) => {
        if (!event.prompt || event.prompt.length < 5) return;

        try {
          // Determine if this is a group/shared context
          const filterSensitive = isGroupContext(ctx?.sessionKey);

          // Use hybrid search for best recall
          const results = await store.hybridSearch(
            event.prompt,
            3,
            0.3,
            0.75,
            filterSensitive,
          );

          if (results.length === 0) return;

          const memoryContext = results
            .map(
              (r) =>
                `- [${r.entry.category}] ${r.entry.text} (${(r.score * 100).toFixed(0)}% relevance)`,
            )
            .join("\n");

          api.logger.info?.(
            `memory-weaviate: injecting ${results.length} memories into context (sensitive filtered: ${filterSensitive})`,
          );

          return {
            prependContext: `<relevant-memories>\nThe following long-term memories may be relevant:\n${memoryContext}\n</relevant-memories>`,
          };
        } catch (err) {
          api.logger.warn(
            `memory-weaviate: recall failed: ${String(err)}`,
          );
        }
      });
    }

    // ========================================================================
    // Lifecycle Hooks - Auto-Capture
    // ========================================================================

    if (cfg.autoCapture && extractor) {
      api.on("agent_end", async (event) => {
        if (!event.success || !event.messages || event.messages.length === 0) {
          return;
        }

        try {
          // Build conversation text from the turn's messages
          const parts: string[] = [];
          for (const msg of event.messages) {
            if (!msg || typeof msg !== "object") continue;
            const msgObj = msg as Record<string, unknown>;
            const role = msgObj.role;
            if (role !== "user" && role !== "assistant") continue;

            const content = msgObj.content;
            if (typeof content === "string") {
              parts.push(`[${role}]: ${content}`);
              continue;
            }
            if (Array.isArray(content)) {
              for (const block of content) {
                if (
                  block &&
                  typeof block === "object" &&
                  "type" in block &&
                  (block as Record<string, unknown>).type === "text" &&
                  "text" in block &&
                  typeof (block as Record<string, unknown>).text === "string"
                ) {
                  parts.push(
                    `[${role}]: ${(block as Record<string, unknown>).text as string}`,
                  );
                }
              }
            }
          }

          if (parts.length === 0) return;

          // Truncate to avoid blowing up the extraction call
          const conversationText = parts.join("\n\n").slice(0, 4000);

          // LLM extracts what's worth remembering (including sensitivity classification)
          const extracted = await extractor.extract(conversationText);
          if (extracted.length === 0) return;

          let stored = 0;
          for (const memory of extracted.slice(0, 5)) {
            const vector = await getVector(memory.text);

            // Deduplicate against existing memories
            const existing = vector
              ? await store.search(memory.text, vector, 1, 0.92)
              : await store.hybridSearch(memory.text, 1, 0.92);
            if (existing.length > 0) continue;

            await store.store(
              {
                text: memory.text,
                importance: memory.importance,
                category: memory.category,
                source: "auto-capture",
                sensitive: memory.sensitive,
              },
              vector,
            );
            stored++;
          }

          if (stored > 0) {
            api.logger.info(
              `memory-weaviate: auto-captured ${stored} memories (via ${cfg.extraction.model})`,
            );
          }
        } catch (err) {
          api.logger.warn(
            `memory-weaviate: capture failed: ${String(err)}`,
          );
        }
      });
    }

    // ========================================================================
    // Service
    // ========================================================================

    api.registerService({
      id: "memory-weaviate",
      start: () => {
        api.logger.info(
          `memory-weaviate: started (${cfg.weaviate.url}, collection: ${cfg.collectionName}, sensitivity: enabled)`,
        );
      },
      stop: async () => {
        await store.close();
        api.logger.info("memory-weaviate: stopped");
      },
    });
  },
};

export default memoryPlugin;
