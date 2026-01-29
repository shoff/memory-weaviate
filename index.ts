/**
 * Clawdbot Memory (Weaviate) Plugin
 *
 * Long-term vector memory backed by Weaviate.
 * Supports OpenAI embeddings or Weaviate's built-in vectorizer.
 * Provides auto-recall (context injection) and auto-capture (conversation mining).
 */

import { Type } from "@sinclair/typebox";
import weaviate, {
  type WeaviateClient,
  type Collection,
} from "weaviate-client";
import OpenAI from "openai";
import { randomUUID } from "node:crypto";
import type { ClawdbotPluginApi } from "clawdbot/plugin-sdk";
import { stringEnum } from "clawdbot/plugin-sdk";

import {
  MEMORY_CATEGORIES,
  type MemoryCategory,
  type MemoryConfig,
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
  sessionKey?: string;
  createdAt: number;
};

type MemorySearchResult = {
  entry: MemoryEntry;
  score: number;
};

// ============================================================================
// Weaviate Memory Store
// ============================================================================

class WeaviateMemoryStore {
  private client: WeaviateClient | null = null;
  private collection: Collection | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(
    private readonly config: MemoryConfig,
  ) {}

  private async ensureInitialized(): Promise<void> {
    if (this.collection) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<void> {
    // Connect to Weaviate
    const connectOpts: Parameters<typeof weaviate.connectToLocal>[0] = {
      host: new URL(this.config.weaviate.url).hostname,
      port: parseInt(new URL(this.config.weaviate.url).port || "8080"),
      grpcPort: 50051,
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
        { name: "sessionKey", dataType: "text" },
        { name: "createdAt", dataType: "int" },
      ],
    };

    // If using Weaviate's built-in vectorizer, configure it
    // Otherwise we'll provide vectors manually (OpenAI provider)
    if (this.config.embedding.provider === "weaviate") {
      // Uses whatever vectorizer module is configured in Weaviate
      // (e.g., text2vec-openai, text2vec-transformers)
      collectionConfig.vectorizers = [
        weaviate.configure.vectorizer.text2VecOpenAI({
          model: this.config.embedding.model,
          sourceProperties: ["text"],
        }),
      ];
    } else {
      // No vectorizer - we provide vectors ourselves
      collectionConfig.vectorizers = [
        weaviate.configure.vectorizer.none({
          vectorIndexConfig: weaviate.configure.vectorIndex.hnsw(),
        }),
      ];
    }

    await this.client!.collections.create(collectionConfig);
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
      sessionKey: entry.sessionKey ?? "",
      createdAt,
    };

    if (vector && this.config.embedding.provider === "openai") {
      await this.collection!.data.insert({
        properties,
        id,
        vectors: vector,
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
  ): Promise<MemorySearchResult[]> {
    await this.ensureInitialized();

    let results: any;

    if (vector && this.config.embedding.provider === "openai") {
      // Vector search with provided embeddings
      results = await this.collection!.query.nearVector(vector, {
        limit,
        returnMetadata: ["distance"],
        returnProperties: [
          "text",
          "importance",
          "category",
          "source",
          "sessionKey",
          "createdAt",
        ],
      });
    } else {
      // Use Weaviate's built-in vectorizer with nearText
      results = await this.collection!.query.nearText(queryText, {
        limit,
        returnMetadata: ["distance"],
        returnProperties: [
          "text",
          "importance",
          "category",
          "source",
          "sessionKey",
          "createdAt",
        ],
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
  ): Promise<MemorySearchResult[]> {
    await this.ensureInitialized();

    const results = await this.collection!.query.hybrid(queryText, {
      limit,
      alpha,
      returnMetadata: ["score"],
      returnProperties: [
        "text",
        "importance",
        "category",
        "source",
        "sessionKey",
        "createdAt",
      ],
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
    const result = await this.collection!.aggregate.overAll({
      returnMetrics: ["meta"],
    });
    return (result as any)?.meta?.count ?? 0;
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
// Auto-capture triggers
// ============================================================================

const MEMORY_TRIGGERS = [
  /remember|don't forget|keep in mind/i,
  /i prefer|i like|i hate|i love|i want|i need/i,
  /we decided|we agreed|going with|we'll use/i,
  /my name is|i'm called|call me/i,
  /my .+ is|i work at|i live in/i,
  /always|never|important to me/i,
  /\+\d{10,}/, // phone numbers
  /[\w.-]+@[\w.-]+\.\w+/, // emails
];

function shouldCapture(text: string): boolean {
  if (text.length < 10 || text.length > 1000) return false;
  if (text.includes("<relevant-memories>")) return false;
  if (text.startsWith("<") && text.includes("</")) return false;
  if (text.includes("**") && text.includes("\n-")) return false;
  const emojiCount = (text.match(/[\u{1F300}-\u{1F9FF}]/gu) || []).length;
  if (emojiCount > 3) return false;
  return MEMORY_TRIGGERS.some((r) => r.test(text));
}

function detectCategory(text: string): MemoryCategory {
  const lower = text.toLowerCase();
  if (/prefer|like|love|hate|want|favorite/i.test(lower)) return "preference";
  if (/decided|agreed|will use|going with/i.test(lower)) return "decision";
  if (
    /\+\d{10,}|@[\w.-]+\.\w+|name is|called|works at|lives in/i.test(lower)
  )
    return "entity";
  if (/is|are|has|have/i.test(lower)) return "fact";
  return "other";
}

// ============================================================================
// Plugin
// ============================================================================

const memoryPlugin = {
  id: "memory-weaviate",
  name: "Memory (Weaviate)",
  description:
    "Weaviate-backed long-term vector memory with hybrid search, auto-recall, and auto-capture",
  kind: "memory" as const,
  configSchema: memoryConfigSchema,

  register(api: ClawdbotPluginApi) {
    const cfg = memoryConfigSchema.parse(api.pluginConfig);
    const store = new WeaviateMemoryStore(cfg);

    // Optional OpenAI embeddings client (for provider = "openai")
    const embeddings =
      cfg.embedding.provider === "openai" && cfg.embedding.apiKey
        ? new Embeddings(cfg.embedding.apiKey, cfg.embedding.model)
        : null;

    api.logger.info(
      `memory-weaviate: registered (url: ${cfg.weaviate.url}, collection: ${cfg.collectionName}, embedding: ${cfg.embedding.provider}, lazy init)`,
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
        }),
        async execute(_toolCallId, params) {
          const {
            text,
            importance = 0.7,
            category = "other",
          } = params as {
            text: string;
            importance?: number;
            category?: MemoryCategory;
          };

          // Check for near-duplicates
          const vector = await getVector(text);
          const existing = vector
            ? await store.search(text, vector, 1, 0.95)
            : await store.hybridSearch(text, 1, 0.95);

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
            { text, importance, category, source: "manual" },
            vector,
          );

          return {
            content: [
              { type: "text", text: `Stored: "${text.slice(0, 100)}${text.length > 100 ? "..." : ""}"` },
            ],
            details: { action: "created", id: entry.id },
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
                text: `Memory store: ${count} memories in collection "${cfg.collectionName}" on ${cfg.weaviate.url}`,
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
          .action(async (query: string, opts: any) => {
            const limit = parseInt(opts.limit);
            let results: MemorySearchResult[];

            if (opts.mode === "hybrid") {
              results = await store.hybridSearch(query, limit, 0.1);
            } else {
              const vector = await getVector(query);
              results = await store.search(query, vector, limit, 0.1);
            }

            const output = results.map((r) => ({
              id: r.entry.id,
              text: r.entry.text,
              category: r.entry.category,
              importance: r.entry.importance,
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
          .action(async (text: string, opts: any) => {
            const vector = await getVector(text);
            const entry = await store.store(
              {
                text,
                importance: parseFloat(opts.importance),
                category: opts.category as MemoryCategory,
                source: "manual",
              },
              vector,
            );
            console.log(`Stored: ${entry.id}`);
          });

        mem
          .command("forget")
          .description("Delete a memory by ID")
          .argument("<id>", "Memory UUID")
          .action(async (id: string) => {
            await store.delete(id);
            console.log(`Deleted: ${id}`);
          });
      },
      { commands: ["wmem"] },
    );

    // ========================================================================
    // Lifecycle Hooks - Auto-Recall
    // ========================================================================

    if (cfg.autoRecall) {
      api.on("before_agent_start", async (event) => {
        if (!event.prompt || event.prompt.length < 5) return;

        try {
          // Use hybrid search for best recall
          const results = await store.hybridSearch(
            event.prompt,
            3,
            0.3,
            0.75,
          );

          if (results.length === 0) return;

          const memoryContext = results
            .map(
              (r) =>
                `- [${r.entry.category}] ${r.entry.text} (${(r.score * 100).toFixed(0)}% relevance)`,
            )
            .join("\n");

          api.logger.info?.(
            `memory-weaviate: injecting ${results.length} memories into context`,
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

    if (cfg.autoCapture) {
      api.on("agent_end", async (event) => {
        if (!event.success || !event.messages || event.messages.length === 0) {
          return;
        }

        try {
          const texts: string[] = [];
          for (const msg of event.messages) {
            if (!msg || typeof msg !== "object") continue;
            const msgObj = msg as Record<string, unknown>;
            const role = msgObj.role;
            if (role !== "user" && role !== "assistant") continue;

            const content = msgObj.content;
            if (typeof content === "string") {
              texts.push(content);
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
                  texts.push(
                    (block as Record<string, unknown>).text as string,
                  );
                }
              }
            }
          }

          const toCapture = texts.filter((t) => t && shouldCapture(t));
          if (toCapture.length === 0) return;

          let stored = 0;
          for (const text of toCapture.slice(0, 5)) {
            const category = detectCategory(text);
            const vector = await getVector(text);

            // Deduplicate
            const existing = vector
              ? await store.search(text, vector, 1, 0.95)
              : await store.hybridSearch(text, 1, 0.95);
            if (existing.length > 0) continue;

            await store.store(
              {
                text,
                importance: 0.7,
                category,
                source: "auto-capture",
              },
              vector,
            );
            stored++;
          }

          if (stored > 0) {
            api.logger.info(
              `memory-weaviate: auto-captured ${stored} memories`,
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
          `memory-weaviate: started (${cfg.weaviate.url}, collection: ${cfg.collectionName})`,
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
