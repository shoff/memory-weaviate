import { homedir } from "node:os";

// ============================================================================
// Types
// ============================================================================

export type EmbeddingProvider = "openai" | "weaviate";

export type MemoryConfig = {
  weaviate: {
    url: string;
    apiKey?: string;
  };
  embedding: {
    provider: EmbeddingProvider;
    apiKey?: string;
    model: string;
  };
  collectionName: string;
  autoCapture: boolean;
  autoRecall: boolean;
};

export const MEMORY_CATEGORIES = [
  "preference",
  "fact",
  "decision",
  "entity",
  "conversation",
  "other",
] as const;

export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

// ============================================================================
// Defaults
// ============================================================================

const DEFAULT_WEAVIATE_URL = "http://localhost:8080";
const DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small";
const DEFAULT_COLLECTION = "ClawdbotMemory";

// ============================================================================
// Dimensions lookup (for OpenAI provider)
// ============================================================================

const EMBEDDING_DIMENSIONS: Record<string, number> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
};

export function vectorDimsForModel(model: string): number {
  const dims = EMBEDDING_DIMENSIONS[model];
  if (!dims) {
    throw new Error(`Unsupported embedding model: ${model}`);
  }
  return dims;
}

// ============================================================================
// Env var resolution
// ============================================================================

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

function assertAllowedKeys(
  value: Record<string, unknown>,
  allowed: string[],
  label: string,
) {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length === 0) return;
  throw new Error(`${label} has unknown keys: ${unknown.join(", ")}`);
}

// ============================================================================
// Config parser
// ============================================================================

export const memoryConfigSchema = {
  parse(value: unknown): MemoryConfig {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("memory-weaviate config required");
    }
    const cfg = value as Record<string, unknown>;
    assertAllowedKeys(
      cfg,
      ["weaviate", "embedding", "collectionName", "autoCapture", "autoRecall"],
      "memory-weaviate config",
    );

    // Weaviate connection
    const weaviate = cfg.weaviate as Record<string, unknown> | undefined;
    if (!weaviate || typeof weaviate.url !== "string") {
      throw new Error("weaviate.url is required");
    }
    assertAllowedKeys(weaviate, ["url", "apiKey"], "weaviate config");

    // Embedding config
    const embedding = (cfg.embedding as Record<string, unknown>) ?? {};
    assertAllowedKeys(embedding, ["provider", "apiKey", "model"], "embedding config");

    const provider = (embedding.provider as EmbeddingProvider) ?? "openai";
    if (provider === "openai" && typeof embedding.apiKey !== "string") {
      throw new Error(
        "embedding.apiKey is required when provider is 'openai'",
      );
    }

    const model =
      typeof embedding.model === "string"
        ? embedding.model
        : DEFAULT_EMBEDDING_MODEL;

    if (provider === "openai") {
      vectorDimsForModel(model); // validate
    }

    return {
      weaviate: {
        url: resolveEnvVars(weaviate.url),
        apiKey:
          typeof weaviate.apiKey === "string"
            ? resolveEnvVars(weaviate.apiKey)
            : undefined,
      },
      embedding: {
        provider,
        apiKey:
          typeof embedding.apiKey === "string"
            ? resolveEnvVars(embedding.apiKey)
            : undefined,
        model,
      },
      collectionName:
        typeof cfg.collectionName === "string"
          ? cfg.collectionName
          : DEFAULT_COLLECTION,
      autoCapture: cfg.autoCapture !== false,
      autoRecall: cfg.autoRecall !== false,
    };
  },
};
