// ============================================================================
// Types
// ============================================================================

export type EmbeddingProvider = "openai" | "weaviate";

export type ExtractionConfig = {
  /** OpenAI-compatible base URL (e.g. http://localhost:11434/v1 for Ollama, http://localhost:1234/v1 for LM Studio) */
  baseUrl?: string;
  /** API key - required for OpenAI, optional/ignored for local providers */
  apiKey?: string;
  /** Model name (e.g. "gpt-4o-mini", "llama3.2", "qwen2.5-coder") */
  model: string;
  /** Max tokens for extraction response */
  maxTokens: number;
};

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
  extraction: ExtractionConfig;
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
const DEFAULT_EXTRACTION_MODEL = "gpt-4o-mini";
const DEFAULT_EXTRACTION_MAX_TOKENS = 1024;
const DEFAULT_COLLECTION = "ClawdbotMemory";

// ============================================================================
// Dimensions lookup (for OpenAI provider only)
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

function resolveOptionalEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (match, envVar) => {
    return process.env[envVar] ?? match;
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
      ["weaviate", "embedding", "extraction", "collectionName", "autoCapture", "autoRecall"],
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

    // Extraction config (LLM-based memory extraction)
    const extraction = (cfg.extraction as Record<string, unknown>) ?? {};
    assertAllowedKeys(extraction, ["baseUrl", "apiKey", "model", "maxTokens"], "extraction config");

    const extractionBaseUrl =
      typeof extraction.baseUrl === "string"
        ? resolveOptionalEnvVars(extraction.baseUrl)
        : undefined;

    const extractionApiKey =
      typeof extraction.apiKey === "string"
        ? resolveEnvVars(extraction.apiKey)
        : undefined;

    const extractionModel =
      typeof extraction.model === "string"
        ? extraction.model
        : DEFAULT_EXTRACTION_MODEL;

    const extractionMaxTokens =
      typeof extraction.maxTokens === "number"
        ? extraction.maxTokens
        : DEFAULT_EXTRACTION_MAX_TOKENS;

    // If no explicit extraction API key and no baseUrl, fall back to embedding API key
    // (backwards compat: OpenAI key used for both embedding + extraction)
    const resolvedExtractionApiKey =
      extractionApiKey ??
      (typeof embedding.apiKey === "string" && !extractionBaseUrl
        ? resolveEnvVars(embedding.apiKey)
        : undefined);

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
      extraction: {
        baseUrl: extractionBaseUrl,
        apiKey: resolvedExtractionApiKey,
        model: extractionModel,
        maxTokens: extractionMaxTokens,
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
