from enum import Enum


class OpenAIEmbedModels(str, Enum):
    TEXT_EMBED_3_SMALL = "text-embedding-3-small"
    TEXT_EMBED_3_LARGE = "text-embedding-3-large"
    ADA_002 = "text-embedding-ada-002"


class OpenAILLMModels(str, Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_35_TURBO = "gpt-3.5-turbo"


class OpenAIModelConfig:
    """OpenAI model configurations"""

    # Embedding settings
    EMBEDDING_MODEL = OpenAIEmbedModels.TEXT_EMBED_3_SMALL
    EMBEDDING_DIMENSIONS = 1024
    EMBEDDING_BATCH_SIZE = 2048
    EMBEDDING_CONTEXT_SIZE = 8192

    # LLM settings
    LLM_MODEL = OpenAILLMModels.GPT_4O_MINI
    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 4096
