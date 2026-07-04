import os

from config import root_config
from config.models import OpenAIModelConfig, OpenAIModelGroupConfig

openai_config = root_config.services.openai
_BENCHMARK_TEXT_MODEL_ENV = "TKNT_BENCHMARK_TEXT_MODEL"


def _configured_value(value: str | None) -> str | None:
    if value is None:
        return None
    clean = value.strip()
    if not clean or (clean.startswith("${") and clean.endswith("}")):
        return None
    return clean


def _benchmark_text_model_name() -> str:
    return os.getenv(_BENCHMARK_TEXT_MODEL_ENV, "").strip()


def _generation_models() -> OpenAIModelGroupConfig:
    models = openai_config.models
    forced_model_name = _benchmark_text_model_name()
    if not forced_model_name:
        return models
    return OpenAIModelGroupConfig(
        primary=OpenAIModelConfig(name=forced_model_name),
        helper=OpenAIModelConfig(name=forced_model_name),
        embedding=models.embedding_model(),
    )


def _agent_models() -> dict[str, str]:
    forced_model_name = _benchmark_text_model_name()
    if forced_model_name:
        return {
            name: forced_model_name
            for name in openai_config.agent_models
            if name.strip()
        }
    return {name: model.name for name, model in openai_config.agent_models.items()}


def _openai_api_key() -> str | None:
    return _configured_value(openai_config.api_key)


class OpenAIConfig:
    ENABLED: bool = openai_config.enabled
    IS_OPENAI_AZURE: bool = openai_config.azure
    OPENAI_API_KEY: str | None = _openai_api_key()
    API_VERSION: str = openai_config.api_version
    BENCHMARK_TEXT_MODEL: str = _benchmark_text_model_name()
    MODELS: OpenAIModelGroupConfig = _generation_models()
    AGENT_MODELS: dict[str, str] = _agent_models()
