from config import root_config
from config.models import GeminiModelGroupConfig

gemini_config = root_config.services.gemini


class GeminiConfig:
    ENABLED: bool = gemini_config.enabled
    API_KEY: str | None = gemini_config.api_key
    BASE_URL: str = gemini_config.base_url
    MODELS: GeminiModelGroupConfig | None = gemini_config.models
    AGENT_MODELS: dict[str, str] = {
        name: model.name for name, model in gemini_config.agent_models.items()
    }
