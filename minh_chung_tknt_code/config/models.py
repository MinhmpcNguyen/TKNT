from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class APIKeyConfig(BaseModel):
    name: str
    value: str


class Auth(BaseModel):
    api_key: APIKeyConfig


class OpenAIModelConfig(BaseModel):
    name: str
    azure_endpoint: str | None = None


class OpenAIModelGroupConfig(BaseModel):
    primary: OpenAIModelConfig | None = None
    helper: OpenAIModelConfig | None = None
    embedding: OpenAIModelConfig | None = None

    def primary_model(self) -> OpenAIModelConfig:
        return self._require_model(self.primary, "primary")

    def helper_model(self) -> OpenAIModelConfig:
        return self._require_model(self.helper, "helper")

    def embedding_model(self) -> OpenAIModelConfig:
        return self._require_model(self.embedding, "embedding")

    def resolved_models(self) -> dict[str, OpenAIModelConfig]:
        return {
            "primary": self.primary_model(),
            "helper": self.helper_model(),
            "embedding": self.embedding_model(),
        }

    @staticmethod
    def _require_model(
        model_config: OpenAIModelConfig | None,
        config_key: str,
    ) -> OpenAIModelConfig:
        if model_config is None:
            raise ValueError(f"services.openai.models.{config_key} is required.")
        return model_config


class OpenAIConfig(BaseModel):
    enabled: bool = False
    azure: bool = False
    api_version: str = "2024-08-01-preview"
    api_key: str | None = None
    models: OpenAIModelGroupConfig
    agent_models: dict[str, OpenAIModelConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_access(self) -> Self:
        if not self.enabled:
            return self
        if self.api_key:
            return self
        raise ValueError("Configure services.openai.api_key.")

    @model_validator(mode="after")
    def validate_azure_endpoints(self) -> Self:
        if not self.azure:
            return self

        missing_models: list[str] = []
        for model_name, model_config in self.models.resolved_models().items():
            if not model_config.azure_endpoint:
                missing_models.append(model_name)

        if missing_models:
            raise ValueError(
                "Missing azure_endpoint for Azure models: "
                + ", ".join(sorted(missing_models))
            )
        return self


class SemanticSearchConfig(BaseModel):
    enabled: bool = True


class PresetOptionConfig(BaseModel):
    label: str
    prompt_suffix: str | None = None
    reference_image: str | None = None


class PresetsConfig(BaseModel):
    lights: dict[str, PresetOptionConfig] = Field(default_factory=dict)
    sceneries: dict[str, PresetOptionConfig] = Field(default_factory=dict)
    styles: dict[str, PresetOptionConfig] = Field(default_factory=dict)


class Service(BaseModel):
    openai: OpenAIConfig
    semantic_search: SemanticSearchConfig = Field(default_factory=SemanticSearchConfig)


class RootConfig(BaseModel):
    authentication: Auth
    services: Service
    presets: PresetsConfig = Field(default_factory=PresetsConfig)
