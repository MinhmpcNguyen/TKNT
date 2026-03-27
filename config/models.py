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
    gpt_4o: OpenAIModelConfig
    gpt_4o_mini: OpenAIModelConfig
    text_embedding_3_small: OpenAIModelConfig


class OllamaModelConfig(BaseModel):
    name: str


class OllamaModelGroupConfig(BaseModel):
    primary: OllamaModelConfig
    helper: OllamaModelConfig
    embedding: OllamaModelConfig


class MistralModelConfig(BaseModel):
    name: str


class MistralModelGroupConfig(BaseModel):
    primary: MistralModelConfig
    helper: MistralModelConfig
    embedding: MistralModelConfig


class OpenAIConfig(BaseModel):
    azure: bool
    api_version: str
    api_key: str | None = None
    public_key: str | None = None
    private_key_file: str | None = None
    models: OpenAIModelGroupConfig

    @model_validator(mode="after")
    def validate_access(self) -> Self:
        if self.api_key:
            return self
        if self.public_key and self.private_key_file:
            return self
        raise ValueError(
            "Configure either services.openai.api_key or both "
            "services.openai.public_key and services.openai.private_key_file."
        )

    @model_validator(mode="after")
    def validate_azure_endpoints(self) -> Self:
        if not self.azure:
            return self

        missing_models: list[str] = []
        for model_name, model_config in (
            ("gpt_4o", self.models.gpt_4o),
            ("gpt_4o_mini", self.models.gpt_4o_mini),
            ("text_embedding_3_small", self.models.text_embedding_3_small),
        ):
            if not model_config.azure_endpoint:
                missing_models.append(model_name)

        if missing_models:
            raise ValueError(
                "Missing azure_endpoint for Azure models: "
                + ", ".join(sorted(missing_models))
            )
        return self


class OllamaConfig(BaseModel):
    enabled: bool = False
    api_key: str = "ollama"
    base_url: str | None = None
    models: OllamaModelGroupConfig | None = None

    @model_validator(mode="after")
    def validate_enabled_config(self) -> Self:
        if not self.enabled:
            return self
        if not self.base_url:
            raise ValueError(
                "services.ollama.base_url is required when Ollama is enabled."
            )
        if self.models is None:
            raise ValueError(
                "services.ollama.models is required when Ollama is enabled."
            )
        return self


class MistralConfig(BaseModel):
    enabled: bool = False
    api_key: str = "mistral"
    base_url: str | None = None
    models: MistralModelGroupConfig | None = None

    @model_validator(mode="after")
    def validate_enabled_config(self) -> Self:
        if not self.enabled:
            return self
        if not self.base_url:
            raise ValueError(
                "services.mistral.base_url is required when Mistral is enabled."
            )
        if self.models is None:
            raise ValueError(
                "services.mistral.models is required when Mistral is enabled."
            )
        return self


class SemanticSearchConfig(BaseModel):
    enabled: bool = True


class Service(BaseModel):
    openai: OpenAIConfig
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    mistral: MistralConfig = Field(default_factory=MistralConfig)
    semantic_search: SemanticSearchConfig = Field(default_factory=SemanticSearchConfig)

    @model_validator(mode="after")
    def validate_provider_selection(self) -> Self:
        if self.ollama.enabled and self.mistral.enabled:
            raise ValueError(
                "Enable only one local provider at a time: services.ollama or services.mistral."
            )
        return self


class RootConfig(BaseModel):
    authentication: Auth
    services: Service
