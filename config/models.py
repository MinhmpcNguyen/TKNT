from pydantic import BaseModel, model_validator
from typing_extensions import Self


class APIKeyConfig(BaseModel):
    name: str
    value: str


class Auth(BaseModel):
    api_key: APIKeyConfig


class OpenAIModelConfig(BaseModel):
    name: str
    azure_endpoint: str


class OpenAIModelGroupConfig(BaseModel):
    gpt_4o: OpenAIModelConfig
    gpt_4o_mini: OpenAIModelConfig
    text_embedding_3_small: OpenAIModelConfig


class OpenAIConfig(BaseModel):
    azure: bool
    api_version: str
    public_key: str
    private_key_file: str
    models: OpenAIModelGroupConfig

    @model_validator(mode="after")
    def validate_api_key(self) -> Self:
        if not self.public_key:
            raise ValueError("Missing OpenAI Public API Key")
        return self


class Service(BaseModel):
    openai: OpenAIConfig


class RootConfig(BaseModel):
    authentication: Auth
    services: Service
