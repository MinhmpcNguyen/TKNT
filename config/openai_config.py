import logging
import os

from cryptography.fernet import Fernet

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


def _resolve_private_key_location() -> str | None:
    private_key_file = _configured_value(openai_config.private_key_file)
    if private_key_file is None:
        return None
    return os.path.join(os.path.dirname(__file__), os.pardir, private_key_file)


PRIVATE_KEY_LOCATION = _resolve_private_key_location()


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


class OpenAIAccess:
    @staticmethod
    def _encrypt_string(input_string: str) -> None:
        if PRIVATE_KEY_LOCATION is None:
            raise ValueError("Missing services.openai.private_key_file configuration.")

        key = Fernet.generate_key()
        cipher_suite = Fernet(key)

        # Encrypt sensitive data
        encrypted_data = cipher_suite.encrypt(input_string.encode())

        with open(PRIVATE_KEY_LOCATION, "wb") as config_file:
            _ = config_file.write(encrypted_data)

        logging.info("Encrypted in %s!", PRIVATE_KEY_LOCATION)
        logging.info(f"The public key = {key}, please save it for further processing!")

    @staticmethod
    def get_openai_access() -> str | None:
        api_key = _configured_value(openai_config.api_key)
        if api_key:
            return api_key

        public_key = _configured_value(openai_config.public_key)
        if not public_key or PRIVATE_KEY_LOCATION is None:
            return None

        try:
            cipher_suite: Fernet = Fernet(key=public_key)
            with open(file=PRIVATE_KEY_LOCATION, mode="rb") as config_file:
                encrypted_data: bytes = config_file.read()
                decrypted_data: str = cipher_suite.decrypt(
                    token=encrypted_data
                ).decode()
                return decrypted_data
        except Exception:
            logging.exception(
                msg="Unexpected error during initialization OpenAI Access"
            )
            return None


class OpenAIConfig:
    ENABLED: bool = openai_config.enabled
    IS_OPENAI_AZURE: bool = openai_config.azure
    OPENAI_API_KEY: str | None = OpenAIAccess.get_openai_access()
    API_VERSION: str = openai_config.api_version
    BENCHMARK_TEXT_MODEL: str = _benchmark_text_model_name()
    MODELS: OpenAIModelGroupConfig = _generation_models()
    AGENT_MODELS: dict[str, str] = _agent_models()
