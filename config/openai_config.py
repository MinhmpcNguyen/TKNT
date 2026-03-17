import logging
import os

from cryptography.fernet import Fernet

from config import root_config
from config.models import OpenAIModelGroupConfig

openai_config = root_config.services.openai

private_key_location: str = os.path.join(
    os.path.dirname(__file__), os.pardir, openai_config.private_key_file
)


class OpenAIAccess:
    @staticmethod
    def _encrypt_string(input_string: str):
        key = Fernet.generate_key()
        cipher_suite = Fernet(key)

        # Encrypt sensitive data
        encrypted_data = cipher_suite.encrypt(input_string.encode())

        with open(private_key_location, "wb") as config_file:
            _ = config_file.write(encrypted_data)

        logging.info(f"Encrypted in {private_key_location}!")
        logging.info(f"The public key = {key}, please save it for further processing!")

    @staticmethod
    def get_openai_access() -> str | None:
        try:
            cipher_suite: Fernet = Fernet(key=openai_config.public_key)
            with open(file=private_key_location, mode="rb") as config_file:
                encrypted_data: bytes = config_file.read()
                decrypted_data: str = cipher_suite.decrypt(token=encrypted_data).decode()
                return decrypted_data
        except Exception:
            logging.exception(msg="Unexpected error during initialization OpenAI Access")


class OpenAIConfig:
    IS_OPENAI_AZURE: bool = openai_config.azure
    OPENAI_API_KEY: str | None = OpenAIAccess.get_openai_access()
    API_VERSION: str = openai_config.api_version
    MODELS: OpenAIModelGroupConfig = openai_config.models
