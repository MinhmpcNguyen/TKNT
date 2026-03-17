from __future__ import annotations

import argparse
import os
from pathlib import Path

from cryptography.fernet import Fernet


def _load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def _get_env_value(env: dict[str, str], key: str) -> str | None:
    if key in os.environ and os.environ[key].strip():
        return os.environ[key].strip()
    return env.get(key)


def _clean_key(value: str) -> str:
    key = value.strip().strip('"').strip("'").strip()
    if key.startswith("b'") and key.endswith("'"):
        key = key[2:-1]
    if key.startswith('b"') and key.endswith('"'):
        key = key[2:-1]
    return key.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Decrypt OpenAI API key from encrypted file.")
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output file path to write decrypted key (prints to stdout if omitted).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env_path = (repo_root / args.env).resolve()
    env = _load_env(env_path)

    public_key = _get_env_value(env, "OPENAI_PUBLIC_KEY")
    private_key_file = _get_env_value(env, "OPENAI_PRIVATE_KEY") or "private_key.enc"

    if not public_key:
        raise SystemExit("Missing OPENAI_PUBLIC_KEY in .env or environment.")

    public_key = _clean_key(public_key)

    private_path = (repo_root / private_key_file).resolve()
    if not private_path.exists():
        raise SystemExit(f"Encrypted key file not found: {private_path}")

    try:
        cipher = Fernet(public_key.encode())
    except Exception as exc:
        raise SystemExit(
            f"Invalid OPENAI_PUBLIC_KEY format. Expected Fernet key (44 chars). Got len={len(public_key)}. Error: {exc}"
        ) from exc

    encrypted_data = private_path.read_bytes()
    try:
        decrypted = cipher.decrypt(encrypted_data).decode("utf-8")
    except Exception as exc:
        raise SystemExit(
            "Failed to decrypt private_key.enc with the provided OPENAI_PUBLIC_KEY. "
            "This usually means the public key does not match the encrypted file. "
            "Re-encrypt the private key with the current public key or update .env to the matching key."
        ) from exc

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.write_text(decrypted)
    else:
        print(decrypted)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
