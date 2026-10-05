"""Shared helpers: config loading, Gemini client, small utilities."""
import os
import time
import yaml
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

_CONFIG = None


def config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        with open(ROOT / "config.yaml") as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def gemini_client():
    """Return a configured google-genai client. Import is lazy so that
    data-prep steps that don't need the API can run without a key."""
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")
    return genai.Client(api_key=key)


def retry(fn, tries=4, base=2.0):
    """Minimal exponential backoff for transient API errors."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i == tries - 1:
                break
            time.sleep(base * (2 ** i))
    raise last


def _gen_config(json_schema=None, temperature=0.0):
    from google.genai import types
    kwargs = dict(
        temperature=temperature,
        # We never want the model picking tools; keep calls pure text/JSON.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if json_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = json_schema
    return types.GenerateContentConfig(**kwargs)


def generate_text(client, model, prompt, temperature=0.0):
    def call():
        return client.models.generate_content(
            model=model, contents=prompt, config=_gen_config(temperature=temperature)
        )
    return retry(call).text.strip()


def generate_json(client, model, prompt, schema, temperature=0.0):
    """Structured generation; `schema` is a pydantic model. Returns parsed JSON."""
    import json as _json

    def call():
        return client.models.generate_content(
            model=model, contents=prompt, config=_gen_config(schema, temperature)
        )
    resp = retry(call)
    return _json.loads(resp.text)
