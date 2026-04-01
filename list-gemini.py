from google import genai

client = genai.Client(api_key="AIzaSyBDyF0vlgztCNrYebc34hSjxT74WbRkGtE")

for model in client.models.list():
    name = getattr(model, "name", "")
    methods = (
        getattr(model, "supported_actions", None)
        or getattr(model, "supported_generation_methods", None)
        or []
    )
    if "generateContent" in methods and (
        "flash" in name or "lite" in name or "gemma" in name
    ):
        print(name)
