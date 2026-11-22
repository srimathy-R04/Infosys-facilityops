import os, json, urllib.request as u, urllib.error as ue

key = os.environ.get("GEMINI_API_KEY", "")
print("key length:", len(key), "| starts with:", key[:3])
H = {"Content-Type": "application/json", "x-goog-api-key": key}
BASE = "https://generativelanguage.googleapis.com/v1beta/"

# 1) Can the key list models at all?
usable = []
try:
    data = json.load(u.urlopen(u.Request(BASE + "models?pageSize=200", headers=H)))
    for m in data.get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            usable.append(m["name"].replace("models/", ""))
    print("LIST OK. Models that can generate text:")
    print([n for n in usable if "flash" in n][:15])
except ue.HTTPError as e:
    print("LIST FAILED", e.code, e.read().decode()[:200])

# 2) Try generating with the newest flash models first
order = [n for n in usable if n.startswith("gemini-3") and "flash" in n and "image" not in n and "tts" not in n]
order = order + ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash"]
for model in dict.fromkeys(order):
    req = u.Request(BASE + f"models/{model}:generateContent",
                    json.dumps({"contents": [{"parts": [{"text": "say hi"}]}]}).encode(), H)
    try:
        print(model, "OK:", json.load(u.urlopen(req))["candidates"][0]["content"]["parts"][0]["text"])
        break
    except ue.HTTPError as e:
        print(model, "FAILED", e.code, e.read().decode()[:120].replace("\n", " "))