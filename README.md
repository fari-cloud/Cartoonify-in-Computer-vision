# Cartoonify-in-Computer-vision

## Mistral chat completion example

Minimal chat completion with the official Mistral Python SDK, using `mistral-small-latest`.

### 1. Generate an API key

1. Sign in at [console.mistral.ai](https://console.mistral.ai) (La Plateforme / Mistral Studio).
2. Go to **API Keys** (or your workspace settings → API Keys).
3. Click **Create new key**, give it a name, and copy the generated key — it is shown only once.

### 2. Install the SDK

```bash
pip install -r requirements.txt
```

### 3. Set the key in your terminal

```bash
export MISTRAL_API_KEY="your-api-key-here"
```

(Windows PowerShell: `$env:MISTRAL_API_KEY="your-api-key-here"`)

### 4. Run it

```bash
python main.py
```

Reference: [Chat completions docs](https://docs.mistral.ai/inference/conversations/chat-completion)