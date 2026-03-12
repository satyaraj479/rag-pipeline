import os
import anthropic
from dotenv import load_dotenv

load_dotenv()


def test_with_api_key():
    """Test Claude using ANTHROPIC_API_KEY environment variable."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[api_key] ANTHROPIC_API_KEY not set, skipping.")
        return

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say 'API key works!' and nothing else."}],
    )
    print(f"[api_key] Response : {message.content[0].text}")
    print(f"[api_key] Tokens   : input={message.usage.input_tokens}, output={message.usage.output_tokens}")


def test_with_auth_token():
    """Test Claude using a bearer auth token (ANTHROPIC_AUTH_TOKEN)."""
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not auth_token:
        print("[auth_token] ANTHROPIC_AUTH_TOKEN not set, skipping.")
        return

    client = anthropic.Anthropic(auth_token=auth_token)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say 'Auth token works!' and nothing else."}],
    )
    print(f"[auth_token] Response : {message.content[0].text}")
    print(f"[auth_token] Tokens   : input={message.usage.input_tokens}, output={message.usage.output_tokens}")


def main():
    test_with_api_key()
    test_with_auth_token()


if __name__ == "__main__":
    main()
