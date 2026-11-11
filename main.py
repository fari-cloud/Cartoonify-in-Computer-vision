import os
import sys

from mistralai.client import Mistral


def main() -> None:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        sys.exit("Error: MISTRAL_API_KEY is not set in the environment.")

    client = Mistral(api_key=api_key)

    chat_response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?",
            },
        ],
    )

    print(chat_response.choices[0].message.content)


if __name__ == "__main__":
    main()
