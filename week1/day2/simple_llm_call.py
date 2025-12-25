import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables from .env
load_dotenv()

# 2. Read API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Please set it in your .env file.")

# 3. Create OpenAI client
client = OpenAI(api_key=api_key)

def ask_llm(prompt: str) -> str:
    """
    Send a prompt to the OpenAI LLM and return the response text.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",      # lightweight, cheap, good for practice
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI tutor. "
                    "Explain things clearly and concisely for a software engineer transitioning into AI."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.4,           # lower = more focused & deterministic
    )

    # 4. Extract the text content from the first choice
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    # 5. Test prompt
    user_prompt = "Explain what an embedding is in one short paragraph, like I'm a backend engineer."
    answer = ask_llm(user_prompt)
    print("🧠 LLM RESPONSE:\n")
    print(answer)
