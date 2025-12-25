import json
import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# Load env vars
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found in .env")

client = OpenAI()

# ----------------------------
# Output contract (schema)
# ----------------------------
class Summary(BaseModel):
    tldr: str = Field(..., description="1–2 sentence summary")
    key_points: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)

SYSTEM_PROMPT = (
    "You are a precise AI assistant. "
    "You MUST return ONLY valid JSON that matches the schema. "
    "No markdown, no explanations, no extra text."
)

def _extract_json(text: str) -> str:
    """Fallback in case model wraps JSON with text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found in model output")
    return text[start : end + 1]

def summarize_text(text: str) -> Summary:
    schema_hint = {
        "tldr": "string",
        "key_points": ["string"],
        "action_items": ["string"],
        "risks": ["string"],
        "open_questions": ["string"],
    }

    user_prompt = f"""
Summarize the TEXT below using the JSON schema.

Schema:
{json.dumps(schema_hint, indent=2)}

Rules:
- Return ONLY JSON
- TLDR max 2 sentences
- key_points: 3–6 bullets
- Use empty arrays [] if nothing applies

TEXT:
{text}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()

    try:
        return Summary.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        repaired = _extract_json(raw)
        return Summary.model_validate(json.loads(repaired))


if __name__ == "__main__":
    sample_text = """
We discussed migrating on-prem workloads to cloud infrastructure.
Security review must be completed before migration.
Ganesh will draft the migration plan by Friday.
Risk: delays if security review is not completed on time.
Open question: which region should host the primary workloads?
"""

    summary = summarize_text(sample_text)
    print(summary.model_dump_json(indent=2))
