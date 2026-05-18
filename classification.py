from __future__ import annotations

from typing import Optional
from ollama import AsyncClient


SYSTEM_PROMPT = """
You are a hate speech classification system.

Classify the given text into exactly one of the following labels:
hatespeech
offensive
normal

Return only the label. Do not explain your answer.
""".strip()


async def classify_query(
    client: AsyncClient,
    model: str,
    query_id: str | int,
    text: str,
    seed: int,
    think: bool = False,
    persona_prompt: Optional[str] = None,
) -> dict:
    """
    Runs one classification query against one Ollama model.
    """
    system_prompt = SYSTEM_PROMPT

    if persona_prompt:
        system_prompt = persona_prompt.strip() + "\n\n" + SYSTEM_PROMPT

    response = await client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text:\n{text}"},
        ],
        stream=False,
        think=think,
        options={
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": seed,
            "num_ctx": 1024
        },
    )

    raw_label = response["message"]["content"].strip()

    return {
        "query_id": query_id,
        "model": model,
        "seed": seed,
        "text": text,
        "raw_output": raw_label
    }
