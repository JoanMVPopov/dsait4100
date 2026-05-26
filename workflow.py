import asyncio
import json
import pandas as pd
from pathlib import Path

from ollama import AsyncClient
from tqdm import tqdm

from classification import classify_query

PROJECT_ROOT = Path(__file__).resolve().parent

# HUMAN_DATA_PATH = PROJECT_ROOT / "data" / "hatexplain" / "hatexplain_sampled_30.csv"
# RESULTS_PATH = PROJECT_ROOT / "data" / "hatexplain" / "results_ollama_runs.jsonl"

# OUTPUT_PATH = Path("/workspace/results_ollama_runs.jsonl")
OUTPUT_PATH = PROJECT_ROOT / "results_ollama_runs.jsonl"

SEEDS = [42, 43, 44, 45, 46]

MODELS = [
    {"name": "qwen3.5:9b-q8_0", "think": False},
    {"name": "llama3.1:8b-instruct-q8_0", "think": False},
    {"name": "deepseek-r1:8b-0528-qwen3-q8_0", "think": False},
]

PERSONAS = {
    "no_persona": "",
    "neutral": """
You are a neutral content moderation evaluator.
You classify text based only on the wording provided, without assuming missing context.
You do not favor either stricter or more permissive interpretations of hate speech.
""".strip(),
    "conservative": """
You are a content moderation evaluator with conservative political views.
You prioritize individual liberty, freedom of expression, social order, and caution against over-censorship.
You classify speech as hate speech only when it clearly targets a protected group with hateful, threatening, or dehumanizing language.
""".strip(),
    "liberal": """
You are a content moderation evaluator with liberal political views.
You prioritize social justice, equality, anti-discrimination, and the protection of marginalized groups.
You are sensitive to language that may reinforce prejudice, exclusion, or systemic harm.
You may classify speech as hate speech when it targets or harms a protected group, even if the wording is indirect or context-dependent.
""".strip(),
}

# QUERIES = pd.read_csv("/workspace/hatexplain_sampled_wflow_ready_30.csv").to_dict(orient="records")
QUERIES_PATH = PROJECT_ROOT / "data" / "hatexplain" / "hatexplain_sampled_wflow_ready_30.csv"
QUERIES = pd.read_csv(QUERIES_PATH).to_dict(orient="records")

async def run_one_request(
    client,
    model_name,
    think,
    persona_name,
    persona_prompt,
    query,
    seed,
):
    result = await classify_query(
        client=client,
        model=model_name,
        query_id=query["query_id"],
        text=query["text"],
        seed=seed,
        think=think,
        persona_prompt=persona_prompt,
    )

    result["persona"] = persona_name
    return result


async def run_model_persona(client, model_name, think, persona_name, persona_prompt):
    tasks = []

    for query in QUERIES:
        for seed in SEEDS:
            tasks.append(
                run_one_request(
                    client=client,
                    model_name=model_name,
                    think=think,
                    persona_name=persona_name,
                    persona_prompt=persona_prompt,
                    query=query,
                    seed=seed,
                )
            )

    total = len(tasks)

    print(f"\nStarting model: {model_name}")
    print(f"Persona: {persona_name}")
    print(f"Total requests for this condition: {total}")

    completed = 0

    with OUTPUT_PATH.open("a", encoding="utf-8") as f:
        for finished_task in tqdm(
            asyncio.as_completed(tasks),
            total=total,
            desc=f"{model_name} | {persona_name}",
            unit="request",
        ):
            result = await finished_task
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            completed += 1

    print(f"Finished {model_name} | {persona_name}: {completed} requests")


async def main():
    client = AsyncClient(host="http://localhost:11434")

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    total_requests = (
        len(QUERIES)
        * len(SEEDS)
        * len(MODELS)
        * len(PERSONAS)
    )

    print("Starting experiment")
    print(f"Number of queries: {len(QUERIES)}")
    print(f"Number of seeds: {len(SEEDS)}")
    print(f"Number of models: {len(MODELS)}")
    print(f"Number of persona conditions: {len(PERSONAS)}")
    print(f"Total requests: {total_requests}")
    print(f"Output file: {OUTPUT_PATH}")

    for model_info in MODELS:
        for persona_name, persona_prompt in PERSONAS.items():
            await run_model_persona(
                client=client,
                model_name=model_info["name"],
                think=model_info["think"],
                persona_name=persona_name,
                persona_prompt=persona_prompt,
            )

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
