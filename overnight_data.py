import json
import random
from collections import defaultdict
from datetime import date, timedelta

from smoke_data import memory_prompt, render_prompt

LONG_LIMIT = 1920
TOPICS = (
    "walking routes", "library visits", "grocery shopping", "herb gardening",
    "museum tickets", "bicycle maintenance", "table lamps", "coffee filters",
    "reusable notebooks", "weather forecasts", "weekend buses", "public concerts",
    "paper recycling", "painting supplies",
)


def memory_cases(tokenizer):
    rng = random.Random(42)
    codes = rng.sample(range(1000, 9999), 608 * 2)
    for index in range(608):
        kind = ("recall", "update", "missing")[index % 3]
        subject = ("locker", "storage box", "equipment cabinet", "reservation")[index % 4]
        old, new = (f"K{code}" for code in codes[2 * index:2 * index + 2])
        statements = []
        for topic in rng.sample(TOPICS, len(TOPICS)):
            statements.append([
                {"role": "user", "content": (
                    f"I am looking up {topic} this week. I have limited time because I work on "
                    "Tuesday and Wednesday. Could you give me two practical ideas for getting started "
                    "and tell me which supplies I can borrow instead of buying? I want to make a short "
                    "written checklist and share it with a friend on Friday."
                )},
                {"role": "assistant", "content": (
                    "Start with one small task and review your notes before buying supplies. "
                    "When you share the checklist, include the time you need, items you can borrow, "
                    "and anything your friend should confirm in advance."
                )},
            ])
        if kind != "missing":
            statements.insert(rng.randrange(0, 7), [{"role": "user", "content": f"My {subject} code is {old}."}])
        if kind == "update":
            statements.insert(rng.randrange(8, len(statements) + 1), [
                {"role": "user", "content": f"Update: my {subject} code changed to {new}."}
            ])
        first = date(2025, 1, 1) + timedelta(days=index)
        dates = [(first + timedelta(days=day)).isoformat() for day in range(len(statements) + 1)]
        question = f"What is my latest {subject} code? Output only the code, or UNKNOWN if it is absent."
        yield {
            "id": f"synthetic-{index:03d}",
            "task_type": kind,
            "question": question,
            "completion": new if kind == "update" else old if kind == "recall" else "UNKNOWN",
            "prompt": render_prompt(tokenizer, memory_prompt(zip(dates, statements), dates[-1], question)),
        }


def fixed_oracle_cases(tokenizer, path, count):
    raw = json.loads(path.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    if len(raw) != 500 or len({row["question_id"] for row in raw}) != 500:
        raise ValueError("Expected the 500 unique questions in LongMemEval Oracle.")
    for row in sorted(raw, key=lambda item: item["question_id"]):
        kind = "abstention" if row["question_id"].endswith("_abs") else row["question_type"]
        text = memory_prompt(zip(row["haystack_dates"], row["haystack_sessions"], strict=True),
                             row["question_date"], row["question"])
        prompt = render_prompt(tokenizer, text)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) + 64 <= 4096:
            groups[kind].append({"id": row["question_id"], "task_type": kind, "input_ids": ids})
    selected = []
    for index in range(max(map(len, groups.values()), default=0)):
        for kind in sorted(groups):
            if index < len(groups[kind]):
                selected.append(groups[kind][index])
                if len(selected) == count:
                    return selected
    raise ValueError(f"Only {len(selected)} Oracle prompts fit the 4096-token inference limit.")
