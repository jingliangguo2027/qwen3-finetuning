import json
from datetime import date, timedelta


def render_prompt(tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def memory_prompt(sessions, question_date, question):
    history = "\n\n".join(
        date + "\n" + "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
        for date, turns in sorted(sessions, key=lambda session: session[0])
    )
    return (
        "Answer the question using the dated conversation history. "
        "If there is insufficient evidence, say so. Give a concise answer.\n\n"
        f"{history}\n\nQuestion date: {question_date}\nQuestion: {question}"
    )


def synthetic_memory_example(tokenizer, person, month, old_code, new_code, *, distractor_rounds=1):
    start = date(2026, month, 1)
    dates = [
        f"{start + timedelta(days=day):%Y/%m/%d} 09:00"
        for day in range(19 + 14 * (distractor_rounds - 1))
    ]
    sessions = [
        (dates[0], [{"role": "user", "content": f"My locker code is {old_code}."}])
    ]
    topics = (
        "weekend bus routes", "indoor plants", "museum opening hours", "soup recipes",
        "rainy-day shoes", "local birds", "paper recycling", "running trails",
        "bike repairs", "used books", "garden herbs", "calendar reminders",
        "drawing supplies", "coffee filters",
    )
    for index, topic in enumerate(topics * distractor_rounds, start=1):
        content = (
            f"I am looking up {topic} this week. I have limited time because I work on "
            "Tuesday and Wednesday. Could you give me two practical ideas for getting "
            "started, and tell me which supplies I can borrow instead of buying? "
            "I want to make a short written checklist and share it with a friend on Friday."
        )
        sessions.append((dates[index], [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "Start with one small task, then review your notes before buying supplies. "
             "When you share the checklist, include the time you need, the items you can borrow, "
             "and anything your friend should confirm in advance."},
        ]))
    sessions.append((dates[-2], [
        {"role": "user", "content": f"Please replace my old locker code with {new_code}."}
    ]))
    question = f"What is the latest locker code for the user, {person}?"
    return {
        "id": f"synthetic-{person.lower()}",
        "prompt": render_prompt(tokenizer, memory_prompt(sessions, dates[-1], question)),
        "completion": new_code,
    }


def oracle_prompts(path, tokenizer, max_length, limit):
    rows = json.loads(path.read_text(encoding="utf-8"))
    selected = []
    for row in rows:
        sessions = zip(row["haystack_dates"], row["haystack_sessions"], strict=True)
        prompt = render_prompt(
            tokenizer, memory_prompt(sessions, row["question_date"], row["question"])
        )
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if token_count <= max_length:
            selected.append({"question_id": row["question_id"], "prompt": prompt,
                             "prompt_tokens": token_count})
            if len(selected) == limit:
                return selected
    if not selected:
        raise ValueError("No Oracle example fits the specified context budget.")
    return selected
