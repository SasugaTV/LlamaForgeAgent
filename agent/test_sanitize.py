def _sanitize_messages(msgs):
    out = []
    for m in msgs:
        if not m.get("content"):
            continue
        if out and out[-1]["role"] == m["role"]:
            # If we need to merge content
            if isinstance(out[-1]["content"], list) or isinstance(m["content"], list):
                # Combine lists/strings into a new list
                c1 = out[-1]["content"] if isinstance(out[-1]["content"], list) else [{"type": "text", "text": out[-1]["content"]}]
                c2 = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
                out[-1] = {"role": m["role"], "content": c1 + [{"type": "text", "text": "\n\n"}] + c2}
            else:
                out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append(dict(m))
    first_user = next((i for i, m in enumerate(out) if m["role"] == "user"), None)
    if first_user is None:
        return out
    return [m for i, m in enumerate(out) if m["role"] == "system" or i >= first_user]

query_context = [
    {"role": "system", "content": "You are a bot"},
    {"role": "system", "content": "Relevant past memories:\nHi\nBye"},
    {"role": "user", "content": "Hello"}
]
print("Sanitized:", _sanitize_messages(query_context))
