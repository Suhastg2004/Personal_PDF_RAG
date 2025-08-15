# prompt.py
def build_augmented_prompt(query: str, context_blocks):
    # context_blocks: list of tuples (label, text, source)
    header = (
        "You are a helpful assistant answering strictly from the provided documents.\n"
        "- Only use the retrieved documents; if unsure, say you don't know.\n"
        "- Cite sources at the end of each sentence using [DOC i].\n"
        "- Do not include information not present in the documents.\n"
        "- Be concise and accurate.\n\n"
        "Retrieved Documents:\n"
    )
    ctx = []
    for label, text, source in context_blocks:
        ctx.append(f"[{label}] Source: {source}\n{text}\n")
    ctx_joined = "\n".join(ctx)
    user = f"\nUser Question:\n{query}\n\nAnswer:"
    return header + ctx_joined + user
