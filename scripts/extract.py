"""Extract real hidden states (all layers) from a small open LLM for a corpus of sentences.

Writes data/hidden.npz  -> hidden: [L, N, D] float32 mean-pooled per layer,
                           plus texts, topics, token counts, model name.
Also writes data/traj.npz -> a real generation trajectory: hidden state of the
                           final layer after each generated token.
"""
import json, os, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("LSV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROMPT = "Write a short sentence about the ocean at night."

def mean_pool(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)

def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    D = model.config.hidden_size
    L = model.config.num_hidden_layers + 1
    print(f"model={MODEL} layers={L} hidden={D} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    corpus = json.load(open(os.path.join(ROOT, "data", "corpus.json")))
    texts, topics = [], []
    for topic, sents in corpus.items():
        for s in sents:
            texts.append(s); topics.append(topic)
    N = len(texts)
    print(f"sentences={N} topics={len(corpus)}")

    hidden = np.zeros((L, N, D), dtype=np.float32)
    ntok = np.zeros(N, dtype=np.int32)
    B = 8
    with torch.no_grad():
        for i in range(0, N, B):
            batch = texts[i:i+B]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
            out = model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"]
            for l, h in enumerate(out.hidden_states):
                pooled = mean_pool(h, mask)
                hidden[l, i:i+len(batch)] = pooled.float().numpy()
            ntok[i:i+len(batch)] = mask.sum(1).numpy()
            print(f"  {i+len(batch)}/{N}", flush=True)

    np.savez_compressed(os.path.join(ROOT, "data", "hidden.npz"),
                        hidden=hidden, ntok=ntok, model=np.array([MODEL]))

    # Real generation trajectory: hidden state of the last layer after each token.
    msgs = [{"role": "user", "content": PROMPT}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    gen_ids, step_hidden = [], []
    cur = ids
    with torch.no_grad():
        for _ in range(48):
            out = model(cur, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1].float().numpy()
            step_hidden.append(h)
            nxt = int(torch.argmax(out.logits[0, -1]))
            gen_ids.append(nxt)
            if nxt == tok.eos_token_id:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    text = tok.decode(gen_ids, skip_special_tokens=True)
    np.savez_compressed(os.path.join(ROOT, "data", "traj.npz"),
                        steps=np.array(step_hidden, dtype=np.float32),
                        prompt=np.array([PROMPT]),
                        text=np.array([text]))
    print("trajectory tokens:", len(gen_ids))
    print("generated:", text[:200])
    json.dump({"texts": texts, "topics": topics}, open(os.path.join(ROOT, "data", "meta.json"), "w"))
    print("saved data/hidden.npz and data/traj.npz")

if __name__ == "__main__":
    main()
