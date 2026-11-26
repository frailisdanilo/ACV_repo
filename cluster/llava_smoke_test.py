"""LLaVA-1.5-7B smoke test for VLM-Pruner.

Runs a few POPE questions three times: unpruned, VLM-Pruner B=16, VLM-Pruner B=1.
Inputs are built exactly like llava/eval/model_vqa_loader.py (vicuna_v1 template,
process_images, tokenizer_image_token, float16 images, greedy decoding).

Per question it prints:
  - position of the image token in the prompt (must be 35: the scripts hard-code it)
  - sequence length entering layer 0 -> entering layer K (pruning happens just before layer K)
  - Stage-2 selection time (get_retained_image_token, GPU-synchronised), total time
  - prediction vs ground truth

Run from the repo root (so ./playground/data/eval resolves), on a GPU node:
    python cluster/llava_smoke_test.py
    python cluster/llava_smoke_test.py --n 10
"""
import argparse
import json
import os
import time

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

POPE = "playground/data/eval/pope"


def load_pope(n):
    """First n questions of llava_pope_test.jsonl, with ground truth from coco/coco_pope_<category>.json."""
    questions = [json.loads(l) for l in open(f"{POPE}/llava_pope_test.jsonl")][:n]
    labels = {}  # (category, image) -> list of (question text, label)
    for cat in {q["category"] for q in questions}:
        for l in open(f"{POPE}/coco/coco_pope_{cat}.json"):
            a = json.loads(l)
            labels.setdefault((cat, a["image"]), []).append((a["text"].strip(), a["label"]))
    for q in questions:
        gt = "?"
        for text, label in labels.get((q["category"], q["image"]), []):
            if q["text"].strip().startswith(text):  # LLaVA's text may append an answer-format instruction
                gt = label
                break
        q["gt"] = gt
    return questions


def to_yes_no(text):
    """Same rule as llava/eval/eval_pope.py."""
    text = text.split(".")[0].replace(",", "")
    words = text.split(" ")
    return "no" if ("No" in words or "not" in words or "no" in words) else "yes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--n", type=int, default=5, help="number of POPE questions")
    ap.add_argument("--ratio", type=float, default=0.889, help="reduction ratio (paper: 0.889 -> 64 tokens)")
    ap.add_argument("--K", type=int, default=2, help="pruned layer")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--pivots", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    import transformers
    print(f"torch {torch.__version__} | transformers {transformers.__version__}")
    assert torch.cuda.is_available(), "No GPU visible: are you on a GPU node?"

    disable_torch_init()
    t0 = time.time()
    model_name = get_model_name_from_path(args.model)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model, None, model_name, attn_implementation="sdpa")
    model.eval()
    inner = model.get_model()  # LlavaLlamaModel = LlavaMetaModel + VLMPruner
    print(f"model {model_name} loaded in {time.time() - t0:.0f}s, GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # read-only pre-hooks: sequence length entering layer 0 and layer K (prefill only)
    seen = {}
    def hook(name):
        def fn(_module, args_, kwargs_):
            hs = args_[0] if args_ else kwargs_["hidden_states"]
            if hs.shape[1] > 1:
                seen[name] = hs.shape[1]
        return fn
    inner.layers[0].register_forward_pre_hook(hook("in"), with_kwargs=True)
    inner.layers[args.K].register_forward_pre_hook(hook("after"), with_kwargs=True)

    # time Stage 1-3 selection by wrapping the bound method on this one object (files unchanged)
    sel_times = []
    original_select = inner.get_retained_image_token
    def timed_select(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = original_select(*a, **k)
        torch.cuda.synchronize(); sel_times.append(time.perf_counter() - t)
        return out
    inner.get_retained_image_token = timed_select

    questions = load_pope(args.n)
    configs = [("full", None), ("B16", 16), ("B1", 1)]
    summary = {}

    for name, token_batch in configs:
        cfg = None
        if token_batch is not None:
            cfg = {"K": args.K, "image_token_start_index": 35, "image_token_length": 576,
                   "reduction_ratio": args.ratio, "retain_token_num_for_llava_next": 320,
                   "pivot_image_token": args.pivots, "threshold": args.threshold,
                   "token_batch": token_batch}
        # set on both config objects (normally the same object; this makes sure)
        for c in (model.config, inner.config):
            c.VLMPruner_config = cfg
            c.text_length = None

        print(f"\n[{name}]")
        correct, totals, sels = 0, [], []
        for q in questions:
            prompt_text = DEFAULT_IMAGE_TOKEN + "\n" + q["text"]
            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image = Image.open(os.path.join(POPE, "val2014", q["image"])).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            img_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().flatten().tolist()

            seen.clear(); sel_times.clear()
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(input_ids, images=image_tensor.to(dtype=torch.float16, device="cuda"),
                                     image_sizes=[image.size], do_sample=False, num_beams=1,
                                     max_new_tokens=args.max_new_tokens, use_cache=True)
            torch.cuda.synchronize(); total = time.perf_counter() - t

            answer = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
            pred = to_yes_no(answer)
            ok = pred == q["gt"]
            correct += ok
            totals.append(total)
            sel = sum(sel_times)
            sels.append(sel)
            print(f"  img_pos {img_pos} | seq {seen.get('in')} -> {seen.get('after')} "
                  f"| select {sel * 1000:6.1f} ms | total {total * 1000:6.0f} ms "
                  f"| gt {q['gt']:3s} | pred {pred:3s} ({answer[:30]!r}) {'OK' if ok else 'WRONG'}")
        summary[name] = (correct, sum(totals) / len(totals), sum(sels) / len(sels))

    print("\n=== summary (first question includes GPU warm-up) ===")
    for name, (c, t, s) in summary.items():
        print(f"{name:5s}: {c}/{len(questions)} correct | {t * 1000:6.0f} ms total/question | {s * 1000:6.1f} ms selection/question")
    print(f"peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
