"""Capture the real inputs and outputs of VLM-Pruner's token selection on LLaVA-1.5-7B.

For N POPE images runs the prefill once and, at the pruning layer, saves:
  - the exact intermediates the authors' code computes on the GPU before Stage 2:
      pivots (Stage 1) and the 576x576 cosine-similarity matrix over the top-256 channels
  - the image tokens' hidden states (input of Stage 3, needed to build new similarity variants)
  - the authors' result for B=1 and B=16: kept token indices + merged hidden states
and once per run: the 576x576 spatial-distance matrix exactly as the authors' code builds it.
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

EVAL = "playground/data/eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of distinct POPE images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--ratio", type=float, default=0.889)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--pivots", type=int, default=4)
    ap.add_argument("--batches", default="1,16", help="token batch sizes whose authors' result is saved")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    batches = [int(b) for b in args.batches.split(",")]

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model, None, get_model_name_from_path(args.model), device_map="cuda:0", attn_implementation="sdpa")
    model.eval()
    inner = model.get_model()
    print(f"GPUs visible: {torch.cuda.device_count()} | {torch.cuda.get_device_name(0)}")

    cfg = {"K": args.K, "image_token_start_index": 35, "image_token_length": 576,
           "reduction_ratio": args.ratio, "retain_token_num_for_llava_next": 320,
           "pivot_image_token": args.pivots, "threshold": args.threshold, "token_batch": 1}
    for c in (model.config, inner.config):
        c.VLMPruner_config = cfg
        c.text_length = None

    S, N = cfg["image_token_start_index"], cfg["image_token_length"]
    common = {"N": N, "R": round(N * (1 - args.ratio)), "width": 24, "norm_factor": 1.0 / (24 * 2 ** 0.5),
              "lam": 0.5, "threshold": args.threshold, "pivots": args.pivots, "ratio": args.ratio,
              "K": args.K, "q_channels": 256, "beta": 0.3, "batches": batches}

    original = inner.get_retained_image_token
    state = {"current": None, "spatial_saved": False}

    def capture(config, last_layer_state, any_states):
        item = state["current"]
        dev = last_layer_state.device
        with torch.no_grad():
            # same lines as the authors' get_retained_image_token (Stage 1 + similarity) 
            ks = any_states.permute(0, 2, 1, 3).reshape(any_states.shape[0], any_states.shape[2], -1)
            k_img = ks[0][S:S + N, :].detach()
            k_norm = torch.norm(k_img, p=1, dim=-1)
            piv = torch.zeros(args.pivots, dtype=torch.long, device=dev)
            piv[0] = k_norm.argmax()
            for i in range(1, args.pivots):
                d = torch.cdist(k_img, k_img[piv[:i]])
                piv[i] = d.min(dim=1).values.argmax()
            tokens_state = last_layer_state[0, torch.arange(N, device=dev) + S, :]
            var = tokens_state.var(dim=0)
            top = torch.topk(var, k=256).indices
            screen = tokens_state[:, top]
            sim_all = torch.nn.functional.cosine_similarity(screen.unsqueeze(1), screen.unsqueeze(0), dim=-1)
            if not state["spatial_saved"]:
                pos = inner.get_spatial_pos(torch.arange(N, device=dev))
                common["spatial"] = torch.cdist(pos, pos).cpu()
                state["spatial_saved"] = True

            # the authors' function for each batch size 
            refs, out_b1 = {}, None
            for B in batches:
                cfg_b = dict(config.VLMPruner_config, token_batch=B)
                t = time.perf_counter()
                idx, st = original(SimpleNamespace(VLMPruner_config=cfg_b), last_layer_state.clone(), any_states.clone())
                torch.cuda.synchronize()
                refs[B] = {"kept": (idx - S).cpu(), "merged_rows": st[0, idx, :].cpu(),
                           "gpu_ms": 1000 * (time.perf_counter() - t)}
                if B == 1:
                    out_b1 = (idx, st)

        torch.save({"question_id": item["question_id"], "image": item["image"],
                    "pivots": piv.cpu(), "sim": sim_all.cpu(), "top_channels": top.cpu(),
                    "hidden_img": tokens_state.cpu(), "refs": refs},
                   f"{args.out}/img_{state['count']:03d}.pt")
        state["count"] += 1
        return out_b1 if out_b1 is not None else original(config, last_layer_state, any_states)

    inner.get_retained_image_token = capture
    state["count"] = 0

    # distinct images, in file order
    items, seen_img = [], set()
    for l in open(f"{EVAL}/pope/llava_pope_test.jsonl"):
        q = json.loads(l)
        if q["image"] not in seen_img:
            seen_img.add(q["image"]); items.append(q)
        if len(items) == args.n:
            break

    t0 = time.time()
    for q in items:
        state["current"] = q
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + q["text"])
        conv.append_message(conv.roles[1], None)
        image = Image.open(f"{EVAL}/pope/val2014/{q['image']}").convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        with torch.inference_mode():
            model.generate(input_ids, images=image_tensor.to(dtype=torch.float16, device="cuda"),
                           image_sizes=[image.size], do_sample=False, num_beams=1, max_new_tokens=1, use_cache=True)

    torch.save(common, f"{args.out}/common.pt")
    print(f"captured {state['count']} images in {time.time() - t0:.0f}s -> {args.out}/ "
          f"({sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
