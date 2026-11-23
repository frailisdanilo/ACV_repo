"""
Smoke test for the VLM-Pruner Qwen2-VL fork

Runs a few POPE samples through Qwen2-VL-7B-Instruct twice: once with pruning
off (VLMPruner_config = None) and once with pruning on, and reports for each run
  - how many tokens enter the LLM and how many survive after the pruning layer
  - the model's answer vs. the ground truth
  - prefill+generate time

Usage (from the repo root, on a GPU node):
    python cluster/smoke_test.py               # 5 samples, paper settings
    python cluster/smoke_test.py --n 10 --ratio 0.667
"""

import argparse
import glob
import importlib
import io
import os
import sys
import time

import torch
from PIL import Image


# repo root (the folder above cluster/) on the import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# expects <repo>/qwen2_vl/modeling_qwen2_vl_self.py (+ configuration_qwen2_vl.py next to it)
from qwen2_vl.modeling_qwen2_vl_self import Qwen2VLForConditionalGeneration

def load_pope(n):
    from huggingface_hub import snapshot_download

    root = snapshot_download("lmms-lab/POPE", repo_type="dataset")
    files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))

    if not files:
        sys.exit(f"No parquet files under {root}. Did `hf download lmms-lab/POPE --repo-type dataset` finish?")

    import pyarrow.parquet as pq
    
    rows = next(pq.ParquetFile(files[0]).iter_batches(batch_size=n)).to_pylist()
    samples = []

    for r in rows:
        img = r["image"]
        img = Image.open(io.BytesIO(img["bytes"])) if isinstance(img, dict) else img
        samples.append((img.convert("RGB"), r["question"], str(r["answer"]).strip().lower()))

    return samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--n", type=int, default=5, help="number of POPE samples")
    ap.add_argument("--pixels", type=int, default=1280*28*28, help="min_pixels=max_pixels")
    ap.add_argument("--ratio", type=float, default=1 - 150/1350, help="reduction ratio (paper is 88-9%)")
    ap.add_argument("--K", type=int, default=2, help="layer at which pruning happens")
    ap.add_argument("--pivots", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--token_batch", type=int, default=16)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "aeger", "flash_attention_2"])
    args = ap.parse_args()

    import transformers
    from transformers import AutoProcessor

    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | transformers {transformers.__version__}")
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")


    t0 = time.time()

    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, 
                                                            torch_dtype=torch.bfloat16, 
                                                            attn_implementation=args.attn).to("cuda").eval()

    processor = AutoProcessor.from_pretrained(args.model,
                                              min_pixels=args.pixels,
                                              max_pixels=args.pixels )
    print(f"loaded in {time.time() - t0:.0f}s, GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    #record sequence length entering layer 0 and the layer after pruning (prefill only)

    seen = {}
    layers = model.model.layers

    def hook(name):
        def fn(_module, args_, kwargs_):
            hs = args_[0] if args_ else kwargs_["hidden_states"]
            if hs.shape[1]>1:
                seen[name] = hs.shape[1]
        return fn

    layers[0].register_forward_pre_hook(hook("in"), with_kwargs=True)
    layers[args.K + 1].register_forward_pre_hook(hook("after"), with_kwargs=True)

    merge = model.config.vision_config.spatial_merge_size
    image_token_id = model.config.image_token_id

    samples = load_pope(args.n)


    results = {}
    for mode in ("full", "pruned"):
        correct, times, lines = 0, [], []

        for img, question, gt in samples:
            messages = [{"role": "user", "content": [{"type":"image"}, 
                                                     {"type": "text", "text": question + " Answer yes or no."}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda")

            ids = inputs.input_ids[0]
            pos = (ids == image_token_id).nonzero().flatten()

            _, h, w = inputs.image_grid_thw[0].tolist()

            height, width = h//merge, w//merge  
            assert height*width == len(pos)

            cfg = None
            if mode == "pruned":
                cfg = dict(K=args.K, 
                           image_token_start_index=pos[0].item(),
                           image_token_length=len(pos),
                           reduction_ratio=args.ratio,
                           pivot_image_token=args.pivots,
                           height=height,
                           width=width,
                           threshold=args.threshold,
                           token_batch=args.token_batch)
                
            model.config.VLMPruner_config = cfg
            model.model.config.VLMPruner_config = cfg

            seen.clear()

            torch.cuda.synchronize()
            t = time.time()

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=5, do_sample=False)

            torch.cuda.synchronize()
            times.append(time.time() - t)

            answer = processor.batch_decode(out[:, ids.shape[0]:], skip_special_tokens=True)[0].strip().lower()
            ok = answer.startswith(gt)
            correct += ok

            lines.append(f"  img tokens {len(pos):5d} | LLM seq {seen.get('in')} -> {seen.get('after')} "
                             f"| gt {gt:3s} | pred {answer!r} {'OK' if ok else 'WRONG'}")

        results[model] = (correct, sum(times)/len(times))
        print(f"\n[{model}]")
        print("\n".join(lines))

    print("\n=== summary ===")
    for mode, (c, t) in results.items():
        print(f"{mode:6s}: {c}/{len(samples)} correct, {t:.2f}s per sample")
    print(f"peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

if __name__ == "__main__":
    main()