"""
Full LLaVA-1.5-7B evaluation on POPE and/or OCRBench, for one pruning configuration

Reproduces the authors' evaluation (llava/eval/model_vqa_loader.py + eval_pope.py for POPE,
llava/eval/model_vqa_ocrbench.py for OCRBench): same prompts, same generation settings, same
scoring. The differences are:
  - "full" (unpruned) uses a config with K=-1 instead of None.
  - model on ONE GPU (device_map="cuda:0"); the job script picks which GPU that is.
  - per-question records: prediction, ground truth, correctness, generate time, selection
    time, sequence length before/after pruning (for paired comparisons and timing).
  - one unscored warm-up question before timing starts.

"""

import argparse
import json
import os
import statistics
import time

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


EVAL = "playground/data/eval"
CONV = "vicuna_v1"

METHODS = {"full":None, "B16":16, "B1":1}


#DATA
def load_pope(limit):
    questions = [json.loads(l) for l in open(f"{EVAL}/pope/llava_pope_test.jsonl")]
    labels = {}

    for cat in sorted({q["category"] for q in questions}):
        path = f"{EVAL}/pope/coco/coco_pope_{cat}.json"
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; {EVAL}/pope/coco contains: {sorted(os.listdir(f'{EVAL}/pope/coco'))}")
        labels[cat] = [json.loads(l)["label"] for l in open(path)]
    
    seen_per_cat = {}
    for q in questions:
        k = seen_per_cat.get(q["category"], 0)
        q["gt"] = labels[q["category"]][k]
        seen_per_cat[q["category"]] = k+1

    for cat, n in seen_per_cat.items():
        assert n == len(labels[cat]), f"POPE {cat}: {n} questions but {len(labels[cat])} labels"

    items = [{"id": q["question_id"], "image": f"{EVAL}/pope/val2014/{q['image']}", "question":q["text"], "gt":q["gt"], "group":q["category"]} for q in questions]
    return items[:limit] if limit else items

def load_ocrbench(limit):
    data = json.load(open(f"{EVAL}/OCRBench/OCRBench/OCRBench.json"))
    items = [{"id": i, "image": f"{EVAL}/OCRBench/OCRBench_Images/{d['image_path']}",
              "question": d["question"] + "\nAnswer the question using a single word or phrase.",
              "gt": d["answers"], "group": d["type"], "dataset_name": d["dataset_name"]}
             for i, d in enumerate(data)]
    return items[:limit] if limit else items

#SCORING

def pope_yes_no(text):
    #no if in the first sentence No/not/no appear
    if text.find(".") != -1:
        text = text.split(".")[0]
    words = text.replace(",", "").split(" ")
    return "no" if ("No" in words or "not" in words or "no" in words) else "yes"

def pope_metrics(records):
    tp = sum(r["pred"]=="yes" and r["gt"]=="yes" for r in records)
    fp = sum(r["pred"]=="yes" and r["gt"]=="no" for r in records)
    tn = sum(r["pred"]=="no" and r["gt"]=="no" for r in records)
    fn = sum(r["pred"]=="no" and r["gt"]=="yes" for r in records)

    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall / (precision+recall) if precision+recall else 0.0
    return {"n": len(records), 
            "f1": 100 * f1, "acc": 100 * (tp + tn) / len(records),
            "precision": 100 * precision, "recall": 100 * recall,
            "yes_ratio": 100 * (tp + fp) / len(records)}

def ocrbench_correct(item, predict):
    answers = item["gt"] if isinstance(item["gt"], list) else [item["gt"]]

    if item["dataset_name"] == "HME100k":
        norm = lambda s: s.strip().replace("\n", " ").replace(" ", "")
    else:
        norm = lambda s: s.lower().strip().replace("\n", " ")

    return any(norm(a) in norm(predict) for a in answers)


#MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--bench", default="pope,ocrbench", help="comma-separated: pope, ocrbench")
    ap.add_argument("--out", required=True, help="output folder (one per run)")
    ap.add_argument("--model", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions per benchmark (0 = all)")
    ap.add_argument("--ratio", type=float, default=0.889)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--pivots", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)  # authors' default for both benchmarks
    ap.add_argument("--progress_every", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import transformers
    assert torch.cuda.is_available(), "No GPU visible: are you on a GPU node?"

    env = {"torch": torch.__version__, "transformers": transformers.__version__,
           "gpu": torch.cuda.get_device_name(0), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "gpus_visible": torch.cuda.device_count(), "commit": os.environ.get("GIT_COMMIT"),
           "job": os.environ.get("PBS_JOBID")}
    print(json.dumps(env))


    disable_torch_init()
    t0 = time.time()
    tokenizer, model, image_processor, _ = load_pretrained_model( args.model, None, get_model_name_from_path(args.model), device_map="cuda:0", attn_implementation="sdpa")

    model.eval()
    inner = model.get_model()
    print(f"model loaded in {time.time() - t0:.0f}s")

    #pruning config, full --> K never reached

    token_batch = METHODS[args.method]
    cfg = {"K":args.K if token_batch is not None else -1,
           "image_token_start_index":35,
           "image_token_length": 576,
           "reduction_ratio": args.ratio,
           "retain_token_num_for_llava_next": 320,
           "pivot_image_token": args.pivots,
           "threshold": args.threshold,
           "token_batch": token_batch}

    for c in (model.config, inner.config):
        c.VLMPruner_config = cfg
        c.text_length = None

    run_info = {"method": args.method, "config":cfg, "args":vars(args), "env":env}
    json.dump(run_info, open(f"{args.out}/run_info.json", "w"), indent=2)

    #read only hooks, timing wrapper

    seen = {}
    def hook(name):
        def fn(_m, a, k):
            hs = a[0] if a else k["hidden_states"]
            if hs.shape[1] > 1:
                seen[name] = hs.shape[1]
        return fn

    inner.layers[0].register_forward_pre_hook(hook("in"), with_kwargs=True)
    inner.layers[args.K].register_forward_pre_hook(hook("after"), with_kwargs=True)

    sel_times = []
    original_select = inner.get_retained_image_token

    def timed_select(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = original_select(*a, **k)
        torch.cuda.synchronize(); sel_times.append(time.perf_counter() -t)
        return out

    inner.get_retained_image_token = timed_select

    conv0 = conv_templates[CONV]
    stop_str = conv0.sep if conv0.sep_style != SeparatorStyle.TWO else conv0.sep2

    def answer(item):
        conv = conv_templates[CONV].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + item["question"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image = Image.open(item["image"]).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        seen.clear(); sel_times.clear()
        torch.cuda.synchronize(); t = time.perf_counter()

        with torch.inference_mode():
            out = model.generate(input_ids, 
                                 images=image_tensor.to(dtype=torch.float16, device="cuda"),
                                 image_sizes=[image.size],
                                 do_sample=False,
                                 temperature=0,
                                 top_p=None,
                                 num_beams=1,
                                 max_new_tokens=args.max_new_tokens,
                                 use_cache=True)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        if text.endswith(stop_str):
            text = text[:-len(stop_str)].strip()
        return text, dt 

    summary = {"method":args.method, "env":env}

    for bench in [b.strip() for b in args.bench.split(",") if b.strip()]:
        items = load_pope(args.limit) if bench == "pope" else load_ocrbench(args.limit)
        print(f"\n[{bench}] {len(items)} questions, method {args.method}")
        answer(items[0]) #warmup

        records = []
        with open(f"{args.out}/{bench}_answers.jsonl", "w") as f:
            t_bench = time.time()
            for i, item in enumerate(items):
                text, dt = answer(item)
                if bench == "pope":
                    pred = pope_yes_no(text)
                    correct = pred==item["gt"]
                else:
                    pred = text
                    correct = ocrbench_correct(item, text)
                rec = {"id":item["id"],
                       "group":item["group"],
                       "gt":item["gt"],
                       "raw":text,
                       "pred":pred,
                       "correct":bool(correct),
                       "total_ms":1000*dt,
                       "select_ms":1000*sum(sel_times),
                       "seq_in":seen.get("in"),
                       "seq_after":seen.get("after")}
                records.append(rec)
                f.write(json.dumps(rec) + "\n")

                if (i+1) % args.progress_every == 0 or i+1 == len(items):
                    f.flush()
                    print(f"  {i + 1}/{len(items)} | {time.time() - t_bench:.0f}s | "
                          f"running acc {100 * sum(r['correct'] for r in records) / len(records):.1f}%")

        # authors' answer format, so llava/eval/eval_pope.py can cross-check the score
        if bench == "pope":
            with open(f"{args.out}/pope_llava_format.jsonl", "w") as f:
                for r in records:
                    f.write(json.dumps({"question_id":r["id"], "text":r["raw"]}) + "\n")

        totals = [r["total_ms"] for r in records]
        sels = [r["select_ms"] for r in records]
        res = {"n": len(records),
               "time_ms_median": statistics.median(totals), "time_ms_mean": statistics.mean(totals),
               "select_ms_median": statistics.median(sels), "select_ms_mean": statistics.mean(sels),
               "seq_in_median": statistics.median(r["seq_in"] for r in records),
               "seq_after_median": statistics.median(r["seq_after"] for r in records)}

        if bench == "pope":
            per_cat = {c: pope_metrics([r for r in records if r["group"] == c]) for c in sorted({r["group"] for r in records})}
            res["per_category"] = per_cat
            res["f1_mean"] = statistics.mean(m["f1"] for m in per_cat.values())
            res["acc_mean"] = statistics.mean(m["acc"] for m in per_cat.values())

            print(f"POPE F1 (mean of categories) {res['f1_mean']:.2f} | acc {res['acc_mean']:.2f}")

            for c, m in per_cat.items():
                print(f"  {c:12s} F1 {m['f1']:.2f} acc {m['acc']:.2f} yes-ratio {m['yes_ratio']:.1f} (n={m['n']})")
        else:
            per_type={}
            for r in records:
                per_type[r["group"]] = per_type.get(r["group"], 0) + r["correct"]
            res["per_type"] = per_type
            res["score"] = sum(r["correct"] for r in records)
            print(f"OCRBench score {res['score']} / {len(records)}")
            for t, s in sorted(per_type.items()):
                print(f"  {t}: {s}")

        print(f"time per question: median {res['time_ms_median']:.0f} ms | selection median {res['select_ms_median']:.1f} ms"
              f" | seq {res['seq_in_median']:.0f} -> {res['seq_after_median']:.0f}")
        summary[bench] = res

    summary["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=2)
    print(f"\nsaved to {args.out}/")


if __name__ == "__main__":
    main()