"""
eval_multi.py -- cham NHIEU cau hinh nen trong 1 lan doc video: giai ma video + vision encoder 1 lan / video,
moi cau hinh chi them phan chon token + LLM. Cung harness voi infer_rap_videomme.py (128 frame,
max_pixels_units=16384, time_mode=lmms, 16 token sinh, greedy).

--variant TEN=CAU_HINH (lap lai), CAU_HINH:
  dense                  : 100% token
  uniform@R              : luoi linspace (= select_uniform cua bo moc)
  rule:center|var|norm|medoid@R : HDS Pha 0, 1 token / hop 3D theo luat (hds_model.rule_select)
  ckpt:DUONG_DAN@R       : checkpoint RAP cu (AnchoredResampler) hoac HDSPolicy
  holitom@R              : HoliTom w/o M (baselines/holitom_port.py)
  vidcom2@R | fastvid@R | dycoke@R : VidCom^2 / FastVID / DyCoke (chi TTM), cung ban port voi run_inference_qwen25vl_kitoke.py
Moi cau hinh ghi <output_dir>/<TEN>.json (+ .stats.jsonl), tu bo qua cau da co (resume).

Usage (tu thu muc KTV):
  python -m rap.eval_multi --gt_file playground/gt_qa_files/VideoMME/val_qa.json --output_dir outputs/hds \\
      --variant P0_center_r010=rule:center@0.10 --variant P0_var_r010=rule:var@0.10
"""

import argparse
import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common import run_inference_qwen25vl_dyseg as dm
from common.local_data import local_mirror
from common.qwen_vision_process_fastvid import process_vision_info
from common.run_inference_qwen25vl_2pass import build_pruned_sequence
from rap.hds_model import HDSPolicy, load_policy, rule_select
from rap.rap_model import uniform_indices


def parse_variant(s):
    name, spec = s.split("=", 1)
    ret = None
    if "@" in spec:
        spec, r = spec.rsplit("@", 1)
        ret = float(r)
    kind, _, arg = spec.partition(":")
    return {"name": name, "kind": kind, "arg": arg, "ret": ret}


def norm_sample(s):
    """Chuan hoa file cau hoi cac bo khac nhau ve dang Video-MME (candidates la list, answer_number la int)."""
    c = s["candidates"]
    if isinstance(c, str):
        c = ast.literal_eval(c)
    s["candidates"] = list(c)
    s["answer_number"] = int(s["answer_number"])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gt_file", default="playground/gt_qa_files/VideoMME/val_qa.json")
    ap.add_argument("--video_dir", default="Datasets/Video-MME/data/data")
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_frames", type=int, default=128)
    ap.add_argument("--max_pixels_units", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=2)
    args = ap.parse_args()
    args.video_dir = local_mirror(args.video_dir)
    args.model_path = local_mirror(args.model_path)
    device = "cuda"
    max_pixels = args.max_pixels_units * 28 * 28
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=max_pixels, min_pixels=dm.MIN_PIXELS)
    llm, vid = model.model, model.config.video_token_id

    variants = [parse_variant(v) for v in args.variant]
    for v in variants:
        if v["kind"] == "ckpt":
            v["policy"], ck = load_policy(v["arg"], device)
            v["policy"].eval()
            print(f"[eval_multi] {v['name']}: {v['arg']} step={ck.get('step')} config={ck['config']}", flush=True)
    need_attn = any(v["kind"] == "holitom" for v in variants)
    if need_attn:
        from baselines import holitom_port as ht
    kinds = {v["kind"] for v in variants}
    if "vidcom2" in kinds:
        from baselines import vidcom2_port as vc
    if "dycoke" in kinds:
        from baselines import dycoke_port as dk
    if "fastvid" in kinds:
        from baselines import fastvid_port as fv
    os.makedirs(args.output_dir, exist_ok=True)
    for v in variants:
        v["out"] = os.path.join(args.output_dir, f"{v['name']}.json")
        v["seen"] = dm.load_answered_ids(v["out"])
        v["fo"] = open(v["out"], "a", encoding="utf-8")
        v["fs"] = open(os.path.join(args.output_dir, f"{v['name']}.stats.jsonl"), "a", encoding="utf-8")

    gt = [norm_sample(s) for s in json.load(open(args.gt_file))]
    if args.limit > 0:
        gt = gt[: args.limit]
    gt.sort(key=lambda s: s["video_name"])   # gom cau cung video -> vision encoder 1 lan / video
    aid_of = lambda s: str(s.get("id", s.get("question_id")))  # noqa: E731
    todo_q = [s for s in gt if any(aid_of(s) not in v["seen"] for v in variants)]
    vids = []
    for s in todo_q:
        vp = os.path.join(args.video_dir, s["video_name"])
        if os.path.exists(vp) and (not vids or vids[-1] != vp):
            vids.append(vp)
    print(f"[eval_multi] {len(todo_q)} cau con lai / {len(gt)}, {len(vids)} video, {len(variants)} cau hinh", flush=True)

    def decode(vp):
        msgs = dm.build_messages(vp, "", [], max_pixels)
        msgs[1]["content"][0]["max_frames"] = args.max_frames
        return process_vision_info(msgs, return_video_kwargs=True)[1]
    pool = ThreadPoolExecutor(max(1, args.prefetch)) if args.prefetch > 0 else None
    pending, nxt = {}, 0

    def fill(cur):
        nonlocal nxt
        while pool is not None and nxt < len(vids) and len(pending) < args.prefetch:
            if vids[nxt] != cur and vids[nxt] not in pending:
                pending[vids[nxt]] = pool.submit(decode, vids[nxt])
            nxt += 1

    cache = {}
    for sample in tqdm(todo_q):
        aid = aid_of(sample)
        vp = os.path.join(args.video_dir, sample["video_name"])
        if not os.path.exists(vp):
            continue
        try:
            messages = dm.build_messages(vp, sample["question"], sample["candidates"], max_pixels)
            messages[1]["content"][0]["max_frames"] = args.max_frames
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if cache.get("vp") != vp:
                cache.clear()
                fut = pending.pop(vp, None)
                fill(vp)
                video_inputs = fut.result() if fut is not None else process_vision_info(messages, return_video_kwargs=True)[1]
                cache["vp"], cache["video_inputs"] = vp, video_inputs
            inputs = processor(text=[text], images=None, videos=cache["video_inputs"], padding=True, return_tensors="pt").to(device)
            inputs["second_per_grid_ts"] = torch.full_like(inputs["second_per_grid_ts"], 1.0)
            with torch.inference_mode():
                full_embeds = model.get_input_embeddings()(inputs["input_ids"])
                thw = inputs["video_grid_thw"][0]
                T, H, W = int(thw[0]), int(thw[1]) // 2, int(thw[2]) // 2
                N = H * W
                if "kept" not in cache:
                    if need_attn:
                        feat, attn = ht.holitom_frame_attn(llm, inputs["pixel_values_videos"], inputs["video_grid_thw"])
                    else:
                        feat = llm.get_video_features(inputs["pixel_values_videos"], video_grid_thw=inputs["video_grid_thw"]).pooler_output
                        attn = None
                    if isinstance(feat, (list, tuple)):
                        feat = torch.cat(feat, dim=0)
                    feat = feat.to(full_embeds.dtype).to(full_embeds.device)
                    assert feat.shape[0] == T * N
                    X = feat.float().view(T, N, -1)
                    cache["kept"] = {}
                    for v in variants:
                        st = {}
                        if v["kind"] == "dense":
                            ki, ke = torch.arange(T * N, device=device), feat
                        elif v["kind"] == "uniform":
                            ki = uniform_indices(T * N, v["ret"], device)
                            ke = feat[ki]
                        elif v["kind"] == "rule":
                            ki, st = rule_select(X, H, W, v["ret"], v["arg"])
                            ke = feat[ki]
                        elif v["kind"] == "holitom":
                            ki, ke, _ = ht.holitom_select(feat, attn.to(feat.device), inputs["video_grid_thw"],
                                                          ht.holitom_params(SimpleNamespace(retention=v["ret"])))
                        elif v["kind"] == "vidcom2":
                            ki, ke, _ = vc.vidcom2_select(feat, inputs["video_grid_thw"], v["ret"])
                        elif v["kind"] == "dycoke":
                            ki, ke, _ = dk.dycoke_ttm_select(feat, inputs["video_grid_thw"], dk.k_for_retention(v["ret"]))
                        elif v["kind"] == "fastvid":
                            if "fv" not in cache:   # FastVID can attention rieng cua block cuoi vision tower
                                fvf, fva = fv.fastvid_frame_attn(llm, inputs["pixel_values_videos"], inputs["video_grid_thw"])
                                cache["fv"] = (fvf.to(feat.dtype).to(feat.device), fva)
                            torch.manual_seed(0)   # ban goc cong torch.rand*1e-6 vao density
                            ki, ke, _ = fv.fastvid_select(cache["fv"][0], cache["fv"][1], inputs["video_grid_thw"],
                                                          fv.fastvid_params(SimpleNamespace(retention=v["ret"])))
                        elif v["kind"] == "ckpt":
                            pol = v["policy"]
                            if isinstance(pol, HDSPolicy):
                                T2, lay, st = pol(X, H, W, v["ret"])
                                ki = lay["kept_idx"]
                            else:
                                ki = uniform_indices(T * N, v["ret"], device)
                                T2, _ = pol(X, ki, H, W)
                            ke = T2.to(feat.dtype)
                        else:
                            raise ValueError(v["kind"])
                        cache["kept"][v["name"]] = (ki, ke)
                        st = {k: x for k, x in st.items() if isinstance(x, (int, float, str))}
                        v["fs"].write(json.dumps({"video_name": sample["video_name"], "T": T, "N": N,
                                                  "n_kept": int(ki.numel()), **st}) + "\n")
                        v["fs"].flush()
                vpos = (inputs["input_ids"][0] == vid).nonzero(as_tuple=True)[0]
                position_ids, _ = llm.get_rope_index(
                    input_ids=inputs["input_ids"], image_grid_thw=None, video_grid_thw=inputs["video_grid_thw"],
                    second_per_grid_ts=inputs.get("second_per_grid_ts"), attention_mask=inputs["attention_mask"],
                    mm_token_type_ids=inputs.get("mm_token_type_ids"),
                )
                for v in variants:
                    if aid in v["seen"]:
                        continue
                    ki, ke = cache["kept"][v["name"]]
                    fe, pid, am, _ = build_pruned_sequence(full_embeds, inputs, vid, vpos, position_ids, ki, ke)
                    gen = model.generate(inputs_embeds=fe, position_ids=pid, attention_mask=am,
                                         eos_token_id=processor.tokenizer.eos_token_id,
                                         pad_token_id=processor.tokenizer.pad_token_id,
                                         do_sample=False, num_beams=1, max_new_tokens=16)
                    out = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
                    v["fo"].write(json.dumps({
                        "task_name": sample.get("task_name", "unknown"), "question": sample["question"],
                        "question_id": str(sample.get("question_id", aid)), "id": aid,
                        "answer_number": sample["answer_number"], "candidates": sample["candidates"],
                        "answer": sample["answer"], "pred": out, "bucket": sample.get("bucket"),
                    }) + "\n")
                    v["fo"].flush()
                    v["seen"].add(aid)
        except Exception as e:
            print(f"SKIP {aid} ({vp}): {type(e).__name__}: {str(e)[:300]}", flush=True)
            cache.clear()
            torch.cuda.empty_cache()
    print("[eval_multi] done", flush=True)


if __name__ == "__main__":
    main()
