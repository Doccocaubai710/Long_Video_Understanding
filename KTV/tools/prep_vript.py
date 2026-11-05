"""prep_vript.py -- chuan bi Vript (video dai, caption theo canh) cho train_rap_elastic.py --vript_index.

Moi video: lay deu 128 frame tren TOAN BO video, resize 588 x 336 (= luoi 12 x 21 token/unit nhu make_stitched_sample),
luu JPEG vao frames/<video_id>/000.jpg..127.jpg. Index jsonl: 1 dong / video
  {"video_id", "frames", "duration", "title", "scenes": [[scene_title, content], ...]}  (canh theo dung thu tu).
Chi giu video co caption day du (integrity) va duration >= --min_dur giay. Chay lai duoc (bo qua video da trich).

  python -m tools.prep_vript --root Datasets/Vript --workers 48
License Vript: chi dung cho nghien cuu hoc thuat, khong phan phoi lai.
"""
import argparse
import json
import os
from multiprocessing import Pool

FRAME_H, FRAME_W, NF = 336, 588, 128


def extract(job):
    vid, path, out = job
    if os.path.exists(os.path.join(out, f"{NF - 1:03d}.jpg")):
        return vid, True, "skip"
    try:
        import decord
        from PIL import Image
        vr = decord.VideoReader(path, num_threads=1)
        n = len(vr)
        if n < NF:
            return vid, False, f"chi {n} frame"
        idx = [min(n - 1, round(i * (n - 1) / (NF - 1))) for i in range(NF)]
        os.makedirs(out, exist_ok=True)
        for k in range(0, NF, 16):   # doc theo khoi de khong giu ca 128 frame 720p trong RAM
            arr = vr.get_batch(idx[k:k + 16]).asnumpy()
            for j, a in enumerate(arr):
                Image.fromarray(a).resize((FRAME_W, FRAME_H), Image.BICUBIC).save(os.path.join(out, f"{k + j:03d}.jpg"), quality=90)
        return vid, True, "ok"
    except Exception as e:
        return vid, False, f"{type(e).__name__}: {str(e)[:120]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Datasets/Vript")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--min_dur", type=float, default=180)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.root, "meta/vript_long_videos_meta.json")))
    scenes, integ, titles = {}, {}, {}
    for line in open(os.path.join(args.root, "captions/vript_long_videos_captions.jsonl")):
        j = json.loads(line)
        v = j["meta"]["video_id"]
        integ[v], titles[v] = j["meta"]["integrity"], j["meta"]["video_title"]
        c = j["caption"]
        scenes.setdefault(v, []).append((j["clip_id"], c.get("scene_title", ""), c.get("content", "")))
    vdir = os.path.join(args.root, "videos")
    jobs = []
    for f in sorted(os.listdir(vdir)):
        v = f[:-4]
        dur = (meta.get(v) or {}).get("duration") or 0
        if f.endswith(".mp4") and integ.get(v) and dur >= args.min_dur and len(scenes.get(v, [])) >= 3:
            jobs.append((v, os.path.join(vdir, f), os.path.join(args.root, "frames", v)))
    print(f"[prep_vript] {len(jobs)} video hop le / {len(os.listdir(vdir))} file", flush=True)
    good = set()
    with Pool(args.workers) as pool:
        for i, (v, ok, msg) in enumerate(pool.imap_unordered(extract, jobs)):
            if ok:
                good.add(v)
            else:
                print(f"  FAIL {v}: {msg}", flush=True)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    with open(os.path.join(args.root, "vript_index.jsonl"), "w") as fo:
        for v, _, out in jobs:
            if v in good:
                sc = [[t, c] for _, t, c in sorted(scenes[v])]
                fo.write(json.dumps({"video_id": v, "frames": out, "duration": meta[v]["duration"], "title": titles[v],
                                     "scenes": sc}, ensure_ascii=False) + "\n")
    print(f"[prep_vript] index: {len(good)} video -> {args.root}/vript_index.jsonl", flush=True)


if __name__ == "__main__":
    main()
