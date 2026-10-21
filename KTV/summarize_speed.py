"""summarize_speed.py -- bang tong hop cho bench_speed_rap.py (trung binh tren cac video).

Usage: python summarize_speed.py outputs/speed/bench_f128.jsonl
"""

import json
import sys
from collections import defaultdict


def main(path):
    rows = [json.loads(l) for l in open(path)]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["retention"])].append(r)

    def mean(rs, k):
        v = [r[k] for r in rs if k in r]
        return sum(v) / len(v) if v else float("nan")

    dense = groups[("dense", 1.0)]
    d_prefill, d_ttft, d_e2e = mean(dense, "prefill_ms"), mean(dense, "ttft_ms"), mean(dense, "e2e_ms")
    d_llm = mean(dense, "prefill_flops")
    d_all = d_llm + mean(dense, "vision_flops")
    print(f"{len(dense)} video, vision encoder = {mean(dense, 'vision_ms'):.0f} ms, "
          f"{mean(dense, 'vision_flops') / 1e12:.1f} TFLOPs (giong nhau moi cau hinh)\n")
    hdr = ("cau hinh", "token", "nen ms", "prefill ms", "prefill x", "TTFT ms", "TTFT x", "e2e ms",
           "peak GB", "LLM TFLOPs", "LLM %", "nen TFLOPs", "tong %")
    print(" | ".join(hdr))
    order = sorted(groups, key=lambda k: (-k[1], k[0] != "uniform"))
    for m, r in order:
        g = groups[(m, r)]
        llm = mean(g, "prefill_flops")
        comp = mean(g, "compress_flops")
        tot = llm + comp + mean(g, "vision_flops")
        name = "dense" if m == "dense" else f"{m} {r:.0%}"
        print(" | ".join([
            name, f"{mean(g, 'n_video'):.0f}", f"{mean(g, 'compress_ms'):.1f}", f"{mean(g, 'prefill_ms'):.0f}",
            f"{d_prefill / mean(g, 'prefill_ms'):.1f}", f"{mean(g, 'ttft_ms'):.0f}", f"{d_ttft / mean(g, 'ttft_ms'):.2f}",
            f"{mean(g, 'e2e_ms'):.0f}", f"{mean(g, 'peak_gb'):.2f}", f"{llm / 1e12:.1f}", f"{100 * llm / d_llm:.1f}",
            f"{comp / 1e12:.2f}", f"{100 * tot / d_all:.1f}",
        ]))


if __name__ == "__main__":
    main(sys.argv[1])
