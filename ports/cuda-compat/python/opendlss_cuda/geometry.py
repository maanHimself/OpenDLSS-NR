"""Geometry and block schedule transcribed from upstream src/nr_graph.cpp.

Legacy geometry uses a multiple-of-64 padded extent or an explicit validated
override. It never substitutes this for the upstream rule.
"""
from __future__ import annotations

import argparse
import json


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def field_alignment(valid: int) -> int:
    reductions, size = 0, valid
    for level in range(6):
        half = align_up((size + 1) // 2, 4)
        reductions += int(half < size)
        reductions += int(level == 0 and half % 8 != 0)
        size = half
    return 1 << reductions


def geometry_from_valid(width: int, height: int, mode: str = "upstream", *,
                        full_width: int | None = None,
                        full_height: int | None = None) -> dict:
    for name, value in (("width", width), ("height", height)):
        if type(value) is not int or value <= 0 or value > 65535:
            raise ValueError(f"{name} must be an integer in 1..65535")
    if mode == "upstream":
        if full_width is not None or full_height is not None:
            raise ValueError("upstream geometry does not accept extent overrides")
        aw, ah = field_alignment(width), field_alignment(height)
        fw, fh = max(320, align_up(width, aw)), max(320, align_up(height, ah))
        if fw % (4 * aw) == 0 and fh % (4 * ah) == 0:
            fw += aw
    elif mode == "legacy_belvedere":
        # Legacy callers align each padded dimension to a multiple of 64.
        fw = align_up(max(320, width), 64) if full_width is None else full_width
        fh = align_up(max(320, height), 64) if full_height is None else full_height
        for name, value, valid in (("full_width", fw, width), ("full_height", fh, height)):
            if type(value) is not int or value < max(320, valid) or value % 64:
                raise ValueError(f"legacy {name} requires integer >= max(320, valid), multiple of 64")
    else:
        raise ValueError(f"unknown geometry mode: {mode}")
    levels = []
    w, h = fw, fh
    for _ in range(6):
        w, h = align_up((w + 1) // 2, 4), align_up((h + 1) // 2, 4)
        levels.append({"width": w, "height": h, "rows": w * h})
    if levels[0]["width"] % 8 or levels[0]["height"] % 8:
        raise ValueError("unsupported size: level 0 must contain whole 8-pixel windows")
    return {"mode": mode, "valid_width": width, "valid_height": height,
            "full_width": fw, "full_height": fh, "full_rows": fw * fh,
            "levels": levels, "vit_tokens": w * h, "padded_vit_tokens": align_up(w * h, 64)}


def require_supported_geometry(geometry: dict) -> dict:
    """Bound this port to the tested landscape 720p allocation envelope.

    The geometry formula remains available for CPU inspection of larger sizes;
    constructing a CUDA graph at those sizes is intentionally unsupported.
    """
    if (geometry["valid_width"] > 1280 or geometry["valid_height"] > 720
            or geometry["full_width"] > 1344 or geometry["full_height"] > 768
            or geometry["padded_vit_tokens"] > 320):
        raise ValueError("Supported inference extent is at most 1280x720 valid / 1344x768 padded")
    return geometry


def block_schedule() -> list[dict]:
    """All 71 blocks, in graph order; level 6 denotes the full field."""
    stage_ranges = [(0, 0, 32, 6, "pre"), (1, 4, 32, 0, "window"),
                    (5, 8, 64, 1, "window"), (9, 14, 128, 2, "window"),
                    (15, 22, 256, 3, "window"), (23, 30, 512, 4, "split"),
                    (31, 38, 1024, 5, "global"), (39, 39, 512, 4, "merge"),
                    (40, 47, 512, 4, "split"), (48, 55, 256, 3, "window"),
                    (56, 61, 128, 2, "window"), (62, 65, 64, 1, "window"),
                    (66, 69, 32, 0, "window"), (70, 70, 32, 6, "post")]
    phases, blocks = [0] * 7, []
    shifts = [(0, 0), (4, 4), (4, 0), (0, 4)]
    down = {0: (0, 32), 4: (1, 64), 8: (2, 128), 14: (3, 256), 22: (4, 512), 30: (5, 1024)}
    up = {39: (5, 30), 48: (4, 22), 56: (3, 14), 62: (2, 8), 66: (1, 4)}
    for first, last, c, level, kind in stage_ranges:
        for block in range(first, last + 1):
            item = {"block": block, "type": kind, "channels": c, "heads": c // 32,
                    "level": level, "qkv_order": "head_q_k_v", "parameter_prefix": f"block{block}."}
            if kind not in ("global", "merge"):
                phase = phases[level]
                phases[level] += 1
                item.update(phase=phase % 4, phase_index=phase,
                            shift_x=shifts[phase % 4][0], shift_y=shifts[phase % 4][1])
            if block in down:
                dest, nc = down[block]
                item["after"] = {"operation": "pool" if block == 0 else "pool_project",
                                 "level": dest, "channels": nc, "save_skip": True}
            if block in up:
                low, skip = up[block]
                item["before"] = {"operation": "project_upsample_merge", "input_level": low,
                                  "input_channels": c * 2, "skip_block": skip}
            if block == 70:
                item["before"] = {"operation": "upsample_blend", "input_level": 0, "skip_block": 0}
            blocks.append(item)
    return blocks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    parser.add_argument("--mode", default="upstream", choices=("upstream", "legacy_belvedere"))
    parser.add_argument("--full-width", type=int)
    parser.add_argument("--full-height", type=int)
    args = parser.parse_args()
    print(json.dumps(geometry_from_valid(args.width, args.height, args.mode,
                     full_width=args.full_width, full_height=args.full_height), indent=2))
