import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

from relaykv import TierManager, build_blocks


RESULTS_DIR = Path("results")
PREVIEW_JSON = RESULTS_DIR / "tier_preview.json"

CONTEXT_LENGTHS = [2048, 4096, 8192]
HOT_WINDOW = 1024
BLOCK_SIZE = 128


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_results_dir()

    tier_manager = TierManager(hot_window=HOT_WINDOW)
    previews = []

    for seq_len in CONTEXT_LENGTHS:
        split = tier_manager.split_range(seq_len)
        blocks = build_blocks(split.cold_range, block_size=BLOCK_SIZE)

        preview = {
            "seq_len": seq_len,
            "hot_window": HOT_WINDOW,
            "block_size": BLOCK_SIZE,
            "cold_range": list(split.cold_range),
            "hot_range": list(split.hot_range),
            "num_blocks": len(blocks),
            "blocks": [
                {"block_id": b.block_id, "start": b.start, "end": b.end}
                for b in blocks
            ],
        }
        previews.append(preview)

        print(f"seq_len={seq_len}")
        print(f"  cold={split.cold_range}")
        print(f"  hot={split.hot_range}")
        print(f"  num_blocks={len(blocks)}")

    with PREVIEW_JSON.open("w", encoding="utf-8") as f:
        json.dump(previews, f, indent=2, ensure_ascii=False)

    print(f"saved: {PREVIEW_JSON}")


if __name__ == "__main__":
    main()