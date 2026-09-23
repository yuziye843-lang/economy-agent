"""数据集处理：从原版 CSV 均衡抽样 3000 条，补全字段，落盘 reviews_3000.json，并提供带 limit 的加载接口。"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .models import Review

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCE_CSV = _DATA_DIR / "online_shopping_10_cats.csv"
REVIEWS_JSON = _DATA_DIR / "reviews_3000.json"

PER_CATEGORY = 300   # 每品类抽样条数
POS_NEG_EACH = 150   # 每品类目标：好评/差评各 150
SEED = 42


def _clean(text: str) -> str:
    return str(text).replace("﻿", "").strip()


def prepare_data(source: Path | None = None, output: Path | None = None, seed: int = SEED) -> int:
    """读取原版 CSV，按 10 品类均衡抽样并补全字段，写入 reviews_3000.json。返回写入条数。"""
    source = source or SOURCE_CSV
    output = output or REVIEWS_JSON
    rng = random.Random(seed)

    by_cat: dict[str, dict[str, list[str]]] = {}
    with source.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cat = _clean(row["cat"])
            label = "1" if str(row["label"]).strip() == "1" else "0"
            by_cat.setdefault(cat, {"1": [], "0": []})[label].append(_clean(row["review"]))

    samples: list[tuple[str, str, int]] = []  # (category, content, rating)
    for cat in sorted(by_cat):
        pos, neg = by_cat[cat]["1"], by_cat[cat]["0"]
        # 差评不足 150 时用好评补足，保证每品类共 300 条
        neg_n = min(POS_NEG_EACH, len(neg))
        pos_n = min(PER_CATEGORY - neg_n, len(pos))
        rng.shuffle(pos)
        rng.shuffle(neg)
        for text in pos[:pos_n]:
            samples.append((cat, text, 5))
        for text in neg[:neg_n]:
            samples.append((cat, text, 1))

    rng.shuffle(samples)
    reviews = [
        Review(
            review_id=f"REV_{i + 1:04d}",
            order_id=f"ORD_{i + 10001}",
            user_id=f"USER_{i + 20001}",
            category=cat,
            content=content,
            rating=rating,
        )
        for i, (cat, content, rating) in enumerate(samples)
    ]
    output.write_text(
        json.dumps([r.model_dump() for r in reviews], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(reviews)


def load_reviews(path: Path | None = None, limit: int | None = None) -> list[Review]:
    """加载 reviews_3000.json；limit=None 加载全部，limit=N 只加载前 N 条（用于开发调试）。"""
    path = path or REVIEWS_JSON
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        raw = raw[:limit]
    return [Review(**item) for item in raw]


if __name__ == "__main__":
    n = prepare_data()
    print(f"✅ 已生成 {n} 条数据 → {REVIEWS_JSON.name}")
    sample = load_reviews(limit=10)
    print(f"验证：加载前 {len(sample)} 条")
    for r in sample:
        print(
            f"  {r.review_id} | {r.order_id} | {r.user_id} | {r.category} "
            f"| rating={r.rating} | {r.content[:20]}"
        )
