"""命令行评测：加载标注集，输出各策略 Recall@K / MRR / nDCG@K 与平均延迟。

用法：
    python -m scripts.run_eval --set data/eval/eval_set.jsonl --k 5 --out reports/eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import Base, SessionLocal, engine, init_pgvector_extension  # noqa: E402
from app.retrieval.evaluation import run_eval  # noqa: E402


def load_samples(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True, help="标注集 JSONL 路径")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", help="把报告写入 JSON 文件")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        init_pgvector_extension(db)
        samples = load_samples(Path(args.set))
        report = run_eval(db, samples, k=args.k)
    finally:
        db.close()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
