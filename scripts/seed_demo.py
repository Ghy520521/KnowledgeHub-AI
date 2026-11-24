"""本地演示：写入 3 篇示例文档并完成向量化，然后打印一次混合检索结果。

用法：python -m scripts.seed_demo
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import Base, SessionLocal, engine, init_pgvector_extension  # noqa: E402
from app.ingestion.pipeline import process_document, register_pending_bytes  # noqa: E402
from app.models import Document  # noqa: E402
from app.retrieval.hybrid import retrieve  # noqa: E402

DEMO_DOCS = {
    "nova_buds_faq.txt": (
        "NovaBuds Pro 耳机常见问题\n"
        "问：NovaBuds Pro 左右耳机断连怎么办？\n"
        "答：先把两只耳机放回充电盒 10 秒后重新取出；若仍断连，在蓝牙设置中忽略设备后重新配对，"
        "并把耳机固件升级到最新版本。\n"
        "问：耳机续航多久？\n"
        "答：单次充电聆听约 8 小时，配合充电盒总续航约 32 小时，充电 10 分钟可使用约 2 小时。\n"
    ),
    "return_policy.md": (
        "退货与售后规则\n"
        "商品签收后 7 天内支持无理由退货，15 天内出现性能故障可换货。\n"
        "退款原路返回，银行卡退款通常 3-5 个工作日到账；若退款失败，系统会生成退款失败工单并通知客服人工处理。\n"
        "定制类商品（刻字、定制型号）不支持 7 天无理由退货。\n"
    ),
    "shipping_guide.txt": (
        "物流与发货说明\n"
        "普通订单 48 小时内发货，预售商品以详情页标注的发货时间为准。\n"
        "物流长时间不更新或包裹破损，请提供订单号并发起物流工单，客服将在 24 小时内回复。\n"
        "偏远地区可能加收偏远附加费并延长 2-3 个工作日配送。\n"
    ),
}


def main() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        init_pgvector_extension(db)
        for filename, content in DEMO_DOCS.items():
            doc = Document(
                title=filename.rsplit(".", 1)[0],
                source_filename=filename,
                file_type=filename.rsplit(".", 1)[-1],
                status="pending",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            register_pending_bytes(doc.id, content.encode("utf-8"))
            process_document(db, doc.id)
            print(f"已摄入: {filename} ({doc.chunk_count} 块)")

        print("\n--- 混合检索示例：耳机断连怎么处理 ---")
        for c in retrieve(db, "耳机断连怎么处理", strategy="hybrid")[:3]:
            print(f"[RRF={c.rrf_score:.4f}] {c.document_title} :: {c.content[:80]}...")
    finally:
        db.close()


if __name__ == "__main__":
    main()
