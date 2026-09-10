"""量化 1 + 量化 2：跨会话记忆。

量化 1（跨会话重复重述下降）：模拟「老用户二次来访、只讲半截话」的场景。
  - 同一 thread 先沉淀若干条情景记忆（门锁/音箱/灯泡/订单/地址…）。
  - 再用省略背景的追问（"那台门锁…""那个订单…"）去检索记忆。
  - 命中率 = 用户无需重复交代背景的比例 → 「重复重述下降 X%」。无记忆基线 = 0%。

量化 2（token 成本下降）：对比「全量塞历史」 vs 「记忆检索 + 摘要」的 prompt 长度。
  - 全量塞历史：把全部对话原样拼进 prompt。
  - 记忆检索 + 摘要：工作记忆（WorkingMemory 摘要，截断）+ 情景记忆 top-k 检索。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.common import count_tokens  # noqa: E402


# ============================================================
#  量化 1：跨会话重复重述下降
# ============================================================
_PRIOR_MEMORIES = [
    "我买的是星屋 A100 智能门锁",
    "我的智能音箱一直连不上 Wi-Fi",
    "我买了一个智能灯泡",
    "我的扫地机器人回充失败",
    "我下单了订单 A123",
    "我住在新疆",
]

_FOLLOWUPS = [
    ("那台门锁的保修期是多久？", ["门锁", "A100"]),
    ("上次说的音箱连不上网，现在解决了吗？", ["音箱", "Wi-Fi"]),
    ("我上次买的那个灯泡，保修多久？", ["灯泡"]),
    ("那个扫地机器人回充失败怎么办？", ["扫地", "回充"]),
    ("那个订单发货了吗？", ["订单", "A123"]),
    ("我住新疆，包裹几天能到？", ["新疆"]),
]


async def _repeat_reduction() -> dict:
    from app.memory.models import MemoryItem, MemoryType
    from app.memory.episodic import EpisodicMemory

    mem = EpisodicMemory(collection="eval_memory")
    thread = "eval_user_1"
    for i, text in enumerate(_PRIOR_MEMORIES):
        item = MemoryItem(
            id=f"eval_epi_{i}", type=MemoryType.EPISODIC, content=text,
            importance=0.5, source="eval", metadata={"thread_id": thread},
        )
        await mem.add(item)

    hits = 0
    detail = []
    for followup, keywords in _FOLLOWUPS:
        ret = await mem.search(followup, k=3, thread_id=thread)
        contents = [it.content for it in ret.items]
        hit = any(any(kw in c for kw in keywords) for c in contents)
        if hit:
            hits += 1
        detail.append({"followup": followup, "hit": hit,
                       "top1": contents[0][:20] if contents else "(空)"})

    rate = hits / len(_FOLLOWUPS)
    print(f"[量化1] 重复重述下降：{hits}/{len(_FOLLOWUPS)} 条追问命中记忆 = {rate:.1%}")
    print("  （含义：这 %d 条里用户无需重复交代背景，系统从长期记忆捞回了上下文）" % hits)
    for d in detail:
        print(f"    {'✓' if d['hit'] else '✗'} 「{d['followup']}」 → 命中记忆：{d['top1']}")
    return {"hit": hits, "total": len(_FOLLOWUPS), "rate": round(rate, 4), "detail": detail}


# ============================================================
#  量化 2：token 成本下降
# ============================================================
# 模拟「老用户历史」：16 条跨话题的历史问答（每条即一次 record 落库的情景记忆）
_HISTORY = [
    "用户问：我买的星屋 A100 智能门锁怎么设置指纹？\n助答：打开 App 进入设备设置-指纹管理，按提示录入，最多支持 50 组指纹。",
    "用户问：门锁的保修期是多久？\n助答：整机保修 1 年，主要部件如锁芯保修 2 年。",
    "用户问：我的智能音箱连不上 Wi-Fi 怎么办？\n助答：确认 Wi-Fi 为 2.4GHz 频段，重启路由器后重新配对。",
    "用户问：音箱语音唤醒失败？\n助答：检查是否静音，在安静环境下离设备 1-3 米说你好星屋。",
    "用户问：扫地机器人回充失败？\n助答：确认充电座靠墙、左右 0.5 米无障碍，清洁充电触点。",
    "用户问：扫地机器人报错 E5？\n助答：滚刷缠绕异物，关机清理滚刷与边刷后重启。",
    "用户问：我的订单 A123 发货了吗？\n助答：现货商品下单后 24-48 小时内发货。",
    "用户问：订单 A123 的物流到哪了？\n助答：可通过 App 我的订单-查看物流实时追踪。",
    "用户问：我住新疆，包裹几天能到？\n助答：偏远地区一般 5-10 天。",
    "用户问：商品签收后发现破损怎么办？\n助答：24 小时内联系客服并提供照片，运费商家承担。",
    "用户问：用支付宝付款的退款多久到账？\n助答：退款处理后即时到账，一般 1-3 个工作日。",
    "用户问：无理由退货的运费谁承担？\n助答：无理由退货退回运费由消费者承担。",
    "用户问：金卡会员有什么权益？\n助答：年消费满 5000 元享免运费、优先维修、以旧换新补贴。",
    "用户问：官方延保一年多少钱？\n助答：延保 1 年为商品价的 8%，延保 2 年为 15%。",
    "用户问：智能灯泡无法配对？\n助答：灯泡连续开关 3 次进入配网模式，App 内重新添加。",
    "用户问：忘记管理员密码？\n助答：长按后面板恢复键 5 秒恢复出厂设置。",
]


async def _token_cost() -> dict:
    """量化 2：全量塞历史 vs 记忆检索（top-k）。

    - 全量塞历史：把 16 条历史 Q:A 原样全塞进 prompt（朴素做法，随历史线性膨胀）。
    - 记忆检索：当前问题只检索相关的 top-3 条情景记忆。
    """
    from app.memory.models import MemoryItem, MemoryType
    from app.memory.episodic import EpisodicMemory

    full_tokens = count_tokens("\n".join(_HISTORY))

    mem = EpisodicMemory(collection="eval_memory_tok")
    thread = "eval_user_2"
    for i, text in enumerate(_HISTORY):
        await mem.add(MemoryItem(
            id=f"eval_tok_{i}", type=MemoryType.EPISODIC, content=text,
            importance=0.5, source="eval", metadata={"thread_id": thread},
        ))

    current_q = "那台门锁的指纹怎么重新录入？"
    ret = await mem.search(current_q, k=3, thread_id=thread)
    retrieved_tokens = count_tokens("\n".join(it.content for it in ret.items))

    reduction = 1 - retrieved_tokens / full_tokens if full_tokens else 0.0
    print(f"[量化2] token 成本：全量历史 {full_tokens} tok vs 记忆检索 top-3 {retrieved_tokens} tok"
          f" → 下降 {reduction:.1%}")
    print(f"  全量塞历史（{len(_HISTORY)} 条 Q:A 原文）：{full_tokens} tok")
    print(f"  记忆检索（当前问题相关 top-3）：{retrieved_tokens} tok")
    print(f"  命中的 3 条：")
    for it in ret.items:
        print(f"    - {it.content[:24]}…")
    return {
        "full_history_tokens": full_tokens,
        "retrieved_tokens": retrieved_tokens,
        "reduction": round(reduction, 4),
        "n_history": len(_HISTORY),
        "k": 3,
    }


async def main() -> dict:
    print("=" * 70)
    r1 = await _repeat_reduction()
    print("-" * 70)
    r2 = await _token_cost()
    print("=" * 70)
    return {"repeat_reduction": r1, "token_cost": r2}


if __name__ == "__main__":
    asyncio.run(main())
