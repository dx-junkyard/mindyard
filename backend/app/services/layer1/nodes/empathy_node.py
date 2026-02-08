"""
MINDYARD - Empathy Node
感情的な入力に対して共感を示すノード

共感特化のプロンプトで、聞く姿勢を重視した応答を生成する。
"""
import logging
from typing import Any, Dict, Optional

from app.core.llm import llm_manager
from app.core.llm_provider import LLMProvider, LLMUsageRole

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """あなたはMINDYARDの傾聴アシスタントです。
ユーザーの感情に寄り添い、共感を示すことが役割です。

重要なルール:
- 絶対にアドバイスや解決策を提示しない
- ユーザーの感情を受け止め、共感を言葉にする
- 「〜すべき」「〜したらどうですか」は禁止
- 感情のラベリングを行う（「それは悔しいですよね」「不安になりますよね」）
- 話を聞いている姿勢を明確に示す
- 日本語で応答する

応答パターン例:
- 「それは本当に大変でしたね。」
- 「そう感じるのは当然だと思います。」
- 「話してくれてありがとうございます。」
"""


def _get_provider() -> Optional[LLMProvider]:
    try:
        return llm_manager.get_client(LLMUsageRole.FAST)
    except Exception:
        return None


async def run_empathy_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    共感ノード: 感情的な入力に共感を示す応答

    アドバイスは一切行わず、傾聴に徹する。
    """
    input_text = state["input_text"]
    provider = _get_provider()

    if not provider:
        return {"response": "お気持ち、受け止めました。話してくれてありがとうございます。"}

    try:
        await provider.initialize()
        result = await provider.generate_text(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ],
            temperature=0.5,
        )
        return {"response": result.content}
    except Exception as e:
        logger.warning(f"Empathy node LLM call failed: {e}")
        return {"response": "お気持ち、受け止めました。話してくれてありがとうございます。"}
