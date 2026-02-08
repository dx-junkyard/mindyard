"""
MINDYARD - Chit-Chat Node
雑談・カジュアルな会話を処理するノード

気軽で親しみやすいトーンで応答する。
"""
import logging
from typing import Any, Dict, Optional

from app.core.llm import llm_manager
from app.core.llm_provider import LLMProvider, LLMUsageRole

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """あなたはMINDYARDのチャットアシスタントです。
ユーザーとの雑談・カジュアルな会話を担当しています。

トーン:
- 親しみやすく自然な会話スタイル
- 簡潔に、でも温かみのある応答
- 相手の話題に関心を示す

注意:
- アドバイスや教訓は不要。気軽な会話として応答する
- 日本語で応答する
"""


def _get_provider() -> Optional[LLMProvider]:
    try:
        return llm_manager.get_client(LLMUsageRole.FAST)
    except Exception:
        return None


async def run_chat_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    雑談ノード: カジュアルな会話に応答

    LLMが利用できない場合はフォールバック応答を返す。
    """
    input_text = state["input_text"]
    provider = _get_provider()

    if not provider:
        return {"response": "なるほど！いいですね。"}

    try:
        await provider.initialize()
        result = await provider.generate_text(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ],
            temperature=0.7,
        )
        return {"response": result.content}
    except Exception as e:
        logger.warning(f"Chat node LLM call failed: {e}")
        return {"response": "なるほど！いいですね。"}
