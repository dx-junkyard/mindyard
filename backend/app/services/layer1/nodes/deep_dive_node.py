"""
MINDYARD - Deep-Dive Node
課題解決・深掘りを行うノード

BALANCEDモデルを使用して、問題の構造化と解決策の提示を行う。
"""
import logging
from typing import Any, Dict, Optional

from app.core.llm import llm_manager
from app.core.llm_provider import LLMProvider, LLMUsageRole

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """あなたはMINDYARDの課題解決アシスタントです。
ユーザーの課題や問題を深掘りし、構造的に整理・分析することが役割です。

手順:
1. 問題の構造化: 何が本質的な課題なのかを見極める
2. 要因分析: 考えられる原因や要因を洗い出す
3. 選択肢の提示: 複数の解決アプローチを提示する
4. 次のアクション: 具体的な次の一歩を提案する

トーン:
- 論理的で整理された応答
- 箇条書きを活用して視認性を高める
- 「答え」を押し付けるのではなく、思考を促す質問も交える
- 日本語で応答する
"""


def _get_provider() -> Optional[LLMProvider]:
    try:
        return llm_manager.get_client(LLMUsageRole.BALANCED)
    except Exception:
        return None


async def run_deep_dive_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    深掘りノード: 課題を構造化して解決策を提示

    BALANCEDモデルを使用し、品質重視の回答を生成。
    """
    input_text = state["input_text"]
    provider = _get_provider()

    if not provider:
        return {
            "response": "課題を整理してみましょう。もう少し詳しく教えていただけますか？"
        }

    try:
        await provider.initialize()
        result = await provider.generate_text(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ],
            temperature=0.4,
        )
        return {"response": result.content}
    except Exception as e:
        logger.warning(f"Deep-dive node LLM call failed: {e}")
        return {
            "response": "課題を整理してみましょう。もう少し詳しく教えていただけますか？"
        }
