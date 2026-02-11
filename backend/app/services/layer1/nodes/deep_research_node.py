"""
MINDYARD - Deep Research Node
Gemini API を使用した Deep Research ノード

ユーザーの承認後に実行され、元のクエリに基づいて詳細なリサーチを行い、
結果をチャット履歴に返す。
"""
import os
from typing import Any, Dict, Optional

from app.core.llm import llm_manager
from app.core.llm_provider import LLMProvider, LLMUsageRole
from app.core.logger import get_traced_logger
from app.core.config import settings

logger = get_traced_logger("DeepResearchNode")

_SYSTEM_PROMPT = """あなたはMINDYARDの Deep Research アシスタントです。
ユーザーのクエリに対して、徹底的かつ包括的な調査レポートを作成してください。

### 調査方針:
1. **多角的な視点**: 複数の観点からトピックを分析する
2. **構造化された回答**: 見出し・箇条書きを使って情報を整理する
3. **エビデンスベース**: 主張には根拠や出典の方向性を示す
4. **実用性重視**: ユーザーが次のアクションを取れるような具体的な情報を提供する

### 出力フォーマット:
- 概要（1-2文のサマリー）
- 主要な発見・知見（箇条書き）
- 詳細分析（各ポイントの掘り下げ）
- 次のステップの提案

### 注意事項:
- 日本語で応答する
- 確証のない情報は「〜の可能性があります」等と明記する
- 専門用語には簡潔な説明を付ける
"""

# Gemini Deep Research 用のモデル設定
# 環境変数 GEMINI_DEEP_RESEARCH_MODEL で上書き可能
_DEFAULT_RESEARCH_MODEL = "gemini-2.0-flash-thinking-exp"


def _get_research_provider() -> Optional[LLMProvider]:
    """Deep Research 用のプロバイダーを取得（DEEP ティアを使用）"""
    try:
        return llm_manager.get_client(LLMUsageRole.DEEP)
    except Exception:
        return None


async def run_deep_research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep Research ノード: Gemini API を使用した詳細リサーチ

    ユーザーの元のクエリに基づいて詳細なリサーチを実行し、
    包括的な調査レポートを生成する。
    """
    input_text = state["input_text"]
    previous_response = state.get("previous_response", "")
    provider = _get_research_provider()

    if not provider:
        return {
            "response": (
                "申し訳ありません。Deep Research サービスが現在利用できません。\n"
                "通常の回答をご参照ください。"
            ),
        }

    try:
        await provider.initialize()

        # コンテキストとして前回の回答も含める
        research_query = input_text
        if previous_response:
            research_query = (
                f"元の質問: {input_text}\n\n"
                f"初回の回答（これを深掘りしてください）:\n{previous_response}"
            )

        logger.info(
            "Deep Research request",
            metadata={"query_preview": research_query[:200]},
        )

        result = await provider.generate_text(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": research_query},
            ],
            temperature=0.3,
        )

        research_response = result.content
        logger.info(
            "Deep Research completed",
            metadata={"response_preview": research_response[:200]},
        )

        return {
            "response": f"🔬 **Deep Research 結果**\n\n{research_response}",
        }

    except Exception as e:
        logger.warning("Deep Research failed", metadata={"error": str(e)})
        return {
            "response": (
                "Deep Research の実行中にエラーが発生しました。\n"
                "通常の回答をご参照ください。再度お試しいただくこともできます。"
            ),
        }
