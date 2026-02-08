"""
MINDYARD - Intent Router
LLMを使ってユーザー入力を5つのカテゴリに分類する

分類カテゴリ:
- chat: 雑談・カジュアル
- empathy: 感情的・共感要求
- knowledge: 知識要求・質問
- deep_dive: 課題解決・深掘り
- brainstorm: 発想・アイデア出し
"""
import logging
from typing import Optional

from app.core.llm import llm_manager
from app.core.llm_provider import LLMProvider, LLMUsageRole
from app.schemas.conversation import ConversationIntent

logger = logging.getLogger(__name__)

# ルールベース判定のキーワードマッピング
_KEYWORD_MAP = {
    ConversationIntent.EMPATHY: [
        "つらい", "しんどい", "疲れた", "嫌だ", "ひどい", "悲しい",
        "不安", "怖い", "寂しい", "イライラ", "ムカつく", "最悪",
        "聞いて", "吐き出し", "愚痴", "ため息",
    ],
    ConversationIntent.KNOWLEDGE: [
        "教えて", "知りたい", "とは", "って何", "ですか",
        "違いは", "方法は", "やり方", "調べ", "検索",
        "参考", "文献", "論文", "データ",
    ],
    ConversationIntent.DEEP_DIVE: [
        "どうすれば", "解決", "改善", "対策", "問題",
        "原因", "なぜ", "課題", "困って", "うまくいかない",
        "分析", "検討", "整理したい", "深掘り",
    ],
    ConversationIntent.BRAINSTORM: [
        "アイデア", "案", "ひらめき", "思いつき", "仮説",
        "壁打ち", "ブレスト", "発想", "もし", "可能性",
        "新しい", "試したい", "どうだろう", "妄想",
    ],
}


class IntentRouter:
    """
    Intent Router (意図分類器)

    LLMを使ってユーザー入力を5カテゴリに分類する。
    FASTモデルを使用して低レイテンシで処理。
    LLMが利用できない場合はキーワードベースのフォールバックを使用。
    """

    def __init__(self):
        self._provider: Optional[LLMProvider] = None

    def _get_provider(self) -> Optional[LLMProvider]:
        """LLMプロバイダーを取得（遅延初期化）"""
        if self._provider is None:
            try:
                self._provider = llm_manager.get_client(LLMUsageRole.FAST)
            except Exception:
                pass
        return self._provider

    async def classify(self, input_text: str) -> dict:
        """
        入力テキストの意図を分類

        Returns:
            {
                "intent": ConversationIntent,
                "confidence": float (0.0-1.0),
            }
        """
        provider = self._get_provider()
        if not provider:
            return self._fallback_classify(input_text)

        try:
            await provider.initialize()
            result = await provider.generate_json(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": input_text},
                ],
                temperature=0.1,
            )
            return self._parse_result(result)
        except Exception as e:
            logger.warning(f"Intent classification via LLM failed: {e}")
            return self._fallback_classify(input_text)

    def _get_system_prompt(self) -> str:
        return """あなたはユーザー入力の意図分類器です。
入力テキストを以下の5カテゴリのいずれかに分類してください。

カテゴリ:
- "chat": 雑談、挨拶、日常会話、特に目的のない会話
- "empathy": 感情的な表現、愚痴、不満、悩み、共感を求めている
- "knowledge": 知識や情報を求めている質問、「〜とは？」「〜の方法は？」
- "deep_dive": 具体的な課題や問題の解決を求めている、分析・整理したい
- "brainstorm": アイデア出し、仮説検証、壁打ち、創造的な発想を求めている

必ず以下のJSON形式で応答してください:
{
    "intent": "chat" | "empathy" | "knowledge" | "deep_dive" | "brainstorm",
    "confidence": 0.0〜1.0
}"""

    def _parse_result(self, result: dict) -> dict:
        """LLM結果をパース"""
        intent_str = result.get("intent", "chat").lower()
        intent_map = {
            "chat": ConversationIntent.CHAT,
            "empathy": ConversationIntent.EMPATHY,
            "knowledge": ConversationIntent.KNOWLEDGE,
            "deep_dive": ConversationIntent.DEEP_DIVE,
            "brainstorm": ConversationIntent.BRAINSTORM,
        }
        intent = intent_map.get(intent_str, ConversationIntent.CHAT)

        confidence = result.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        confidence = max(0.0, min(1.0, float(confidence)))

        return {"intent": intent, "confidence": confidence}

    def _fallback_classify(self, input_text: str) -> dict:
        """キーワードベースのフォールバック分類"""
        scores = {intent: 0.0 for intent in ConversationIntent}

        for intent, keywords in _KEYWORD_MAP.items():
            for kw in keywords:
                if kw in input_text:
                    scores[intent] += 1.0

        # 最も高いスコアのインテントを選択
        max_intent = max(scores, key=scores.get)
        max_score = scores[max_intent]

        if max_score == 0:
            # どのキーワードにもマッチしない場合はchat
            return {"intent": ConversationIntent.CHAT, "confidence": 0.3}

        # スコアを正規化して信頼度とする
        total = sum(scores.values())
        confidence = max_score / total if total > 0 else 0.3

        return {"intent": max_intent, "confidence": min(confidence, 0.7)}


# シングルトンインスタンス
intent_router = IntentRouter()
