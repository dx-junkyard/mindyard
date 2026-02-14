"""
MINDYARD - User Profiler
Layer 1: 何気ない会話からユーザーの傾向・状態変化を蓄積型で把握する

カウンセリング・プロファイリング手法に基づき:
- 感情の時系列変化（バーンアウト兆候、慢性ストレス）
- トピック頻度とそれに紐づく感情（どの話題で感情が動くか）
- 投稿パターン（頻度変化、時間帯）
を集計し、ConversationAgent がより深い理解に基づいて応答できるようにする。
"""
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import uuid

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.raw_log import RawLog
from app.models.user import User

logger = logging.getLogger(__name__)

# 集計対象期間
_RECENT_DAYS = 14  # 直近2週間
_TREND_DAYS = 7    # 傾向比較の短期ウィンドウ


class UserProfiler:
    """
    蓄積型ユーザープロファイラー

    ユーザーの過去ログを集計し、以下を含むプロファイルを生成:
    - emotion_trends: 感情の出現頻度と変化
    - topic_emotion_map: トピックごとの感情傾向
    - posting_pattern: 投稿頻度・時間帯
    - detected_signals: カウンセリング的に注目すべきシグナル
    """

    async def build_profile(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
    ) -> Dict:
        """
        ユーザーの直近ログを集計してプロファイルを構築する。
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=_RECENT_DAYS)

        result = await session.execute(
            select(RawLog)
            .where(
                RawLog.user_id == user_id,
                RawLog.created_at >= cutoff,
                RawLog.is_analyzed == True,  # noqa: E712
            )
            .order_by(desc(RawLog.created_at))
            .limit(200)
        )
        logs = list(result.scalars().all())

        if not logs:
            return self._empty_profile()

        # ── 集計 ──
        emotion_trends = self._aggregate_emotions(logs)
        topic_emotion_map = self._aggregate_topic_emotions(logs)
        posting_pattern = self._aggregate_posting_pattern(logs)
        signals = self._detect_signals(logs, emotion_trends, posting_pattern)

        profile = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "log_count": len(logs),
            "period_days": _RECENT_DAYS,
            "emotion_trends": emotion_trends,
            "topic_emotion_map": topic_emotion_map,
            "posting_pattern": posting_pattern,
            "signals": signals,
        }

        return profile

    async def build_and_save(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
    ) -> Dict:
        """プロファイルを構築し、User.profile_data に保存する。"""
        profile = await self.build_profile(session, user_id)

        user = await session.get(User, user_id)
        if user:
            user.profile_data = profile
            await session.commit()
            logger.info(f"Profile updated for user {user_id}: {len(profile.get('signals', []))} signals")

        return profile

    def generate_context_summary(self, profile: Optional[Dict]) -> Optional[str]:
        """
        プロファイルを ConversationAgent のシステムプロンプトに注入する
        自然言語のサマリーに変換する。
        """
        if not profile or profile.get("log_count", 0) < 3:
            return None

        parts: List[str] = []

        # ── 感情傾向 ──
        emotions = profile.get("emotion_trends", {})
        top_emotions = emotions.get("top_emotions", [])
        if top_emotions:
            emotion_desc = "、".join(
                f"{e['emotion']}({e['count']}回)" for e in top_emotions[:3]
            )
            parts.append(f"直近{profile.get('period_days', 14)}日間の感情傾向: {emotion_desc}")

        # ── 感情変化 ──
        trend = emotions.get("recent_trend")
        if trend and trend != "stable":
            trend_labels = {
                "more_negative": "ネガティブな感情が増加傾向",
                "more_positive": "ポジティブな感情が増加傾向",
                "fatigue_increasing": "疲労・ストレスの訴えが増加傾向",
            }
            label = trend_labels.get(trend, trend)
            parts.append(f"変化: {label}")

        # ── トピック×感情 ──
        topic_map = profile.get("topic_emotion_map", {})
        stress_topics = [
            t for t, info in topic_map.items()
            if info.get("dominant_emotion") in ("frustrated", "anxious", "angry")
        ]
        if stress_topics:
            parts.append(
                f"ストレスと関連が深いトピック: {', '.join(stress_topics[:3])}"
            )

        comfort_topics = [
            t for t, info in topic_map.items()
            if info.get("dominant_emotion") in ("achieved", "excited", "relieved")
        ]
        if comfort_topics:
            parts.append(
                f"ポジティブな感情と結びつくトピック: {', '.join(comfort_topics[:3])}"
            )

        # ── シグナル ──
        signals = profile.get("signals", [])
        for sig in signals:
            parts.append(f"⚠ {sig['description']}")

        if not parts:
            return None

        return "\n".join(parts)

    # ════════════════════════════════════════
    # 内部集計メソッド
    # ════════════════════════════════════════

    def _aggregate_emotions(self, logs: List[RawLog]) -> Dict:
        """感情の出現頻度と時系列変化を集計"""
        counter: Counter = Counter()
        recent_counter: Counter = Counter()  # 直近 _TREND_DAYS 日
        older_counter: Counter = Counter()   # それ以前

        trend_cutoff = datetime.now(timezone.utc) - timedelta(days=_TREND_DAYS)

        for log in logs:
            emotions = log.emotions or []
            for emotion in emotions:
                counter[emotion] += 1
                if log.created_at.replace(tzinfo=timezone.utc) >= trend_cutoff:
                    recent_counter[emotion] += 1
                else:
                    older_counter[emotion] += 1

        top_emotions = [
            {"emotion": e, "count": c}
            for e, c in counter.most_common(5)
        ]

        # 傾向判定
        recent_trend = self._detect_emotion_trend(recent_counter, older_counter)

        return {
            "top_emotions": top_emotions,
            "total_entries": sum(counter.values()),
            "recent_trend": recent_trend,
        }

    def _detect_emotion_trend(
        self, recent: Counter, older: Counter
    ) -> str:
        """直近と以前の感情分布を比較して傾向を判定"""
        negative = {"frustrated", "angry", "anxious", "confused"}
        positive = {"achieved", "excited", "relieved"}
        fatigue = {"frustrated", "anxious"}

        recent_total = sum(recent.values()) or 1
        older_total = sum(older.values()) or 1

        recent_neg_ratio = sum(recent[e] for e in negative) / recent_total
        older_neg_ratio = sum(older[e] for e in negative) / older_total

        recent_fatigue = sum(recent[e] for e in fatigue) / recent_total
        older_fatigue = sum(older[e] for e in fatigue) / older_total

        if recent_fatigue > older_fatigue + 0.2:
            return "fatigue_increasing"
        if recent_neg_ratio > older_neg_ratio + 0.15:
            return "more_negative"
        if recent_neg_ratio < older_neg_ratio - 0.15:
            return "more_positive"
        return "stable"

    def _aggregate_topic_emotions(self, logs: List[RawLog]) -> Dict:
        """トピックごとに紐づく感情の分布を集計"""
        topic_emotions: Dict[str, Counter] = defaultdict(Counter)

        for log in logs:
            topics = log.topics or []
            emotions = log.emotions or []
            for topic in topics:
                for emotion in emotions:
                    topic_emotions[topic][emotion] += 1

        result = {}
        for topic, emotion_counter in topic_emotions.items():
            if sum(emotion_counter.values()) < 2:
                continue  # サンプル不足は除外
            dominant = emotion_counter.most_common(1)[0]
            result[topic] = {
                "dominant_emotion": dominant[0],
                "count": sum(emotion_counter.values()),
                "distribution": dict(emotion_counter),
            }

        return result

    def _aggregate_posting_pattern(self, logs: List[RawLog]) -> Dict:
        """投稿パターン（頻度変化・時間帯）を集計"""
        if not logs:
            return {}

        # 日別投稿数
        daily_counts: Counter = Counter()
        hour_counts: Counter = Counter()

        for log in logs:
            dt = log.created_at.replace(tzinfo=timezone.utc)
            daily_counts[dt.date().isoformat()] += 1
            hour_counts[dt.hour] += 1

        total_days = max(len(daily_counts), 1)
        avg_per_day = round(len(logs) / total_days, 1)

        # 最も活発な時間帯
        peak_hours = [h for h, _ in hour_counts.most_common(3)]

        # 直近7日 vs それ以前の投稿頻度比較
        trend_cutoff = datetime.now(timezone.utc) - timedelta(days=_TREND_DAYS)
        recent_count = sum(
            1 for log in logs
            if log.created_at.replace(tzinfo=timezone.utc) >= trend_cutoff
        )
        older_count = len(logs) - recent_count

        recent_avg = recent_count / _TREND_DAYS
        older_days = max(_RECENT_DAYS - _TREND_DAYS, 1)
        older_avg = older_count / older_days

        if older_avg > 0 and recent_avg < older_avg * 0.5:
            frequency_change = "decreasing"
        elif older_avg > 0 and recent_avg > older_avg * 1.5:
            frequency_change = "increasing"
        else:
            frequency_change = "stable"

        return {
            "avg_per_day": avg_per_day,
            "peak_hours": peak_hours,
            "frequency_change": frequency_change,
            "total_logs": len(logs),
        }

    def _detect_signals(
        self,
        logs: List[RawLog],
        emotion_trends: Dict,
        posting_pattern: Dict,
    ) -> List[Dict]:
        """カウンセリング的に注目すべきシグナルを検出"""
        signals: List[Dict] = []

        # 1. 疲労の反復
        fatigue_keywords = ("疲れ", "だるい", "眠い", "しんどい", "つらい", "きつい")
        fatigue_count = sum(
            1 for log in logs
            if any(kw in (log.content or "") for kw in fatigue_keywords)
        )
        if fatigue_count >= 5:
            signals.append({
                "type": "fatigue_repetition",
                "severity": "warning",
                "description": f"疲労に関する投稿が{fatigue_count}回（{_RECENT_DAYS}日間）。慢性的な疲労の可能性",
            })

        # 2. ネガティブ感情の急増
        trend = emotion_trends.get("recent_trend")
        if trend == "fatigue_increasing":
            signals.append({
                "type": "stress_increasing",
                "severity": "warning",
                "description": "直近1週間でストレス・焦りの訴えが増加している",
            })
        elif trend == "more_negative":
            signals.append({
                "type": "negativity_increasing",
                "severity": "info",
                "description": "直近1週間でネガティブな感情が増加傾向",
            })

        # 3. 投稿頻度の急減（回避行動・無気力の兆候）
        freq_change = posting_pattern.get("frequency_change")
        if freq_change == "decreasing":
            signals.append({
                "type": "posting_decrease",
                "severity": "info",
                "description": "投稿頻度が減少傾向。無気力や回避の兆候の可能性",
            })

        # 4. STATE（状態記録）の連続
        state_count = sum(
            1 for log in logs[:20]  # 直近20件
            if log.intent and log.intent.value == "state"
        )
        if state_count >= 10:
            signals.append({
                "type": "state_dominant",
                "severity": "info",
                "description": "状態記録が多い。深い思考よりも日々の状態共有がメイン",
            })

        return signals

    def _empty_profile(self) -> Dict:
        return {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "log_count": 0,
            "period_days": _RECENT_DAYS,
            "emotion_trends": {},
            "topic_emotion_map": {},
            "posting_pattern": {},
            "signals": [],
        }


# シングルトンインスタンス
user_profiler = UserProfiler()
