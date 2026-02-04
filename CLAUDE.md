# Mindyard Project Context (Second Brain)

## Core Vision
個人の「迷い・プロセス・文脈」を記録し、自己記録の「副作用」として組織の知恵へ昇華させるプラットフォーム。
マッチングは目的ではなく、記録の結果として生じる「セレンディピティ」である。

## Architecture Map (3-Story Structure) 
1. **Layer 1 (Private Space):** `backend/app/services/layer1`
   - Role: Raw thought capture. No constraints.
   - Tech: Context Analyzer. Handle messy, unstructured data.
2. **Layer 2 (Gateway):** `backend/app/services/layer2`
   - Role: The Filter. Privacy protection & Distillation.
   - Tech: Insight Distiller, Privacy Sanitizer. **CRITICAL SECURITY AREA.**
3. **Layer 3 (Public Plaza):** `backend/app/services/layer3`
   - Role: Connection. Semantic search & Rule generation.
   - Tech: Serendipity Matcher, Knowledge Store.

## Developer Rules
- **Privacy First:** Never commit code that bypasses `PrivacySanitizer` in Layer 2.
- **Sync Types:** When modifying Pydantic models (`backend`), update TypeScript types (`frontend`) immediately.
- **Test Distillation:** Use the `note-structurer` skill to verify how raw logs are processed.
