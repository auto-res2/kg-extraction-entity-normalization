"""Entity normalization: mention clustering and clustered-entity relation prompts."""

import json
from collections import defaultdict


def cluster_mentions(mentions: list[dict]) -> list[dict]:
    """Group mentions by canonical_name (strip+lower for comparison).

    Returns list of clusters, each with:
        id: "e0", "e1", ...
        canonical_name: original canonical_name (from first occurrence)
        type: entity type (from first occurrence)
        mentions: list of distinct mention_text strings
    """
    # Group by normalized canonical_name
    groups: dict[str, dict] = {}
    order: list[str] = []

    for m in mentions:
        key = m["canonical_name"].strip().lower()
        if key not in groups:
            groups[key] = {
                "canonical_name": m["canonical_name"].strip(),
                "type": m["type"],
                "mention_texts": [],
            }
            order.append(key)
        # Add mention_text if not already present
        if m["mention_text"] not in groups[key]["mention_texts"]:
            groups[key]["mention_texts"].append(m["mention_text"])

    clusters = []
    for i, key in enumerate(order):
        g = groups[key]
        clusters.append({
            "id": f"e{i}",
            "canonical_name": g["canonical_name"],
            "type": g["type"],
            "mentions": g["mention_texts"],
        })

    return clusters


def build_clustered_entity_prompt(
    doc_text: str,
    clusters: list[dict],
    few_shot_text: str,
    few_shot_output: dict,
) -> str:
    """Build a relation extraction prompt that shows clustered entities with all their mentions.

    Asks the LLM to extract relations between the clustered entities.
    """
    few_shot_json = json.dumps(few_shot_output, ensure_ascii=False, indent=2)

    # Format cluster list for the prompt
    cluster_lines = []
    for c in clusters:
        mentions_str = ", ".join(f'"{m}"' for m in c["mentions"])
        cluster_lines.append(
            f"  - {c['id']}: {c['canonical_name']} (type={c['type']}, "
            f"mentions=[{mentions_str}])"
        )

    return f"""## 例
入力文書:
{few_shot_text}

出力:
{few_shot_json}

## 対象文書
{doc_text}

## 検出済みエンティティ（クラスタリング済み）
以下のエンティティが文書中に検出されました。各エンティティには複数のメンション（言及表現）があります。

{chr(10).join(cluster_lines)}

上記のエンティティ間の関係を抽出してください。
- headとtailには上記のエンティティIDを使用してください。
- entitiesには上記のエンティティリストをそのまま使用してください。
- 関係の根拠となるテキストをevidenceとして付与してください。"""
