"""JSON schemas for Gemini Structured Outputs."""

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"],
                    },
                },
                "required": ["id", "name", "type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "head": {"type": "string"},
                    "relation": {"type": "string"},
                    "tail": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["head", "relation", "tail", "evidence"],
            },
        },
    },
    "required": ["entities", "relations"],
}

SPAN_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entity_mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mention_text": {"type": "string"},
                    "sentence_index": {"type": "integer"},
                    "canonical_name": {"type": "string"},
                    "type": {"type": "string", "enum": ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"]},
                },
                "required": ["mention_text", "sentence_index", "canonical_name", "type"],
            },
        },
    },
    "required": ["entity_mentions"],
}

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_index": {"type": "integer"},
                    "keep": {"type": "boolean"},
                },
                "required": ["candidate_index", "keep"],
            },
        },
    },
    "required": ["decisions"],
}
