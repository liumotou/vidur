import json
from typing import Any, Dict, List

from vidur.config import SyntheticRequestGeneratorConfig, TraceRequestGeneratorConfig
from vidur.entities import RequestModality


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _parse_json_object(raw_value: Any, field_name: str) -> Dict[str, Any]:
    if _is_missing(raw_value):
        return {}

    if isinstance(raw_value, dict):
        return raw_value

    if isinstance(raw_value, str):
        parsed = json.loads(raw_value)
        if not isinstance(parsed, dict):
            raise ValueError(f"{field_name} must be a JSON object.")
        return parsed

    raise ValueError(f"{field_name} must be a JSON object or string.")


def _parse_modalities_json(raw_value: Any, field_name: str) -> List[RequestModality]:
    if _is_missing(raw_value):
        return []

    parsed = raw_value
    if isinstance(raw_value, str):
        parsed = json.loads(raw_value)

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} must be a JSON list or object.")

    modalities = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError(f"{field_name} entries must be JSON objects.")
        modalities.append(
            RequestModality(
                modality=str(item["modality"]),
                item_count=int(item.get("item_count", 1)),
                encoder_tokens=int(item.get("encoder_tokens", 0)),
                encoder_time_scale=float(item.get("encoder_time_scale", 1.0)),
                metadata=dict(item.get("metadata", {})),
            )
        )
    return modalities


def build_synthetic_modalities(
    config: SyntheticRequestGeneratorConfig,
) -> List[RequestModality]:
    if not config.enable_multimodal:
        return []

    return [
        RequestModality(
            modality=config.modality_name,
            item_count=config.modality_item_count,
            encoder_tokens=config.modality_encoder_tokens,
            encoder_time_scale=config.modality_encoder_time_scale,
            metadata=_parse_json_object(
                config.modality_metadata, "synthetic_request_generator.modality_metadata"
            ),
        )
    ]


def build_synthetic_request_metadata(
    config: SyntheticRequestGeneratorConfig,
) -> Dict[str, Any]:
    return _parse_json_object(
        config.request_metadata, "synthetic_request_generator.request_metadata"
    )


def build_trace_modalities(
    row: Dict[str, Any], config: TraceRequestGeneratorConfig
) -> List[RequestModality]:
    modalities = _parse_modalities_json(
        row.get(config.modalities_column), config.modalities_column
    )
    if modalities:
        return modalities

    modality_name = row.get(config.modality_column)
    if _is_missing(modality_name):
        return []

    return [
        RequestModality(
            modality=str(modality_name),
            item_count=int(row.get(config.modality_item_count_column, 1)),
            encoder_tokens=int(row.get(config.modality_encoder_tokens_column, 0)),
            encoder_time_scale=float(
                row.get(config.modality_encoder_time_scale_column, 1.0)
            ),
            metadata=_parse_json_object(
                row.get(config.modality_metadata_column),
                config.modality_metadata_column,
            ),
        )
    ]


def build_trace_request_metadata(
    row: Dict[str, Any], config: TraceRequestGeneratorConfig
) -> Dict[str, Any]:
    return _parse_json_object(
        row.get(config.request_metadata_column),
        config.request_metadata_column,
    )
