# Multimodal Workloads

This simulator now supports multimodal request descriptors for VLM-style workloads while preserving the original text-only request path.

## Scope

The current implementation is best suited for:

1. Image-plus-text input.
2. Text output.
3. Studies that need approximate encoder latency, input-length inflation, and higher memory pressure from non-text inputs.

The current implementation does not yet provide:

1. A separate encoder resource pool or encoder-specific scheduler.
2. Modality-specific activation-memory models.
3. Dedicated profiling data for vision/audio/video encoders.

## Trace format

The text-only trace format still works:

```csv
arrived_at,num_prefill_tokens,num_decode_tokens
0.0,374,44
4.314579,396,109
```

For multimodal traces, add a `modalities` JSON column and optionally a `request_metadata` JSON column:

```csv
arrived_at,num_prefill_tokens,num_decode_tokens,modalities,request_metadata
0.0,96,24,"[{""modality"":""image"",""item_count"":1,""encoder_tokens"":576,""encoder_time_scale"":1.35,""metadata"":{""resolution"":""448x448""}}]","{""workload_type"":""vlm_single_image_qa""}"
```

The `modalities` field accepts a JSON list. Each element can contain:

| Field | Meaning |
| --- | --- |
| `modality` | Name such as `image`, `audio`, or `video` |
| `item_count` | Number of items for the modality |
| `encoder_tokens` | Approximate encoder token budget contributed by the modality |
| `encoder_time_scale` | Multiplier applied to encoder-side execution time |
| `metadata` | Free-form JSON object |

## Fallback expanded columns

If JSON-in-CSV is inconvenient, trace replay also supports these fallback columns:

| Column | Meaning |
| --- | --- |
| `modality` | Single modality name |
| `modality_item_count` | Number of items |
| `modality_encoder_tokens` | Encoder token budget |
| `modality_encoder_time_scale` | Encoder time multiplier |
| `modality_metadata` | JSON object string |

## Synthetic generator

The synthetic generator supports multimodal request injection with:

1. `synthetic_request_generator_config_enable_multimodal`
2. `synthetic_request_generator_config_multimodal_fraction`
3. `synthetic_request_generator_config_modality_name`
4. `synthetic_request_generator_config_modality_item_count`
5. `synthetic_request_generator_config_modality_encoder_tokens`
6. `synthetic_request_generator_config_modality_encoder_time_scale`
7. `synthetic_request_generator_config_modality_metadata`
8. `synthetic_request_generator_config_request_metadata`

## Modeling details

The current approximation is:

1. `encoder_tokens` contribute to `total_input_tokens`.
2. `total_input_tokens` are used for initial block reservation and memory planning.
3. Encoder-side execution time is approximated by reusing the prefill execution-time predictor and scaling it with `encoder_time_scale`.
4. Multimodal requests still run through the existing prefill/decode scheduling path.

## Recommended workflow

1. Start with `data/processed_traces/multimodal_vlm_sample.csv`.
2. Set `trace_request_generator_config_max_encoder_tokens_per_request` to the largest encoder-token budget expected in the trace.
3. Compare the same deployment with and without multimodal traces.
4. Inspect the added metrics:
   `request_num_input_tokens`, `request_num_encoder_tokens`, `batch_num_encoder_tokens`, `batch_num_multimodal_requests`, and `multimodal_encoder`.
