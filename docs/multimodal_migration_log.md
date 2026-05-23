# Multimodal Migration Log

This document records the multimodal migration work applied to this repository so future contributors or external AI assistants can understand the intent, scope, and validation status of each step.

## Validation status

As of the latest change set, the repository has passed a minimal multimodal smoke test:

1. Python dependencies were installed successfully, including `pandas` and `scipy`.
2. The simulator successfully loaded `data/processed_traces/multimodal_vlm_sample.csv`.
3. The simulator successfully completed a run using a lightweight linear-regression execution-time predictor configuration.

The smoke-test goal was functional validation, not final accuracy validation.

## Important caveat

The default random-forest execution-time predictor is much slower to train. In restricted environments it may also require:

1. `*_num_training_job_threads 1` to avoid multiprocessing permission issues.
2. More wall-clock time before the first run completes.

This does not indicate a multimodal logic bug; it is an execution environment and model-training cost issue.

## Step 1: Request abstraction

Files:

1. `vidur/entities/request.py`
2. `vidur/entities/__init__.py`

Changes:

1. Added `RequestModality`.
2. Added `RequestWorkload`.
3. Extended `Request` to accept `modalities` and `metadata`.
4. Added `num_encoder_tokens`, `num_modalities`, and `total_input_tokens`.
5. Extended request serialization output.

Intent:

Move from a pure `(prefill_tokens, decode_tokens)` request model to a request object that can also describe non-text input cost.

## Step 2: Request generation

Files:

1. `vidur/config/config.py`
2. `vidur/request_generator/multimodal_request_utils.py`
3. `vidur/request_generator/synthetic_request_generator.py`
4. `vidur/request_generator/trace_replay_request_generator.py`

Changes:

1. Added synthetic multimodal generation knobs.
2. Added trace columns for `modalities` and `request_metadata`.
3. Added fallback expanded modality columns.
4. Added JSON parsing helpers for multimodal request construction.

Intent:

Allow both synthetic workloads and replay traces to generate multimodal requests without breaking the existing text-only path.

## Step 3: Phase and execution-time integration

Files:

1. `vidur/entities/request.py`
2. `vidur/entities/__init__.py`
3. `vidur/entities/batch.py`
4. `vidur/entities/execution_time.py`
5. `vidur/execution_time_predictor/base_execution_time_predictor.py`
6. `vidur/execution_time_predictor/sklearn_execution_time_predictor.py`

Changes:

1. Added `RequestPhase`.
2. Added request-level phase reporting.
3. Added batch-level multimodal aggregation fields.
4. Added `multimodal_encoder_execution_time` to `ExecutionTime`.
5. Added predictor hook for multimodal encoder cost.
6. Approximated encoder cost using existing prefill predictions plus `encoder_time_scale`.

Intent:

Make multimodal requests visible to the execution-time path instead of treating them as ordinary text-prefill requests.

## Step 4: Memory and scheduling approximation

Files:

1. `vidur/config/config.py`
2. `vidur/entities/replica.py`
3. `vidur/scheduler/utils/memory_planner.py`
4. `vidur/scheduler/replica_scheduler/base_replica_scheduler.py`
5. `vidur/scheduler/replica_scheduler/sarathi_replica_scheduler.py`
6. `vidur/scheduler/replica_scheduler/vllm_replica_scheduler.py`
7. `vidur/scheduler/replica_scheduler/lightllm_replica_scheduler.py`

Changes:

1. Added `max_input_tokens` and `max_encoder_tokens_per_request`.
2. Made replica memory planning use total input tokens instead of text-only tokens.
3. Made initial block reservation use `request.total_input_tokens`.
4. Updated schedulers to use the new initial allocation logic.

Intent:

Avoid underestimating KV-cache pressure and initial block reservation for multimodal requests.

## Step 5: Metrics

Files:

1. `vidur/metrics/constants.py`
2. `vidur/metrics/metrics_store.py`

Changes:

1. Added request histograms for input tokens, encoder tokens, and modality count.
2. Added batch metrics for text-prefill tokens, encoder tokens, and multimodal request count.
3. Added multimodal-prefill request latency metrics.
4. Added `multimodal_encoder` operation metric.

Intent:

Make multimodal behavior measurable instead of burying it inside the original text-only metrics.

## Step 6: Example data and docs

Files:

1. `data/processed_traces/multimodal_vlm_sample.csv`
2. `README.md`
3. `docs/multimodal.md`
4. `docs/metrics.md`

Changes:

1. Added a runnable multimodal sample trace.
2. Added multimodal CLI examples.
3. Added a dedicated multimodal documentation page.
4. Documented the new metrics.

Intent:

Make the migration reproducible and understandable for later contributors.

## What has been validated

Validated:

1. Multimodal trace replay loads correctly.
2. The simulator can complete a multimodal run end-to-end.
3. The new request, batch, scheduler, predictor, and metrics paths compile successfully.

Not yet fully validated:

1. Accuracy of the multimodal cost approximation versus a real VLM serving system.
2. Accuracy of the memory approximation for different modality encoders.
3. Performance and accuracy under all schedulers and deployment topologies.

## Recommended commit structure

If you want future AI systems to understand the evolution clearly, use one commit per migration step:

1. `feat: add multimodal request abstraction`
2. `feat: add multimodal request generation support`
3. `feat: integrate multimodal phases into execution-time path`
4. `feat: account for multimodal input in memory planning and scheduling`
5. `feat: add multimodal metrics`
6. `docs: add multimodal sample trace and usage guide`

## Recommended smoke-test command

Use a lightweight predictor for quick validation:

```powershell
$env:WANDB_MODE='disabled'
python -m vidur.main `
  --replica_config_device a100 `
  --replica_config_model_name meta-llama/Meta-Llama-3-8B `
  --cluster_config_num_replicas 1 `
  --replica_config_tensor_parallel_size 1 `
  --replica_config_num_pipeline_stages 1 `
  --request_generator_config_type trace_replay `
  --trace_request_generator_config_trace_file ./data/processed_traces/multimodal_vlm_sample.csv `
  --trace_request_generator_config_max_tokens 2048 `
  --trace_request_generator_config_max_encoder_tokens_per_request 1152 `
  --replica_scheduler_config_type sarathi `
  --sarathi_scheduler_config_batch_size_cap 128 `
  --sarathi_scheduler_config_chunk_size 256 `
  --execution_time_predictor_config_type linear_regression `
  --linear_regression_execution_time_predictor_config_polynomial_degree 2 `
  --linear_regression_execution_time_predictor_config_polynomial_include_bias true `
  --linear_regression_execution_time_predictor_config_polynomial_interaction_only false `
  --linear_regression_execution_time_predictor_config_fit_intercept true `
  --linear_regression_execution_time_predictor_config_prediction_max_prefill_chunk_size 4096 `
  --linear_regression_execution_time_predictor_config_prediction_max_batch_size 128 `
  --linear_regression_execution_time_predictor_config_prediction_max_tokens_per_request 4096 `
  --linear_regression_execution_time_predictor_config_num_training_job_threads 1 `
  --no-metrics_config_write_metrics `
  --no-metrics_config_enable_chrome_trace `
  --no-metrics_config_write_json_trace
```
