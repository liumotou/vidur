# Understanding metrics logged by the simulator

## Preliminaries

For every request, we define the following key metrics:

1. Request arrival time ($a_r$): the time at which a request enters the system
2. Request schedule time ($s_r$): the time at which a given request is scheduled for the first time (irrespective of subsequent restarts).
3. Request completion time ($c_r$): the time at which a request completes.
4. Request prefill completion time ($f_r$): the time at which prefill completes and first output token is produced.
5. Request execution time ($e_r$): the total amount of time a request spends actually executing on GPUs (across all attempts) - excluding the time request is allocated on a replica but not executing due to pipeline-bubbles etc.
6. Request preemption time ($p_r$): the total amount of time a request spends request is allocated on a replica but not executing due to pipeline-bubbles, scheduling preemptions, time between restarts, etc (aggregate across all attempts).
7. Request scheduling delay ($d_r$): the total amount for which the request is waiting before getting scheduled ($s_r - a_r$).

Note that arrival, schedule and completion time refer to a specific point in time, where as, execution, preemption time, scheduling delay refer to period of time.

## Logged Metics

1. `request_inter_arrival_delay_histogram`: Histogram of difference between arrival times of adjacent requests ($a_{r+1} - a_r$).
2. `request_num_tokens_histogram`: Histogram of number of tokens (prefill + decode) across all requests.
3. `request_num_input_tokens_histogram`: Histogram of total input-side tokens for each request, i.e. text prefill tokens plus multimodal encoder tokens.
4. `request_num_encoder_tokens_histogram`: Histogram of multimodal encoder tokens per request.
5. `request_num_modalities_histogram`: Histogram of number of modalities attached to each request.
6. `request_num_restarts_histogram`: Histogram of number of restarts for a given request. Note that this is expected to be a non-zero entity only when using vLLM or dSararthi schedulers - which restart requests in case a replica runs out of memory.
7. `request_e2e_time_cdf`: CDF of end-to-end request latency ($c_r - a_r$).
8. `request_e2e_time_normalised_cdf`: CDF of end-to-end request latency normalised by number of output tokens.
9. `request_execution_plus_preemption_times_cdf`: CDF of total time a request spends in the system excluding initial scheduling delay ($c_r - s_r$).
10. `request_scheduling_delay_cdf`: CDF of request scheduling delay ($s_r - a_r$).
11. `request_execution_time_cdf`: CDF of request execution time ($e_r$).
12. `request_preempted_time_cdf`: CDF of request preemption time ($p_r$).
13. `decode_token_execution_plus_preemption_times`: CDF of per decode token execution time and preemption time - i.e. inter-token delay observed by the user.
14. `batch_num_tokens_cdf`: CDF of total number of tokens to be processed in a batch (sum of prefill tokens + one per decode request). This distribution is useful towards understanding how the compute load is distributed across batches. Note that with iteration level scheduling a batch is formed at every iteration.
15. `batch_num_text_prefill_tokens_cdf`: CDF of text-only prefill tokens in a batch.
16. `batch_num_encoder_tokens_cdf`: CDF of multimodal encoder tokens attributed to a batch.
17. `batch_num_multimodal_requests_cdf`: CDF of the number of multimodal requests present in a batch.
18. `batch_sizes_cdf`: CDF of batch sizes - usually larger batch sizes imply higher throughput.
19. `prefill_time_e2e_cdf`: CDF of end-to-end latency to the first output token (time-to-first-byte), i.e, time elapsed since the request arrival to the point where first output is generated ($f_r - a_r$).
20. `prefill_time_execution_plus_preemption_cdf`: CDF of total prefill process time excluding the initial scheduling delay ($f_r - s_r$). This metric is useful for tracking the prefill efficiency.
21. `prefill_time_execution_plus_preemption_normalized_cdf`: Similar to `prefill_time_execution_plus_preemption_cdf`, but normalized by the number of prefill tokens. This provides distribution independent of request prefill length, and thus, easier to analyze.
22. `multimodal_prefill_e2e_time_cdf`: CDF of end-to-end time-to-first-token restricted to requests that include modalities.
23. `multimodal_prefill_time_execution_plus_preemption_normalized_cdf`: Similar to `prefill_time_execution_plus_preemption_normalized_cdf`, but normalized by total input tokens (text prefill plus encoder tokens) for multimodal requests only.
24. `decode_time_execution_plus_preemption_normalized_cdf`: CDF of total time spent processing decodes ($c_r - f_r$) normalized by the number of decode tokens. This provides an indicator similar to `decode_token_execution_plus_preemption_times`, however, this metric is presents an averaged over all decode tokens in the request.
25. `request_completions_time_series`: Time series of request completion times - this provides an indicator for makespan and helps in identifying the request processing rate (requests per second) by analyzing the slope of the curve.
26. `prefill_completions_time_series`: Time series of prefill token completion times - helps in identifying the prefill processing rate (prefill tokens per second) by analyzing the slope of the curve.
27. `decode_completions_time_series`: Time series of decode  completion times - helps in identifying the decode processing rate (decode tokens per second) by analyzing the slope of the curve.
28. `replica_{replica_id}_memory_usage_weighted_mean`: Memory usage statistics per replica-level - tracks the mean utilization value across entire execution time.
29. `replica_{replica_id}_stage_{stage_id}_busy_time_percent_weighted_mean`: Percentage of time a given replica stage is executing something on device - i.e. not waiting due to scheduling issues or pipeline bubbles.
30. `replica_{replica_id}_stage_{stage_id}_mfu_weighted_mean`: Model FLOPS Utilization (MFU) at a per replica stage level - it tell how much value we are able to extract from the hardware. MFU increases with batch size, reduced bubble time, higher prefill tokens, etc.
31. `request_arrivals_time_series`: Time series of request arrival timestamps.

## Multimodal-specific notes

For multimodal requests, the simulator currently treats `encoder_tokens` as an approximation of the extra sequence length and KV-cache pressure introduced by non-text inputs.

1. `request_num_input_tokens` is `num_prefill_tokens + num_encoder_tokens`.
2. `multimodal_prefill_*` metrics are only emitted for requests with at least one modality attached.
3. `multimodal_encoder` operation time is logged separately in operation metrics and is added on top of the normal prefill path.
