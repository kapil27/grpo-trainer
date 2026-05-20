"""Build Grafana dashboard JSON for vLLM + cluster GPU metrics."""


def _ts(title, exprs, grid_pos, unit="short", legend_formats=None):
    targets = []
    for i, expr in enumerate(exprs):
        lf = legend_formats[i] if legend_formats and i < len(legend_formats) else f"series {i}"
        targets.append({"expr": expr, "legendFormat": lf, "refId": chr(65 + i)})
    return {
        "title": title,
        "type": "timeseries",
        "gridPos": grid_pos,
        "targets": targets,
        "fieldConfig": {"defaults": {"unit": unit}},
    }


def _stat(title, expr, grid_pos, unit="short", decimals=0):
    return {
        "title": title,
        "type": "stat",
        "gridPos": grid_pos,
        "targets": [{"expr": expr, "refId": "A"}],
        "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals}},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
    }


def _row(title, y):
    return {
        "title": title,
        "type": "row",
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "panels": [],
    }


def build_dashboard(cluster_nodes: int = 0, gpu_nodes: int = 0, total_gpus_alloc: int = 0):
    """cluster_nodes/gpu_nodes/total_gpus_alloc from K8s API (shown in panel descriptions)."""
    infra_note = (
        f"K8s: {cluster_nodes} nodes, {gpu_nodes} GPU nodes, "
        f"{total_gpus_alloc} GPUs allocatable (from API at deploy time)"
    )

    panels = [
        _row("Cluster — GPU Infrastructure (NVIDIA DCGM)", 0),
        _stat(
            "GPU Nodes (reporting)",
            "count(count by (Hostname) (DCGM_FI_DEV_GPU_UTIL))",
            {"h": 4, "w": 6, "x": 0, "y": 1},
            unit="short",
        ),
        _stat(
            "Total GPUs (scraped)",
            "count(DCGM_FI_DEV_GPU_UTIL)",
            {"h": 4, "w": 6, "x": 6, "y": 1},
            unit="short",
        ),
        _stat(
            "Avg GPU Utilization",
            "avg(DCGM_FI_DEV_GPU_UTIL)",
            {"h": 4, "w": 6, "x": 12, "y": 1},
            unit="percent",
            decimals=1,
        ),
        _stat(
            "Avg GPU Memory Used",
            "avg(DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) * 100)",
            {"h": 4, "w": 6, "x": 18, "y": 1},
            unit="percent",
            decimals=1,
        ),
        {
            "title": "Cluster inventory (from notebook deploy)",
            "type": "text",
            "gridPos": {"h": 2, "w": 24, "x": 0, "y": 5},
            "options": {
                "mode": "markdown",
                "content": f"**{infra_note}** — GPU metrics from `nvidia-dcgm-exporter` DaemonSet (all GPU nodes).",
            },
        },
        _ts(
            "GPU Utilization % (by node / GPU)",
            ['DCGM_FI_DEV_GPU_UTIL{job="dcgm"}'],
            {"h": 8, "w": 12, "x": 0, "y": 7},
            unit="percent",
            legend_formats=['{{Hostname}} GPU {{gpu}}'],
        ),
        _ts(
            "GPU Memory Used %",
            [
                'DCGM_FI_DEV_FB_USED{job="dcgm"} / (DCGM_FI_DEV_FB_USED{job="dcgm"} + DCGM_FI_DEV_FB_FREE{job="dcgm"}) * 100'
            ],
            {"h": 8, "w": 12, "x": 12, "y": 7},
            unit="percent",
            legend_formats=['{{Hostname}} GPU {{gpu}}'],
        ),
        _ts(
            "GPU Temperature (C)",
            ['DCGM_FI_DEV_GPU_TEMP{job="dcgm"}'],
            {"h": 8, "w": 12, "x": 0, "y": 15},
            unit="celsius",
            legend_formats=['{{Hostname}} GPU {{gpu}}'],
        ),
        _ts(
            "GPU Power (W)",
            ['DCGM_FI_DEV_POWER_USAGE{job="dcgm"}'],
            {"h": 8, "w": 12, "x": 12, "y": 15},
            unit="watt",
            legend_formats=['{{Hostname}} GPU {{gpu}}'],
        ),
        _ts(
            "GPU SM Clock (MHz)",
            ['DCGM_FI_DEV_SM_CLOCK{job="dcgm"}'],
            {"h": 8, "w": 12, "x": 0, "y": 23},
            unit="hertz",
            legend_formats=['{{Hostname}} GPU {{gpu}}'],
        ),
        _ts(
            "GPU Memory Clock (MHz)",
            ['DCGM_FI_DEV_MEM_CLOCK{job="dcgm"}'],
            {"h": 8, "w": 12, "x": 12, "y": 23},
            unit="hertz",
            legend_formats=['{{Hostname}} GPU {{gpu}}'],
        ),
        {
            "title": "Per-GPU snapshot",
            "type": "table",
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 31},
            "targets": [
                {
                    "expr": 'DCGM_FI_DEV_GPU_UTIL{job="dcgm"}',
                    "format": "table",
                    "instant": True,
                    "refId": "util",
                },
                {
                    "expr": 'DCGM_FI_DEV_FB_USED{job="dcgm"}',
                    "format": "table",
                    "instant": True,
                    "refId": "mem_used",
                },
                {
                    "expr": 'DCGM_FI_DEV_GPU_TEMP{job="dcgm"}',
                    "format": "table",
                    "instant": True,
                    "refId": "temp",
                },
            ],
        },
        _row("vLLM Inference", 39),
    ]

    vllm_y = 40
    vllm_panels = [
        _ts(
            "Requests Running / Waiting",
            [
                'vllm:num_requests_running{job="vllm"}',
                'vllm:num_requests_waiting{job="vllm"}',
            ],
            {"h": 8, "w": 12, "x": 0, "y": vllm_y},
            legend_formats=["Running", "Waiting"],
        ),
        _ts(
            "KV Cache Usage (%)",
            ['vllm:kv_cache_usage_perc{job="vllm"} * 100'],
            {"h": 8, "w": 12, "x": 12, "y": vllm_y},
            unit="percent",
            legend_formats=["KV Cache %"],
        ),
        _ts(
            "Token Throughput (tokens/s)",
            [
                'rate(vllm:prompt_tokens_total{job="vllm"}[30s])',
                'rate(vllm:generation_tokens_total{job="vllm"}[30s])',
            ],
            {"h": 8, "w": 12, "x": 0, "y": vllm_y + 8},
            legend_formats=["Prompt tokens/s", "Generation tokens/s"],
        ),
        _ts(
            "Successful Requests (rate/s)",
            ['rate(vllm:request_success_total{job="vllm"}[30s])'],
            {"h": 8, "w": 12, "x": 12, "y": vllm_y + 8},
            unit="reqps",
            legend_formats=["Requests/s"],
        ),
        _ts(
            "E2E Request Latency (p50 / p95 / p99)",
            [
                'histogram_quantile(0.50, rate(vllm:e2e_request_latency_seconds_bucket{job="vllm"}[1m]))',
                'histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket{job="vllm"}[1m]))',
                'histogram_quantile(0.99, rate(vllm:e2e_request_latency_seconds_bucket{job="vllm"}[1m]))',
            ],
            {"h": 8, "w": 12, "x": 0, "y": vllm_y + 16},
            unit="s",
            legend_formats=["p50", "p95", "p99"],
        ),
        _ts(
            "Time to First Token (p50 / p95 / p99)",
            [
                'histogram_quantile(0.50, rate(vllm:time_to_first_token_seconds_bucket{job="vllm"}[1m]))',
                'histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket{job="vllm"}[1m]))',
                'histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket{job="vllm"}[1m]))',
            ],
            {"h": 8, "w": 12, "x": 12, "y": vllm_y + 16},
            unit="s",
            legend_formats=["p50", "p95", "p99"],
        ),
        _ts(
            "Inter-Token Latency (p50 / p95)",
            [
                'histogram_quantile(0.50, rate(vllm:inter_token_latency_seconds_bucket{job="vllm"}[1m]))',
                'histogram_quantile(0.95, rate(vllm:inter_token_latency_seconds_bucket{job="vllm"}[1m]))',
            ],
            {"h": 8, "w": 12, "x": 0, "y": vllm_y + 24},
            unit="s",
            legend_formats=["p50", "p95"],
        ),
        _ts(
            "Cumulative Tokens Processed",
            [
                'vllm:prompt_tokens_total{job="vllm"}',
                'vllm:generation_tokens_total{job="vllm"}',
            ],
            {"h": 8, "w": 12, "x": 12, "y": vllm_y + 24},
            legend_formats=["Prompt tokens", "Generation tokens"],
        ),
        _ts(
            "Prefill / Decode / Queue Time (p95)",
            [
                'histogram_quantile(0.95, rate(vllm:request_prefill_time_seconds_bucket{job="vllm"}[1m]))',
                'histogram_quantile(0.95, rate(vllm:request_decode_time_seconds_bucket{job="vllm"}[1m]))',
                'histogram_quantile(0.95, rate(vllm:request_queue_time_seconds_bucket{job="vllm"}[1m]))',
            ],
            {"h": 8, "w": 12, "x": 0, "y": vllm_y + 32},
            unit="s",
            legend_formats=["Prefill p95", "Decode p95", "Queue p95"],
        ),
        _ts(
            "Preemptions (cumulative)",
            ['vllm:num_preemptions_total{job="vllm"}'],
            {"h": 8, "w": 12, "x": 12, "y": vllm_y + 32},
            legend_formats=["Preemptions"],
        ),
    ]
    panels.extend(vllm_panels)

    return {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "panels": panels,
        "refresh": "5s",
        "schemaVersion": 39,
        "tags": ["vllm", "grpo", "gpu", "dcgm"],
        "templating": {"list": []},
        "time": {"from": "now-15m", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": "vLLM + GPU Cluster Monitor",
        "uid": "vllm-grpo-monitor",
        "version": 2,
    }
