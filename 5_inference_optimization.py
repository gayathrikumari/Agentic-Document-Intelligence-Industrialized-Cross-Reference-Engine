import anthropic
import time

# ── Initialize Client ──────────────────────────────────────────
client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"


def optimized_inference(
    prompt: str,
    system_prompt: str,
    temperature: float = 0.1,
    top_p: float = 0.9,
    top_k: int = 50,
    max_tokens: int = 1000,
    use_cache: bool = True
) -> dict:
    """
    Inference optimization layer:
    - KV Caching via prompt caching (ephemeral cache)
    - Temperature tuning for deterministic outputs
    - Top-p / Top-k tuning for controlled generation
    - Latency measurement for performance tracking
    """
    start_time = time.time()

    system_content = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"} if use_cache else None
        }
    ]

    if not use_cache:
        system_content[0].pop("cache_control")

    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        system=system_content,
        messages=[{"role": "user", "content": prompt}]
    )

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    input_tokens = response.usage.input_tokens
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0)
    cache_created = getattr(response.usage, "cache_creation_input_tokens", 0)

    result = {
        "response": response.content[0].text,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read_tokens": cache_read,
        "cache_created_tokens": cache_created,
        "cache_hit": cache_read > 0,
        "config": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
    }

    print(f"[INFERENCE] Latency: {latency_ms:.0f}ms | "
          f"Cache Hit: {result['cache_hit']} | "
          f"Tokens: {input_tokens} in / "
          f"{response.usage.output_tokens} out")

    return result


def benchmark_inference_configs(
    prompt: str,
    system_prompt: str
) -> list:
    """
    Benchmarks multiple inference configurations
    to identify optimal temperature/top-p settings
    for engineering document analysis tasks.
    """
    configs = [
        {"temperature": 0.0, "top_p": 1.0, "top_k": 1,
         "label": "Deterministic"},
        {"temperature": 0.1, "top_p": 0.9, "top_k": 50,
         "label": "Low Temperature"},
        {"temperature": 0.3, "top_p": 0.95, "top_k": 100,
         "label": "Balanced"},
    ]

    print("\n[BENCHMARK] Running inference configuration benchmark...")
    results = []

    for config in configs:
        label = config.pop("label")
        print(f"\n[BENCHMARK] Testing: {label}")

        result = optimized_inference(
            prompt=prompt,
            system_prompt=system_prompt,
            **config
        )
        result["config_label"] = label
        results.append(result)

    print("\n── BENCHMARK SUMMARY ──")
    for r in results:
        print(f"{r['config_label']:20} | "
              f"Latency: {r['latency_ms']:7.0f}ms | "
              f"Cache: {str(r['cache_hit']):5} | "
              f"Temp: {r['config']['temperature']}")

    return results


# ── Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    system = """You are an expert engineering document analyst.
    Extract part IDs and identify anomalies with high precision."""

    prompt = """Analyze this engineering record and extract all part IDs:
    
    Assembly: Hydraulic Control System v2
    Components referenced: HCU-445-A, VLV-220-X, SCH-7741
    Note: VLV-220-X flagged as retired in Q2 2023
    Cross-reference required against master book."""

    print("── OPTIMIZED INFERENCE ──")
    result = optimized_inference(
        prompt=prompt,
        system_prompt=system,
        temperature=0.1,
        top_p=0.9,
        use_cache=True
    )
    print(f"\nResponse: {result['response'][:300]}...")

    benchmark_inference_configs(prompt, system)
