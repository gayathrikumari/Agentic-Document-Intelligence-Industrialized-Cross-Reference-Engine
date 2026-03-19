import anthropic
import json
from dataclasses import dataclass

# ── Initialize Client ──────────────────────────────────────────
client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"


@dataclass
class EvalResult:
    """Structured output from the LLM-as-a-Judge evaluation."""
    score: float
    passed: bool
    reasoning: str
    anomalies: list
    confidence: str


def llm_judge_evaluate(
    agent_output: str,
    source_document: str,
    master_book_data: str,
    threshold: float = 0.75
) -> EvalResult:
    """
    LLM-as-a-Judge (G-Eval) framework:
    Deterministically validates agent outputs against
    ground truth engineering data.
    Eliminates manual audit bottlenecks.
    """
    print(f"\n[G-EVAL] Evaluating output from: {source_document}")

    eval_prompt = f"""You are an expert Engineering QA Judge performing 
    deterministic validation of AI agent outputs.

    Your task is to evaluate the agent's extraction and verification output
    against the ground truth master book data.

    AGENT OUTPUT TO EVALUATE:
    {agent_output}

    GROUND TRUTH (Master Book):
    {master_book_data}

    SOURCE DOCUMENT:
    {source_document}

    Evaluate on these criteria:
    1. ACCURACY: Are all extracted part IDs correct? (0-1)
    2. COMPLETENESS: Were all parts captured? (0-1)  
    3. ANOMALY DETECTION: Were discrepancies correctly identified? (0-1)
    4. HALLUCINATION CHECK: Did the agent invent any part IDs? (0-1)

    Return ONLY a valid JSON object in this exact format:
    {{
        "overall_score": <float 0.0-1.0>,
        "accuracy_score": <float 0.0-1.0>,
        "completeness_score": <float 0.0-1.0>,
        "anomaly_detection_score": <float 0.0-1.0>,
        "hallucination_score": <float 0.0-1.0>,
        "passed": <true/false>,
        "reasoning": "<detailed explanation>",
        "anomalies": ["<anomaly1>", "<anomaly2>"],
        "confidence": "<HIGH/MEDIUM/LOW>"
    }}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        temperature=0,  # Zero temperature for deterministic validation
        messages=[{"role": "user", "content": eval_prompt}]
    )

    raw_output = response.content[0].text.strip()
    raw_output = raw_output.replace("```json", "").replace("```", "").strip()
    eval_data = json.loads(raw_output)

    result = EvalResult(
        score=eval_data["overall_score"],
        passed=eval_data["passed"],
        reasoning=eval_data["reasoning"],
        anomalies=eval_data.get("anomalies", []),
        confidence=eval_data["confidence"]
    )

    status = "✅ PASSED" if result.passed else "❌ FAILED"
    print(f"[G-EVAL] Score: {result.score:.2f} | {status} | "
          f"Confidence: {result.confidence}")

    if result.anomalies:
        print(f"[G-EVAL] Anomalies detected: {result.anomalies}")

    return result


def run_evaluation_pipeline(
    agent_outputs: list,
    master_book_data: str
) -> dict:
    """
    Runs G-Eval across multiple agent outputs.
    Produces a summary report of overall pipeline accuracy.
    """
    print("\n" + "="*60)
    print("G-EVAL EVALUATION PIPELINE")
    print("="*60)

    results = []
    total_score = 0.0
    passed_count = 0

    for output in agent_outputs:
        result = llm_judge_evaluate(
            agent_output=output["content"],
            source_document=output["source"],
            master_book_data=master_book_data
        )
        results.append(result)
        total_score += result.score
        if result.passed:
            passed_count += 1

    avg_score = total_score / len(results) if results else 0
    pass_rate = (passed_count / len(results) * 100) if results else 0

    summary = {
        "total_evaluated": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": f"{pass_rate:.1f}%",
        "average_score": f"{avg_score:.2f}",
        "all_anomalies": [a for r in results for a in r.anomalies]
    }

    print(f"\n── EVALUATION SUMMARY ──")
    print(json.dumps(summary, indent=2))
    return summary


# ── Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    sample_agent_output = """
    Extracted Parts: HCU-445-A, HCU-445-B, VLV-220-X, SCH-7741
    Anomalies Found: VLV-220-X shows status RETIRED in master book
    but is still referenced in blueprint AD-2024-001.
    Recommendation: Flag for engineering review.
    """

    sample_master_data = """
    Part ID   | Status  | Last Updated
    HCU-445-A | Active  | 2024-01-15
    HCU-445-B | Active  | 2024-01-15
    VLV-220-X | RETIRED | 2023-06-01
    SCH-7741  | Active  | 2024-02-20
    """

    result = llm_judge_evaluate(
        agent_output=sample_agent_output,
        source_document="AD-2024-001.pdf",
        master_book_data=sample_master_data
    )

    print("\n── EVAL RESULT ──")
    print(f"Score: {result.score}")
    print(f"Passed: {result.passed}")
    print(f"Reasoning: {result.reasoning}")
