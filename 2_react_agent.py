import anthropic
import json
from typing import Any

# ── Initialize Anthropic Client ────────────────────────────────
client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"

# ── Define Agent Tools ─────────────────────────────────────────
tools = [
    {
        "name": "extract_part_ids",
        "description": """Extractor Agent: Extracts all engineering part IDs,
        component references, and schematic identifiers from a given
        document chunk. Returns a structured list of findings.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "document_chunk": {
                    "type": "string",
                    "description": "The text chunk to extract part IDs from"
                },
                "source_document": {
                    "type": "string",
                    "description": "Name of the source document"
                }
            },
            "required": ["document_chunk", "source_document"]
        }
    },
    {
        "name": "verify_cross_reference",
        "description": """Verifier Agent: Cross-references extracted part IDs
        against the Excel master book to verify consistency, flag anomalies,
        and identify discrepancies with human-level reasoning accuracy.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "extracted_parts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of part IDs extracted from blueprints"
                },
                "master_book_data": {
                    "type": "string",
                    "description": "Relevant rows from the Excel master book"
                }
            },
            "required": ["extracted_parts", "master_book_data"]
        }
    }
]


def run_extractor_agent(document_chunk: str, source_document: str) -> dict:
    """
    Extractor Agent: Uses ReAct loop to extract part IDs
    from engineering blueprint text.
    Thought → Action → Observation pattern.
    """
    print(f"\n[EXTRACTOR] Processing: {source_document}")

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system="""You are an expert Extractor Agent specializing in engineering 
        documents. Your role is to:
        1. THINK carefully about what part IDs and references exist
        2. ACT by extracting all identifiers systematically
        3. OBSERVE and validate your extractions before returning
        
        Always follow the Thought-Action-Observation loop.
        Return results as a JSON list of part IDs.""",
        messages=[
            {
                "role": "user",
                "content": f"""Extract all engineering part IDs and component 
                references from this document chunk.
                
                Source: {source_document}
                
                Content:
                {document_chunk}
                
                Return a JSON list of extracted part IDs."""
            }
        ]
    )

    result_text = response.content[0].text
    print(f"[EXTRACTOR] Extracted: {result_text[:200]}...")
    return {"source": source_document, "extracted_content": result_text}


def run_verifier_agent(extracted_parts: list, master_book_data: str) -> dict:
    """
    Verifier Agent: Cross-references extracted parts against
    the master book and identifies anomalies.
    """
    print(f"\n[VERIFIER] Cross-referencing {len(extracted_parts)} parts...")

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system="""You are an expert Verifier Agent specializing in engineering 
        data validation. Your role is to:
        1. THINK about potential discrepancies between blueprint and master book
        2. ACT by systematically comparing each part ID
        3. OBSERVE and report all anomalies with confidence scores
        
        Always follow the Thought-Action-Observation loop.
        Return a structured anomaly report.""",
        messages=[
            {
                "role": "user",
                "content": f"""Cross-reference these extracted part IDs against 
                the master book data. Identify any anomalies or discrepancies.
                
                Extracted Parts: {json.dumps(extracted_parts)}
                
                Master Book Data:
                {master_book_data}
                
                Return a structured report of matches, mismatches, and anomalies."""
            }
        ]
    )

    result_text = response.content[0].text
    print(f"[VERIFIER] Verification complete: {result_text[:200]}...")
    return {"verification_report": result_text}


def run_sequential_multi_agent_pipeline(
    document_chunk: str,
    source_document: str,
    master_book_data: str
) -> dict:
    """
    Sequential Multi-Agent Pattern:
    Step 1 → Extractor Agent runs first
    Step 2 → Verifier Agent receives Extractor output
    This is the core ReAct-based orchestration pattern.
    """
    print("\n" + "="*60)
    print("SEQUENTIAL MULTI-AGENT PIPELINE INITIATED")
    print("="*60)

    # ── Step 1: Extractor Agent ────────────────────────────────
    extraction_result = run_extractor_agent(document_chunk, source_document)

    # ── Step 2: Verifier Agent receives Extractor output ──────
    extracted_parts = [extraction_result["extracted_content"]]
    verification_result = run_verifier_agent(extracted_parts, master_book_data)

    # ── Combine Results ────────────────────────────────────────
    final_result = {
        "pipeline": "Sequential Multi-Agent (ReAct)",
        "source_document": source_document,
        "extraction": extraction_result,
        "verification": verification_result,
        "status": "COMPLETE"
    }

    print("\n[PIPELINE] Sequential Multi-Agent Pipeline Complete")
    return final_result


# ── Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    sample_chunk = """
    Assembly Drawing AD-2024-001
    Component: Hydraulic Control Unit
    Part IDs: HCU-445-A, HCU-445-B, VLV-220-X
    Reference Schematic: SCH-7741
    Tolerance: ±0.005mm
    Material Spec: MS-4412-Steel
    """

    sample_master_data = """
    Part ID    | Description              | Status  | Last Updated
    HCU-445-A  | Hydraulic Control Unit A | Active  | 2024-01-15
    HCU-445-B  | Hydraulic Control Unit B | Active  | 2024-01-15
    VLV-220-X  | Control Valve X          | RETIRED | 2023-06-01
    """

    result = run_sequential_multi_agent_pipeline(
        document_chunk=sample_chunk,
        source_document="AD-2024-001.pdf",
        master_book_data=sample_master_data
    )

    print("\n── FINAL OUTPUT ──")
    print(json.dumps(result, indent=2))
