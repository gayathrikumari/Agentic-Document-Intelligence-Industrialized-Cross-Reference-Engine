import anthropic
import fitz  # PyMuPDF
import base64
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

# ── Configuration ──────────────────────────────────────────────
PDF_FOLDER = "data/blueprints/"
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "engineering_blueprints"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Initialize Clients ─────────────────────────────────────────
anthropic_client = anthropic.Anthropic()
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)


def extract_text_and_images_from_pdf(pdf_path: str) -> dict:
    """
    Multimodal extraction: pulls both text and images
    from engineering blueprints using PyMuPDF.
    Preserves spatial relationships in technical drawings.
    """
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_num, page in enumerate(doc):
        page_data = {
            "page": page_num + 1,
            "text": page.get_text(),
            "images": []
        }

        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            page_data["images"].append({
                "index": img_index,
                "base64": image_b64,
                "ext": base_image["ext"]
            })

        pages_data.append(page_data)

    doc.close()
    return {"source": pdf_path, "pages": pages_data}


def analyze_image_with_vlm(image_b64: str, image_ext: str) -> str:
    """
    Vision-Language Model (VLM) analysis:
    Sends blueprint images to Claude for spatial understanding
    of technical drawings and schematic relationships.
    """
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{image_ext}",
                            "data": image_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": """Analyze this engineering blueprint/schematic image.
                        Extract:
                        1. All visible part IDs and component labels
                        2. Spatial relationships between components
                        3. Technical specifications and tolerances
                        4. Any annotations or reference numbers
                        
                        Return findings in structured text format."""
                    }
                ]
            }
        ]
    )
    return response.content[0].text


def semantic_chunk_and_index(pdf_data: dict):
    """
    Semantic chunking strategy:
    Splits text using RecursiveCharacterTextSplitter
    to preserve context boundaries in technical documents.
    Then indexes chunks into ChromaDB for vector retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    source = pdf_data["source"]
    filename = os.path.basename(source)

    for page_data in pdf_data["pages"]:
        page_num = page_data["page"]
        text = page_data["text"]

        if not text.strip():
            continue

        chunks = splitter.split_text(text)

        image_context = ""
        for img in page_data["images"]:
            print(f"  [VLM] Analyzing image on page {page_num}...")
            vlm_analysis = analyze_image_with_vlm(img["base64"], img["ext"])
            image_context += f"\n[Visual Analysis]: {vlm_analysis}"

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}_p{page_num}_c{chunk_idx}"
            enriched_chunk = chunk + image_context

            collection.add(
                documents=[enriched_chunk],
                metadatas=[{
                    "source": filename,
                    "page": page_num,
                    "chunk": chunk_idx
                }],
                ids=[chunk_id]
            )

    print(f"[RAG] Indexed {filename} into ChromaDB")


def retrieve_context(query: str, n_results: int = 5) -> str:
    """
    Retrieves the most semantically relevant chunks
    from ChromaDB for a given engineering query.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    context_parts = []
    for doc, metadata in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        context_parts.append(
            f"[Source: {metadata['source']} | "
            f"Page: {metadata['page']}]\n{doc}"
        )

    return "\n\n---\n\n".join(context_parts)


def multimodal_rag_query(query: str) -> str:
    """
    Full Multimodal RAG pipeline:
    1. Retrieve relevant context from ChromaDB
    2. Send context + query to Claude for answer generation
    """
    print(f"\n[RAG QUERY] {query}")

    context = retrieve_context(query)

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system="""You are an expert engineering document analyst.
        Answer questions using ONLY the provided context from
        engineering blueprints and schematics.
        Always cite the source document and page number.""",
        messages=[
            {
                "role": "user",
                "content": f"""Context from engineering documents:
                {context}
                
                Question: {query}
                
                Provide a precise answer citing specific sources."""
            }
        ]
    )

    answer = response.content[0].text
    print(f"[RAG ANSWER] {answer[:200]}...")
    return answer


def ingest_all_blueprints():
    """
    Ingests all PDF blueprints from the data folder
    into the ChromaDB vector store.
    """
    if not os.path.exists(PDF_FOLDER):
        print(f"[ERROR] PDF folder not found: {PDF_FOLDER}")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    print(f"[RAG] Found {len(pdf_files)} blueprints to ingest")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"[RAG] Processing: {pdf_file}")
        pdf_data = extract_text_and_images_from_pdf(pdf_path)
        semantic_chunk_and_index(pdf_data)

    print(f"\n[RAG] Ingestion complete. {len(pdf_files)} documents indexed.")


# ── Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    ingest_all_blueprints()

    test_query = "What are the part IDs for the hydraulic control unit?"
    answer = multimodal_rag_query(test_query)
    print("\n── RAG ANSWER ──")
    print(answer)
