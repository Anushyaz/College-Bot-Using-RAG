"""
Unit Tests for RAG Pipeline
"""

import os
import shutil
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

import pytest
from document_processor import process_document
from rag_pipeline import CollegeRAGPipeline, POLICY_FALLBACK_RESPONSE


SAMPLE_PDF_PATH = os.path.join("data", "sample_college_regulations.pdf")
TEST_STORE_DIR = "test_vector_store"


@pytest.fixture(scope="module")
def sample_chunks():
    chunks = process_document(SAMPLE_PDF_PATH, filename="sample_college_regulations.pdf", chunk_size=600, chunk_overlap=120)
    return chunks


@pytest.fixture(scope="module")
def rag_pipeline_instance(sample_chunks):
    if os.path.exists(TEST_STORE_DIR):
        shutil.rmtree(TEST_STORE_DIR)
        
    pipeline = CollegeRAGPipeline(vector_store_dir=TEST_STORE_DIR, similarity_threshold=0.25)
    pipeline.build_index(sample_chunks)
    yield pipeline
    
    # Cleanup test store
    if os.path.exists(TEST_STORE_DIR):
        shutil.rmtree(TEST_STORE_DIR)


def test_index_building_and_count(rag_pipeline_instance, sample_chunks):
    assert len(rag_pipeline_instance.chunks) == len(sample_chunks)
    assert rag_pipeline_instance.index is not None
    assert rag_pipeline_instance.index.ntotal == len(sample_chunks)


def test_retrieval_accuracy_attendance(rag_pipeline_instance):
    query = "What is the minimum attendance requirement for registered courses?"
    results = rag_pipeline_instance.retrieve(query, top_k=3)
    assert len(results) > 0
    top = results[0]
    assert top["page_number"] == 1
    assert "75%" in top["text"]


def test_retrieval_re_examination(rag_pipeline_instance):
    query = "What are the rules and fees for supplementary re-examination?"
    results = rag_pipeline_instance.retrieve(query, top_k=3)
    assert len(results) > 0
    pages = [r["page_number"] for r in results]
    assert any(p in [2, 3] for p in pages)


def test_query_out_of_domain_refusal(rag_pipeline_instance):
    query = "What is the mascot of the college football team?"
    response = rag_pipeline_instance.query(query)
    assert response["status"] == "NO_EVIDENCE_FOUND"
    assert response["answer"] == POLICY_FALLBACK_RESPONSE
    assert len(response["citations"]) == 0


def test_persistence_save_and_reload(sample_chunks):
    test_dir = "test_persistence_store"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # 1. Build and save
    pipe1 = CollegeRAGPipeline(vector_store_dir=test_dir)
    pipe1.build_index(sample_chunks)
    assert os.path.exists(os.path.join(test_dir, "faiss.index"))
    assert os.path.exists(os.path.join(test_dir, "metadata.pkl"))

    # 2. Reload into fresh instance
    pipe2 = CollegeRAGPipeline(vector_store_dir=test_dir)
    assert len(pipe2.chunks) == len(sample_chunks)
    assert pipe2.index.ntotal == len(sample_chunks)

    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
