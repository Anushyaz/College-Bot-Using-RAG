"""
Evaluation Suite for College Regulations RAG Pipeline
Runs a standardized benchmark testing in-domain retrieval, citation correctness,
latency, and refusal of out-of-domain / unsupported queries.
"""

import time
from typing import List, Dict, Any
from rag_pipeline import CollegeRAGPipeline, POLICY_FALLBACK_RESPONSE

# Standard benchmark questions for College Regulations
BENCHMARK_DATASET = [
    {
        "id": "Q1",
        "category": "IN_DOMAIN",
        "question": "What is the minimum attendance requirement?",
        "expected_pages": [1],
        "expected_keywords": ["75%", "mandatory", "attendance"],
        "description": "Article 2: Minimum attendance percentage"
    },
    {
        "id": "Q2",
        "category": "IN_DOMAIN",
        "question": "What are the rules and fees for condonation of attendance shortfall?",
        "expected_pages": [1],
        "expected_keywords": ["65%", "50", "condonation"],
        "description": "Article 2: Condonation threshold and fee"
    },
    {
        "id": "Q3",
        "category": "IN_DOMAIN",
        "question": "What happens if a student's attendance falls below 65 percent?",
        "expected_pages": [1],
        "expected_keywords": ["detained", "fa", "re-register"],
        "description": "Article 2: Course detention for attendance"
    },
    {
        "id": "Q4",
        "category": "IN_DOMAIN",
        "question": "What are the rules and fees for applying for a re-examination?",
        "expected_pages": [2, 3],
        "expected_keywords": ["35", "30", "supplementary", "attempts"],
        "description": "Article 3: Supplementary re-examination window and fee"
    },
    {
        "id": "Q5",
        "category": "IN_DOMAIN",
        "question": "How can I apply for a leave of absence and how long is it permitted?",
        "expected_pages": [2, 3],
        "expected_keywords": ["loa-101", "two", "semesters", "registrar"],
        "description": "Article 4: Leave of Absence procedure and limits"
    },
    {
        "id": "Q6",
        "category": "IN_DOMAIN",
        "question": "What is the procedure and processing fee for obtaining an official transcript?",
        "expected_pages": [4],
        "expected_keywords": ["15", "sis", "7", "tatkal", "30"],
        "description": "Article 6: Standard and Tatkal transcript requests"
    },
    {
        "id": "Q7",
        "category": "IN_DOMAIN",
        "question": "What are the eligibility requirements for the Dean's Merit Scholarship?",
        "expected_pages": [4],
        "expected_keywords": ["3.75", "cgpa", "15", "waiver"],
        "description": "Article 5: Merit scholarship criteria"
    },
    {
        "id": "Q8",
        "category": "OUT_OF_DOMAIN",
        "question": "What is the official college sports team mascot and athletic jersey color?",
        "expected_pages": [],
        "expected_keywords": [],
        "description": "Unsupported policy query (Athletics branding)"
    },
    {
        "id": "Q9",
        "category": "OUT_OF_DOMAIN",
        "question": "Can I bring my pet golden retriever into the undergraduate dorm room?",
        "expected_pages": [],
        "expected_keywords": [],
        "description": "Unsupported query (Pet dormitory rules)"
    },
    {
        "id": "Q10",
        "category": "OUT_OF_DOMAIN",
        "question": "How do I claim a refund for a semester motorcycle parking permit?",
        "expected_pages": [],
        "expected_keywords": [],
        "description": "Unsupported query (Parking refunds)"
    }
]


def run_evaluation_suite(pipeline: CollegeRAGPipeline) -> Dict[str, Any]:
    """
    Executes benchmark queries through the RAG pipeline and computes quantitative metrics.
    """
    results_detail = []
    
    in_domain_total = 0
    in_domain_retrieval_success = 0
    citation_correct_total = 0
    answered_total = 0
    
    out_domain_total = 0
    out_domain_refusal_success = 0
    
    latencies = []

    for item in BENCHMARK_DATASET:
        q_id = item["id"]
        category = item["category"]
        question = item["question"]
        exp_pages = item["expected_pages"]

        start_t = time.time()
        res = pipeline.query(question)
        latency = int((time.time() - start_t) * 1000)
        latencies.append(latency)

        status = res["status"]
        answer = res["answer"]
        citations = res["citations"]

        retrieval_pass = False
        citation_pass = False
        refusal_pass = False

        if category == "IN_DOMAIN":
            in_domain_total += 1
            # Check if at least one expected page is present in retrieved citations
            retrieved_pages = [c["page_number"] for c in citations]
            if any(p in retrieved_pages for p in exp_pages):
                in_domain_retrieval_success += 1
                retrieval_pass = True

            # Check citation correctness
            if citations and all(c.get("source") and c.get("page_number") for c in citations):
                citation_correct_total += 1
                citation_pass = True

            if status == "ANSWERED":
                answered_total += 1

            verdict = "PASS" if (retrieval_pass and citation_pass and status == "ANSWERED") else "FAIL"

        else: # OUT_OF_DOMAIN
            out_domain_total += 1
            # Must reject with the standard message
            is_refused = (
                status == "NO_EVIDENCE_FOUND" or 
                POLICY_FALLBACK_RESPONSE.lower() in answer.lower() or
                "could not find sufficient information" in answer.lower()
            )
            if is_refused:
                out_domain_refusal_success += 1
                refusal_pass = True
                verdict = "PASS"
            else:
                verdict = "FAIL (Hallucination / False Positive)"

        results_detail.append({
            "id": q_id,
            "category": category,
            "question": question,
            "description": item["description"],
            "expected_page": str(exp_pages) if exp_pages else "None (Unsupported)",
            "retrieved_pages": [c["page_number"] for c in citations],
            "status": status,
            "latency_ms": latency,
            "verdict": verdict,
            "citations_count": len(citations),
            "answer_preview": answer[:160] + ("..." if len(answer) > 160 else "")
        })

    retrieval_rate = (
        round((in_domain_retrieval_success / in_domain_total) * 100, 1)
        if in_domain_total > 0 else 0.0
    )
    
    citation_rate = (
        round((citation_correct_total / in_domain_total) * 100, 1)
        if in_domain_total > 0 else 0.0
    )

    refusal_rate = (
        round((out_domain_refusal_success / out_domain_total) * 100, 1)
        if out_domain_total > 0 else 0.0
    )

    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    overall_accuracy = round(
        ((in_domain_retrieval_success + out_domain_refusal_success) / len(BENCHMARK_DATASET)) * 100, 1
    )

    return {
        "summary": {
            "total_questions": len(BENCHMARK_DATASET),
            "in_domain_questions": in_domain_total,
            "out_domain_questions": out_domain_total,
            "retrieval_success_rate": retrieval_rate,
            "citation_correctness_rate": citation_rate,
            "refusal_accuracy_rate": refusal_rate,
            "overall_accuracy_rate": overall_accuracy,
            "avg_latency_ms": avg_latency,
            "unsupported_rejected_count": out_domain_refusal_success
        },
        "details": results_detail
    }


if __name__ == "__main__":
    print("Initializing RAG Pipeline for standalone evaluation...")
    pipe = CollegeRAGPipeline()
    report = run_evaluation_suite(pipe)
    print("\n--- BENCHMARK RESULTS ---")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
