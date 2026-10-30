from document_processor import process_document
from rag_pipeline import CollegeRAGPipeline

chunks = process_document("data/sample_college_regulations.pdf", "sample_college_regulations.pdf")
pipeline = CollegeRAGPipeline()
indexed = pipeline.build_index(chunks)
print(f"Indexed {indexed} chunks into vector_store/")
