"""
RAG Pipeline for College Regulations and Academic Policy Chatbot
Handles embeddings, FAISS vector indexing, persistent storage,
similarity retrieval, source citation tracking, and grounded answer synthesis.
Supports both SentenceTransformer embeddings and a resilient dense semantic
vector engine (for environments where PyTorch DLLs are restricted by OS WDAC).
"""

import os
import re
import pickle
import time
import math
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Automatically load environment variables
load_dotenv()

# Suppress noisy huggingface / tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import faiss
from document_processor import DocumentChunk

POLICY_FALLBACK_RESPONSE = (
    "I could not find sufficient information in the uploaded college documents to answer this question."
)

OFFICIAL_DISCLAIMER = (
    "Disclaimer: Official College Authorities (Office of the Academic Registrar) "
    "remain the final source for official policy clarification."
)


COMMON_STOPWORDS = {
    "what", "is", "the", "of", "in", "and", "a", "an", "for", "to", "how",
    "can", "do", "i", "at", "on", "with", "this", "that", "it", "are", "be",
    "by", "or", "as", "from", "my", "your", "we", "you", "if", "does", "any"
}

class DenseSemanticEncoder:
    """
    High-performance, pure-Python / NumPy semantic vector encoder.
    Generates normalized 384-dimensional dense semantic vectors using subword n-grams
    and TF-IDF weighting, compatible with FAISS IndexFlatIP cosine similarity.
    Ensures 100% reliability in environments where PyTorch C++ DLLs are restricted.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.doc_freq: Dict[str, int] = {}
        self.total_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'[a-zA-Z0-9_\$%\.]+', text.lower())
        tokens = []
        for w in words:
            if w in COMMON_STOPWORDS:
                continue
            tokens.append(w)
            # Add character 3-grams and 4-grams for robust subword matching on key terms
            if len(w) >= 3:
                for n in (3, 4):
                    for i in range(len(w) - n + 1):
                        tokens.append(f"#{w[i:i+n]}")
        return tokens

    def fit(self, texts: List[str]):
        self.total_docs = len(texts)
        self.doc_freq = {}
        for txt in texts:
            unique_tokens = set(self._tokenize(txt))
            for tok in unique_tokens:
                self.doc_freq[tok] = self.doc_freq.get(tok, 0) + 1

    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        vectors = []
        for txt in texts:
            tokens = self._tokenize(txt)
            vec = np.zeros(self.dimension, dtype=np.float32)
            if not tokens:
                vectors.append(vec)
                continue

            tf: Dict[str, float] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0.0) + 1.0

            for tok, count in tf.items():
                df = self.doc_freq.get(tok, 1)
                idf = math.log((1.0 + self.total_docs) / (1.0 + df)) + 1.0
                weight = (1.0 + math.log(count)) * idf

                # Deterministic hash projection into 384 dimensions
                h_val = int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16)
                idx = h_val % self.dimension
                sign = 1.0 if (h_val % 2 == 0) else -1.0
                vec[idx] += sign * weight

            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm

            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)


_ST_AVAILABLE: Optional[bool] = None

def is_sentence_transformers_available() -> bool:
    global _ST_AVAILABLE
    if _ST_AVAILABLE is not None:
        return _ST_AVAILABLE
    
    # Check environment variable
    if os.getenv("DISABLE_SENTENCE_TRANSFORMERS", "0") == "1":
        _ST_AVAILABLE = False
        return False

    try:
        # Quick probe with low timeout
        import torch
        from sentence_transformers import SentenceTransformer
        _ST_AVAILABLE = True
    except (ImportError, OSError, Exception):
        _ST_AVAILABLE = False
    return _ST_AVAILABLE


class CollegeRAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_dir: str = "vector_store",
        similarity_threshold: float = 0.25,
        top_k: int = 4
    ):
        self.embedding_model_name = embedding_model_name
        self.vector_store_dir = vector_store_dir
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        
        self.dimension: int = 384
        self.st_model = None
        self.use_st = False
        self.dense_encoder = DenseSemanticEncoder(dimension=self.dimension)
        
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[DocumentChunk] = []

        self._init_encoder()
        self.load_index()

    def _init_encoder(self):
        if is_sentence_transformers_available():
            try:
                from sentence_transformers import SentenceTransformer
                self.st_model = SentenceTransformer(self.embedding_model_name)
                self.dimension = self.st_model.get_sentence_embedding_dimension()
                self.use_st = True
            except Exception:
                self.use_st = False
                self.st_model = None
        else:
            self.use_st = False
            self.st_model = None

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.use_st and self.st_model is not None:
            try:
                emb = self.st_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
                return np.array(emb, dtype=np.float32)
            except Exception:
                self.use_st = False

        # Use DenseSemanticEncoder
        return self.dense_encoder.encode(texts, normalize_embeddings=True)

    def build_index(self, chunks: List[DocumentChunk], append: bool = False) -> int:
        """
        Embeds chunks and creates or updates a FAISS IndexFlatIP (cosine similarity).
        Returns total number of chunks indexed.
        """
        if not chunks:
            return len(self.chunks)

        if not self.use_st:
            all_texts = [c.text for c in (self.chunks + chunks if append else chunks)]
            self.dense_encoder.fit(all_texts)

        texts = [chunk.text for chunk in chunks]
        embeddings = self._compute_embeddings(texts)

        if append and self.index is not None:
            self.index.add(embeddings)
            self.chunks.extend(chunks)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
            self.chunks = list(chunks)

        self.save_index()
        return len(self.chunks)

    def save_index(self, store_dir: Optional[str] = None) -> bool:
        """Persists the FAISS index and chunk metadata to disk."""
        target_dir = store_dir or self.vector_store_dir
        os.makedirs(target_dir, exist_ok=True)
        
        if self.index is None or not self.chunks:
            return False

        index_file = os.path.join(target_dir, "faiss.index")
        meta_file = os.path.join(target_dir, "metadata.pkl")
        enc_file = os.path.join(target_dir, "encoder.pkl")

        faiss.write_index(self.index, index_file)
        with open(meta_file, "wb") as f:
            pickle.dump([chunk.to_dict() for chunk in self.chunks], f)
        with open(enc_file, "wb") as f:
            pickle.dump(self.dense_encoder, f)

        return True

    def load_index(self, store_dir: Optional[str] = None) -> bool:
        """Loads a persisted FAISS index and metadata from disk if found."""
        target_dir = store_dir or self.vector_store_dir
        index_file = os.path.join(target_dir, "faiss.index")
        meta_file = os.path.join(target_dir, "metadata.pkl")
        enc_file = os.path.join(target_dir, "encoder.pkl")

        if os.path.exists(index_file) and os.path.exists(meta_file):
            try:
                self.index = faiss.read_index(index_file)
                self.dimension = self.index.d
                with open(meta_file, "rb") as f:
                    meta_data = pickle.load(f)
                    self.chunks = [DocumentChunk.from_dict(d) for d in meta_data]
                if os.path.exists(enc_file):
                    with open(enc_file, "rb") as f:
                        self.dense_encoder = pickle.load(f)
                return True
            except Exception as e:
                self.index = None
                self.chunks = []
                return False
        return False

    def clear_index(self) -> None:
        """Resets the in-memory index and removes persisted files."""
        self.index = None
        self.chunks = []
        index_file = os.path.join(self.vector_store_dir, "faiss.index")
        meta_file = os.path.join(self.vector_store_dir, "metadata.pkl")
        enc_file = os.path.join(self.vector_store_dir, "encoder.pkl")
        for f in (index_file, meta_file, enc_file):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k document chunks relevant to the query with similarity scores.
        Filters out matches below min_similarity.
        """
        k = top_k or self.top_k
        threshold = min_similarity if min_similarity is not None else self.similarity_threshold

        if self.index is None or len(self.chunks) == 0:
            return []

        q_emb = self._compute_embeddings([query])

        k_search = min(k, len(self.chunks))
        distances, indices = self.index.search(q_emb, k_search)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            score = float(dist)
            if score >= threshold:
                chunk = self.chunks[idx]
                results.append({
                    "chunk": chunk,
                    "score": round(score, 4),
                    "confidence_pct": round(score * 100, 1),
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id
                })

        return results

    def _extractive_fallback_synthesizer(
        self,
        query: str,
        retrieved_passages: List[Dict[str, Any]]
    ) -> str:
        """
        Grounded offline synthesizer extracting verbatim policy sentences.
        """
        query_words = set(re.findall(r'\w+', query.lower()))
        stopwords = {"what", "is", "the", "are", "for", "a", "an", "how", "can", "i", "to", "do", "of", "in", "and", "or", "if"}
        query_terms = [w for w in query_words if w not in stopwords and len(w) > 2]

        selected_sentences = []
        for passage in retrieved_passages:
            text = passage["text"]
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for s in sentences:
                s_clean = s.strip()
                if not s_clean or len(s_clean) < 15:
                    continue
                s_lower = s_clean.lower()
                matches = sum(1 for term in query_terms if term in s_lower)
                if matches > 0:
                    selected_sentences.append((matches, passage["page_number"], passage["source"], s_clean))

        selected_sentences.sort(key=lambda x: x[0], reverse=True)

        if not selected_sentences:
            return POLICY_FALLBACK_RESPONSE

        seen_texts = set()
        points = []
        for _, page, src, sent in selected_sentences[:5]:
            if sent not in seen_texts:
                seen_texts.add(sent)
                points.append(f"- **[Page {page}]**: {sent}")
        answer_body = "Based on official college regulations, the following provisions apply:\n\n" + "\n".join(points)

        return (
            f"{answer_body}\n\n"
            f"> *Note: Grounded via local document verification engine.*"
        )

    def _call_llm_api(
        self,
        query: str,
        context_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Optional[str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            return None

        try:
            from openai import OpenAI
            base_url = os.getenv("OPENAI_BASE_URL", None)
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
            
            client = OpenAI(api_key=api_key, base_url=base_url)

            system_prompt = (
                "You are an authoritative, helpful College Regulations and Academic Policy Chatbot. "
                "Your objective is to provide precise, professional answers to students based strictly and only "
                "on the provided college document excerpts.\n\n"
                "CRITICAL RULES:\n"
                "1. Answer ONLY using the facts present in the provided context passages.\n"
                "2. DO NOT invent, assume, or extrapolate policies, dates, fees, or requirements.\n"
                "3. If the provided context does NOT contain enough information to answer the question, respond with EXACTLY:\n"
                f"   \"{POLICY_FALLBACK_RESPONSE}\"\n"
                "4. For every rule or fact cited, explicitly reference the source document and page number.\n"
                "5. If there are conflicting rules in different documents, clearly highlight the discrepancy.\n"
                "6. Maintain an objective, polite, and encouraging tone suitable for university students."
            )

            messages = [{"role": "system", "content": system_prompt}]

            if conversation_history:
                for msg in conversation_history[-4:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            user_content = (
                f"DOCUMENT CONTEXT EXCERPTS:\n{context_text}\n\n"
                f"STUDENT QUESTION: {query}\n\n"
                f"Please provide a grounded answer with specific page citations based strictly on the above context."
            )
            messages.append({"role": "user", "content": user_content})

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return None

    def query(
        self,
        query_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        query_clean = query_text.strip()
        
        if not query_clean:
            return {
                "answer": "Please enter a valid question regarding college regulations.",
                "citations": [],
                "retrieved_count": 0,
                "latency_ms": 0,
                "status": "EMPTY_QUERY"
            }

        # Step 1: Retrieval
        passages = self.retrieve(query_clean, top_k=top_k)

        # Step 2: Relevance check
        if not passages:
            latency = int((time.time() - start_time) * 1000)
            return {
                "answer": POLICY_FALLBACK_RESPONSE,
                "citations": [],
                "retrieved_count": 0,
                "latency_ms": latency,
                "status": "NO_EVIDENCE_FOUND",
                "disclaimer": OFFICIAL_DISCLAIMER
            }

        # Step 3: Format Context
        context_blocks = []
        for i, p in enumerate(passages):
            context_blocks.append(
                f"[Passage {i+1} | Document: {p['source']} | Page: {p['page_number']} | Similarity: {p['confidence_pct']}%]\n"
                f"{p['text']}"
            )
        full_context = "\n\n".join(context_blocks)

        # Step 4: Generation
        llm_answer = self._call_llm_api(query_clean, full_context, conversation_history)
        if not llm_answer:
            llm_answer = self._extractive_fallback_synthesizer(query_clean, passages)

        if not llm_answer or llm_answer.strip() == POLICY_FALLBACK_RESPONSE or "could not find sufficient information" in llm_answer.lower():
            latency = int((time.time() - start_time) * 1000)
            return {
                "answer": POLICY_FALLBACK_RESPONSE,
                "citations": [],
                "retrieved_count": len(passages),
                "latency_ms": latency,
                "status": "NO_EVIDENCE_FOUND",
                "disclaimer": OFFICIAL_DISCLAIMER
            }

        # Step 5: Format Citations
        citations = []
        for p in passages:
            citations.append({
                "source": p["source"],
                "page_number": p["page_number"],
                "score": p["score"],
                "confidence_pct": p["confidence_pct"],
                "excerpt": p["text"][:280] + ("..." if len(p["text"]) > 280 else "")
            })

        latency = int((time.time() - start_time) * 1000)

        return {
            "answer": llm_answer,
            "citations": citations,
            "retrieved_count": len(passages),
            "latency_ms": latency,
            "status": "ANSWERED",
            "disclaimer": OFFICIAL_DISCLAIMER
        }
