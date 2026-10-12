"""
LLM Generation and Citation Synthesis Engine.
Compatible with OpenAI, Groq, Ollama, and local OpenAI-compatible endpoints.
Supports synchronous generation and real-time streaming with exact document citations.
"""

import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from app.core.reranker import RerankedContext

logger = logging.getLogger("rag_enterprise.generator")

SYSTEM_PROMPT = """You are an Enterprise Principal AI Research Scientist.
Your objective is to answer the user's question accurately using ONLY the provided authoritative document contexts.

Guidelines for Scientific, Mathematical, and Tabular Analysis:
1. Exact Citations: When referencing facts, numbers, equations, or tables, cite the exact source using [Page <P>, Doc: <filename>].
2. Tabular Precision: If tabular data is involved, preserve and quote exact rows/columns or present findings in structured Markdown tables.
3. Mathematical Precision: Keep mathematical formulas, variables, and units intact using standard LaTeX notation ($...$ for inline, $$...$$ for block).
4. Evidence-Grounded: Do not hallucinate or extrapolate beyond what is stated in the contexts. If the context does not contain enough evidence to answer, state clearly: "Based on the provided documents, there is insufficient evidence to determine..."
5. Structure: Provide a clear, well-structured response with Markdown headings, bullet points, and an explicit 'Key Evidence & Sources' section.
"""


class LLMGenerator:
    """
    LLM generation client supporting OpenAI, Groq, Ollama, and local endpoints.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1500
    ):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or "EMPTY"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._async_client = None
        self._sync_client = None

    def _get_async_client(self):
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._async_client

    def _get_sync_client(self):
        if self._sync_client is None:
            from openai import OpenAI
            self._sync_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._sync_client

    def _build_context_prompt(self, contexts: List[RerankedContext]) -> str:
        """Formats top retrieved contexts into numbered authoritative references."""
        context_blocks = []
        for i, ctx in enumerate(contexts, start=1):
            doc_name = ctx.metadata.get("filename", "Unknown")
            page_num = ctx.metadata.get("page_number", 1)
            score = round(ctx.rerank_score, 4)
            block = (
                f"--- CONTEXT BLOCK {i} ---\n"
                f"Source: {doc_name} | Page: {page_num} | Confidence Score: {score}\n\n"
                f"{ctx.content}\n"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def _create_mock_synthesis(self, query: str, contexts: List[RerankedContext]) -> str:
        """
        Deterministic synthesis fallback if no active API key is provided,
        ensuring the application is fully functional offline/in dev mode.
        """
        if not contexts:
            return "No relevant contexts found in the indexed documents to answer your question."

        top_ctx = contexts[0]
        meta = top_ctx.metadata
        summary_lines = [
            f"### Synthesized Response (Local Fallback Mode)\n",
            f"Based on **{meta.get('filename')}** (Page {meta.get('page_number')}, Confidence: {round(top_ctx.rerank_score, 3)}):\n\n",
            f"> {top_ctx.content[:350]}...\n\n",
            "#### Key Evidence Extracted:",
        ]
        for idx, ctx in enumerate(contexts[:3], start=1):
            m = ctx.metadata
            summary_lines.append(
                f"- **Context {idx}** [Page {m.get('page_number')}, {m.get('filename')}]: "
                f"Found relevant discussion with relevance score {round(ctx.rerank_score, 3)}."
            )

        summary_lines.append("\n*Note: Configure OPENAI_API_KEY or GROQ_API_KEY in .env for full LLM generative reasoning.*")
        return "\n".join(summary_lines)

    def generate(
        self,
        query: str,
        contexts: List[RerankedContext]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes synchronous generation returning answer and structured citations.
        """
        citations = [
            {
                "document_id": ctx.metadata.get("document_id"),
                "filename": ctx.metadata.get("filename"),
                "page_number": ctx.metadata.get("page_number"),
                "chunk_id": ctx.chunk_id,
                "relevance_score": round(ctx.rerank_score, 4),
                "snippet": ctx.content[:300] + ("..." if len(ctx.content) > 300 else "")
            }
            for ctx in contexts
        ]

        if not self.api_key or self.api_key == "EMPTY" or "your_" in self.api_key:
            return self._create_mock_synthesis(query, contexts), citations

        context_str = self._build_context_prompt(contexts)
        user_message = (
            f"Authoritative Contexts:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            f"Please provide an accurate, cited response based strictly on the above contexts."
        )

        try:
            client = self._get_sync_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            answer = response.choices[0].message.content
            return answer, citations
        except Exception as e:
            logger.error(f"LLM generation failed: {e}. Utilizing fallback synthesis.")
            return self._create_mock_synthesis(query, contexts), citations

    async def stream_generate(
        self,
        query: str,
        contexts: List[RerankedContext]
    ) -> AsyncGenerator[str, None]:
        """
        Streams generated tokens asynchronously.
        """
        if not self.api_key or self.api_key == "EMPTY" or "your_" in self.api_key:
            fallback = self._create_mock_synthesis(query, contexts)
            words = fallback.split(" ")
            for w in words:
                yield w + " "
            return

        context_str = self._build_context_prompt(contexts)
        user_message = (
            f"Authoritative Contexts:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            f"Please provide an accurate, cited response based strictly on the above contexts."
        )

        try:
            client = self._get_async_client()
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Streaming LLM failed: {e}")
            yield f"\n[Error streaming from LLM provider: {str(e)}]\n"
