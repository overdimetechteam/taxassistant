"""
RAG query engine using LangChain and Gemini 2.5 Flash.
Handles query processing, context retrieval, and response generation
with tax-domain-specific prompting.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.config import GOOGLE_API_KEY, GEMINI_MODEL
from app.services.vector_store import get_vector_store

# System prompt designed for tax advisory RAG
TAX_ADVISOR_SYSTEM_PROMPT = """You are an expert Sri Lankan Tax Advisor AI assistant. Your role is to analyze tax scenarios and identify ALL applicable tax implications based on the provided legislative context from Sri Lankan tax acts.

IMPORTANT GUIDELINES:
1. **Identify ALL applicable taxes**: For any given scenario, consider ALL tax types that may apply:
   - Value Added Tax (VAT)
   - Nation Building Tax (NBT)
   - Betting and Gaming Levy (BNG)
   - Income Tax (under Inland Revenue Act / IRA)
   - Social Security Contribution Levy (SSCL)
   - Any other taxes mentioned in the acts

2. **Be vigilant about product classifications**: Products may fall under multiple tax categories.
   For example:
   - Shoes may be taxed as leather products
   - Electronic devices may have different rates based on components
   - Food items may have exemptions or reduced rates
   - Services may be taxed differently from goods

3. **Quote exact rates and sections**: Always reference the specific section, schedule, and rate from the legislation. Include the act name and year.

4. **Consider amendments**: If there are amendments, use the LATEST applicable rate. Note if a rate has been amended and what it changed from.

5. **Flag exemptions and thresholds**: Mention any exemptions, zero-rated items, registration thresholds, or special conditions that apply.

6. **Be comprehensive**: Cover all angles - import duties, local manufacture, wholesale, retail, service provision, etc. as applicable to the scenario.

7. **If uncertain**: Clearly state when you are uncertain or when the provided context does not contain sufficient information to give a definitive answer.

Based on the following legislative context, answer the user's tax query:

---
LEGISLATIVE CONTEXT:
{context}
---
"""

TAX_ADVISOR_HUMAN_PROMPT = """TAX SCENARIO/QUERY:
{question}

Structure your response in EXACTLY this format with the separator line:

## Quick Summary

Provide 3-6 concise bullet points. Each bullet must follow this pattern:
- **Tax Name** - Rate% (Act Name, Section X, effective from DD.MM.YYYY)

Include any key exemptions or thresholds as additional bullets.

===DETAILED===

## Detailed Analysis

Provide a thorough markdown analysis covering:
1. Each applicable tax type in detail
2. The specific rate(s) with legal references (Act, Section, Schedule)
3. Any exemptions or special conditions
4. Effective dates
5. Product/service classifications that affect taxation
6. Any amendments and rate changes

IMPORTANT: You MUST include the exact line "===DETAILED===" between the summary and the detailed analysis. Do NOT skip it."""


def get_llm() -> ChatGoogleGenerativeAI:
    """Get configured Gemini 2.5 Flash LLM instance."""
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,  # Low temperature for factual accuracy
        max_output_tokens=15000,
    )


def format_docs(docs) -> str:
    """Format retrieved documents into context string with metadata."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = (
            f"[Source {i}] "
            f"Act: {meta.get('act_name', 'Unknown')} | "
            f"Tax Type: {meta.get('tax_type', 'Unknown')} | "
            f"Year: {meta.get('year', 'N/A')} | "
            f"Section: {meta.get('section', 'N/A')} | "
            f"Page: {meta.get('page_number', 'N/A')}"
        )
        if meta.get("tax_rates_mentioned"):
            header += f" | Rates found: {meta['tax_rates_mentioned']}%"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain():
    """
    Build the RAG chain: Retriever -> Prompt -> LLM -> Parser.
    Uses multi-query retrieval for better coverage of tax implications.
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15},  # Retrieve more docs for comprehensive tax coverage
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", TAX_ADVISOR_SYSTEM_PROMPT),
        ("human", TAX_ADVISOR_HUMAN_PROMPT),
    ])

    llm = get_llm()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def _parse_llm_response(raw: str) -> tuple[str, str]:
    """Split LLM response on ===DETAILED=== delimiter into (summary, detailed_analysis)."""
    delimiter = "===DETAILED==="

    if delimiter in raw:
        parts = raw.split(delimiter, 1)
        summary = parts[0].strip()
        detailed = parts[1].strip()
        if summary and detailed:
            return summary, detailed

    # Fallback: treat entire response as detailed, no summary
    return "", raw.strip()


def query_tax_advisor(query: str, top_k: int = 15) -> dict:
    """
    Process a tax query through the RAG pipeline.

    Returns:
        Dictionary with answer, sources, and identified tax types.
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)

    # Format context
    context = format_docs(retrieved_docs)

    # Build and invoke the LLM chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", TAX_ADVISOR_SYSTEM_PROMPT),
        ("human", TAX_ADVISOR_HUMAN_PROMPT),
    ])

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    raw_answer = chain.invoke({"context": context, "question": query})

    # Parse structured JSON response from LLM
    summary, detailed_analysis = _parse_llm_response(raw_answer)

    # Extract unique tax types from retrieved documents
    tax_types = list({
        doc.metadata.get("tax_type", "Unknown")
        for doc in retrieved_docs
        if doc.metadata.get("tax_type")
    })

    # Build source references
    sources = []
    seen = set()
    for doc in retrieved_docs:
        meta = doc.metadata
        key = (meta.get("act_name"), meta.get("section"), meta.get("page_number"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "act_name": meta.get("act_name", "Unknown"),
                "tax_type": meta.get("tax_type", "Unknown"),
                "year": meta.get("year"),
                "act_number": meta.get("act_number", ""),
                "section": meta.get("section", ""),
                "page": meta.get("page_number"),
                "content_preview": doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
            })

    return {
        "query": query,
        "summary": summary,
        "detailed_analysis": detailed_analysis,
        "sources": sources,
        "tax_types_identified": sorted(tax_types),
    }
