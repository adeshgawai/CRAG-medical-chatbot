from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List

import re

from src.llm import load_llm
from src.prompt import doc_eval_system_prompt, prompt_to_llm
from src.retriever import return_retriever



llm = load_llm()

UPPER_TH = 0.7
LOWER_TH = 0.3

def retrieve(state):
    q = state['question']
    retriever = return_retriever()
    return {"docs": retriever.invoke(q)}



# -----------------------------
# Score-based listwise doc evaluator
# -----------------------------
class SingleDocEval(BaseModel):
    score: float = Field(description="Relevance score for the chunk in [0.0, 1.0].")
    reason: str = Field(description="Short reason for the score.")

class AllDocsEval(BaseModel):
    evaluations: List[SingleDocEval] = Field(description="Evaluations for each chunk in the exact order they are provided.")

doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given a question and a list of retrieved chunks.\n"
            "For each chunk, evaluate and assign a relevance score in [0.0, 1.0] and provide a short reason.\n"
            "You must output structured evaluations adhering strictly to the schema.",
        ),
        ("human", "Question: {question}\n\nChunks:\n{chunks_text}"),
    ]
)

all_docs_eval_chain = doc_eval_prompt | llm.with_structured_output(AllDocsEval)

def eval_each_doc(state):
    q = state['question']
    docs = state['docs']

    if not docs:
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": "No documents retrieved.",
        }

    # Format the chunks for a single prompt
    chunks_text = ""
    for i, d in enumerate(docs):
        chunks_text += f"--- Chunk {i+1} ---\n{d.page_content}\n\n"

    try:
        out = all_docs_eval_chain.invoke({"question": q, "chunks_text": chunks_text})
        evals = out.evaluations
    except Exception as e:
        print(f"Error evaluating documents: {e}. Defaulting to empty scores.")
        evals = []

    scores: List[float] = []
    good: List[Document] = []
    reasons: List[str] = []

    for i, d in enumerate(docs):
        eval_obj = evals[i] if i < len(evals) else SingleDocEval(score=0.0, reason="Missing evaluation")
        scores.append(eval_obj.score)
        reasons.append(eval_obj.reason)
        if eval_obj.score > LOWER_TH:
            good.append(d)
    
    if any(s > UPPER_TH for s in scores):
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
        }
    
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        why = "No chunk was sufficient."
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {LOWER_TH}. {why}",
        }
    
    why = "Mixed relevance signals."
    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}. {why}",
    }


# ---------------------------------------------------------------------------
# -----------------------------
# Block-level Context Refiner
# -----------------------------
refine_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a relevance filter for a medical RAG system.\n"
            "Your task is to take a context block and extract ONLY the sentences/information relevant to answering the question.\n"
            "Remove all navigation links, disclaimers, advertisements, html tags, and completely unrelated noise.\n"
            "Do not summarize or synthesize new information; just output the cleaned relevant sentences from the context verbatim or near-verbatim.\n"
            "If nothing in the context is relevant, return an empty string.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

refine_chain = refine_prompt | llm

def refine(state):
    q = state['question']

    # Combine retrieved docs into one context string
    if state.get("verdict") == "CORRECT":
        docs_to_use = state["good_docs"]
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state["web_docs"]
    else:  # AMBIGUOUS
        docs_to_use = state["good_docs"] + state["web_docs"]

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    
    if not context:
        return {
            "strips": [],
            "kept_strips": [],
            "refined_context": ""
        }

    try:
        out = refine_chain.invoke({"question": q, "context": context})
        refined_context = out.content.strip()
    except Exception as e:
        print(f"Error refining context: {e}. Defaulting to original context.")
        refined_context = context

    # For state compatibility
    strips = [context]
    kept = [refined_context]

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context
    }


# -----------------------------------------------------------------------------
# -----------------------------
# Rewriting thw query
#
# -----------------------------
class WebQuery(BaseModel):
    """Search query for web search."""
    # Adding a description helps the LLM understand it's a tool to be called
    query: str = Field(description="The optimized search query keywords.")

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Question: {question}"),
    ]
)

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery, method='json_mode')

def rewrite_query_node(state):
    out = rewrite_chain.invoke({'question': state['question']})
    return {'web_query': out.query}

# -----------------------------
# Web search (Iteration 4)
# Assumption: web search does not fail (no fail node in this branch)
# -----------------------------
tavily = TavilySearchResults(max_results=2)

def web_search_node(state):

    q = state["web_query"]  # no query rewrite
    results = tavily.invoke({"query": q})  # no knowledge selection

    web_docs = []
    for r in results or []:

        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"

        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

    return {"web_docs": web_docs}

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             prompt_to_llm(),
#         ),
#         ("human", "Question: {question}\n\nRefined context:\n{refined_context}"),
#     ]
# )
def generate(state):
    # context = "\n\n".join(d.page_content for d in state["docs"])
    medical_prompt = prompt_to_llm()
    refined = state["refined_context"].strip()
    if not refined:
        if state.get("web_docs"):
            refined = "\n\n".join(d.page_content for d in state["web_docs"])
        elif state.get("good_docs"):
            refined = "\n\n".join(d.page_content for d in state["good_docs"])
    out = (medical_prompt | llm).invoke({"question": state["question"], "refined_context": refined})
    return {"answer": out.content}

# -----------------------------------------------------------------------------

def fail_node(state):
    return {"answer": f"Fail: {state['reason']}"}

# def ambiguous_node(state):
#     return {"answer": f"Ambiguous: {state['reason']}"}

# -----------------------------
# Routing
# CORRECT => refine
# INCORRECT / AMBIGUOUS => rewrite -> web_search -> refine -> generate
# -----------------------------
def route_after_eval(state) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    else:
        return "rewrite_query"


# -----------------------------
# Router
# -----------------------------
class RouteQuery(BaseModel):
    """Route user question to 'medical_rag' or 'general_chat'."""
    datasource: str = Field(
        description="Choose 'general_chat' if the query is a greeting or polite conversation. Choose 'medical_rag' if it requires medical database retrieval."
    )

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert router. Determine if the question is a greeting/general chat or a medical query.\n"
            "If it is a greeting (like hello, hi, how are you) or general polite chit-chat, route to 'general_chat'.\n"
            "If it asks for medical information, symptoms, treatments, or anything health-related, route to 'medical_rag'.\n"
            "Output JSON with a single key: datasource."
        ),
        ("human", "{question}"),
    ]
)

router_chain = router_prompt | llm.with_structured_output(RouteQuery, method='json_mode')

def route_question(state) -> str:
    q = state["question"]
    try:
        out = router_chain.invoke({"question": q})
        if out.datasource == "general_chat":
            return "general_chat"
    except Exception as e:
        print(f"Router error: {e}. Defaulting to medical_rag.")
    return "medical_rag"