from langgraph.graph import StateGraph, START, END
from src.state import State
from src.nodes import *


def build_graph():
    # -----------------------------
    # Graph
    # -----------------------------
    g = StateGraph(State)

    g.add_node("retrieve", retrieve)
    g.add_node("eval_each_doc", eval_each_doc)

    g.add_node('rewrite_query', rewrite_query_node)
    g.add_node("web_search", web_search_node)
    g.add_node("refine", refine)          # uses verdict to pick good_docs vs web_docs
    g.add_node("generate", generate)
    # g.add_node("ambiguous", ambiguous_node)

    # flow
    g.add_conditional_edges(
        START,
        route_question,
        {
            "general_chat": "generate",
            "medical_rag": "retrieve"
        }
    )
    g.add_edge("retrieve", "eval_each_doc")

    # route after evaluation
    g.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {
            "refine": "refine",          # CORRECT -> refine (good_docs)
            "rewrite_query": "rewrite_query",  # INCORRECT -> web_search
            # "web_search": "web_search"
        },
    )

    g.add_edge("rewrite_query", "web_search")
    # ✅ key change: web_search now goes through refine
    g.add_edge("web_search", "refine")    # INCORRECT -> refine (web_docs)
    g.add_edge("refine", "generate")      # CORRECT/INCORRECT -> generate

    g.add_edge("generate", END)
    # g.add_edge("ambiguous", END)

    app = g.compile()
    app

    return app