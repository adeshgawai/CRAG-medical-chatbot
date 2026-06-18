from typing import TypedDict, List
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    docs: List[Document]

    # states for retrieval evaluator
    good_docs: List[Document]
    verdict: str
    reason: str

    # more states added for refinement
    # strips are one statement in paragraph
    strips: List[str] # output of decomposition (sentence strips)
    kept_strips: List[str] # strips after filtering
    refined_context: str #recomposed knowledge

    web_docs: List[Document]

    web_query: str #here we are rewriting the query
    
    answer: str