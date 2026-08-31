"""Local, evidence-grounded RAG for EasyCite."""

from .models import RagDocument, RagRetrievalResult, RagRetrievalStats
from .retriever import RagRetriever

__all__ = ["RagDocument", "RagRetrievalResult", "RagRetrievalStats", "RagRetriever"]
