"""
Knowledge graph layer.

Full Neo4j requires a running server (Docker), which is heavier than a
non-gaming laptop needs for a demo/portfolio project. By default this
module builds an in-memory/on-disk graph with networkx, using
regex-based extraction tuned for legal documents (parties, defined
terms, section/clause references, dates, monetary amounts, citations).

If you DO want real Neo4j (e.g. to show Cypher queries in a demo), set
USE_NEO4J=true and fill in NEO4J_* in .env - see `neo4j_adapter()` below.
Everything else in the pipeline works identically either way.
"""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx

from .config import config
from .document_processor import Chunk

# ---- Legal-domain regex patterns (no spaCy model download required) ----
SECTION_RE = re.compile(r"\b(Section|Clause|Article)\s+\d+(?:\.\d+)*[a-zA-Z]?\b", re.I)
DEFINED_TERM_RE = re.compile(r'"([A-Z][A-Za-z\s]{2,40})"')
PARTY_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\s+"
                       r"(?:Inc\.|LLC|LLP|Ltd\.|Corp\.|Corporation|Company))(?=\s|,|\.|$)")
DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b"
)
MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")
CASE_CITATION_RE = re.compile(r"\b[A-Z][a-zA-Z.]+\s+v\.\s+[A-Z][a-zA-Z.]+\b")


def extract_entities(text: str) -> Dict[str, List[str]]:
    return {
        "sections": list(set(SECTION_RE.findall(text))),
        "defined_terms": list(set(DEFINED_TERM_RE.findall(text))),
        "parties": list(set(PARTY_RE.findall(text))),
        "dates": list(set(DATE_RE.findall(text))),
        "amounts": list(set(MONEY_RE.findall(text))),
        "case_citations": list(set(CASE_CITATION_RE.findall(text))),
    }


class KnowledgeGraph:
    """
    Nodes: chunk, document, party, defined_term, section, case_citation
    Edges: (chunk)-[MENTIONS]->(entity), (chunk)-[BELONGS_TO]->(document),
           (document)-[NEXT_CHUNK]->(chunk) for ordering.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def build(self, chunks: List[Chunk]):
        for c in chunks:
            chunk_node = f"chunk:{c.chunk_id}"
            doc_node = f"doc:{c.doc_id}"

            self.graph.add_node(doc_node, type="document", name=c.doc_name)
            self.graph.add_node(chunk_node, type="chunk", text=c.text[:300],
                                 page=c.page, doc_name=c.doc_name)
            self.graph.add_edge(chunk_node, doc_node, relation="BELONGS_TO")

            entities = extract_entities(c.text)
            for kind, values in entities.items():
                for v in values:
                    v = v.strip()
                    if not v:
                        continue
                    ent_node = f"{kind}:{v.lower()}"
                    if not self.graph.has_node(ent_node):
                        self.graph.add_node(ent_node, type=kind, name=v)
                    self.graph.add_edge(chunk_node, ent_node, relation="MENTIONS")

        self._save()
        print(f"[knowledge_graph] built graph: "
              f"{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _save(self):
        Path(config.KG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(config.KG_PATH, "wb") as f:
            pickle.dump(self.graph, f)

    def load(self) -> bool:
        p = Path(config.KG_PATH)
        if not p.exists():
            return False
        with open(p, "rb") as f:
            self.graph = pickle.load(f)
        return True

    def related_chunks_for_entity(self, entity_text: str, kinds: Tuple[str, ...] = (
        "parties", "defined_terms", "sections", "case_citations"
    )) -> List[str]:
        """Given free text (e.g. a query), find entity nodes it mentions and
        return chunk_ids linked to those same entities elsewhere in the corpus.
        This lets us pull in context that keyword/vector search might miss,
        e.g. 'what does the Agreement say elsewhere about Acme Corp?'"""
        entity_lower = entity_text.lower()
        matched_chunks = set()
        for node, data in self.graph.nodes(data=True):
            if data.get("type") in kinds and data.get("name", "").lower() in entity_lower:
                for pred in self.graph.predecessors(node):
                    if self.graph.nodes[pred].get("type") == "chunk":
                        matched_chunks.add(pred.replace("chunk:", ""))
        return list(matched_chunks)

    def stats(self) -> Dict:
        by_type = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        return {"nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "by_type": by_type}


def neo4j_adapter():
    """
    Optional real-Neo4j adapter, only used if USE_NEO4J=true in .env and
    the `neo4j` package + a running Neo4j instance are available.
    Kept separate so the default laptop path never imports neo4j.
    """
    if os.environ.get("USE_NEO4J", "false").lower() != "true":
        return None
    from neo4j import GraphDatabase  # local import: optional dependency

    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    return GraphDatabase.driver(uri, auth=(user, password))
