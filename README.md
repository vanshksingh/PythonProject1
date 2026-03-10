# THREAD-RAG

### Traversal-Heuristic Retrieval for Embedded And Distributed Retrieval-Augmented Generation

THREAD-RAG is a **retrieval architecture for long-document reasoning** that enables LLM agents to navigate documents like structured threads rather than isolated chunks.

Instead of retrieving disconnected fragments, THREAD-RAG allows an agent to:

1. **Jump** to a relevant location using semantic search
2. **Walk** through the document sequentially
3. **Reconstruct context and reasoning paths**

This approach significantly improves coherence when working with **long technical documents, policies, manuals, or research papers.**

---

# Overview

Traditional Retrieval-Augmented Generation systems treat documents as independent chunks.

```
query → top-k chunks → answer
```

THREAD-RAG introduces **thread-aware retrieval**.

```
query → semantic jump → thread entry
      → sequential traversal
      → optional cross-thread comparison
      → answer
```

Instead of retrieving static chunks, the system **navigates document structure.**

---

# Core Idea

Documents are modeled as **ordered semantic threads**.

Each document is converted into a sequential chain of chunks:

```
DOC1_000
DOC1_001
DOC1_002
DOC1_003
```

Each chunk contains:

```json
{
  "chunk_id": "DOC1_001",
  "doc_id": "DOC1",
  "chunk_content": "text of the chunk"
}
```

This ordered structure allows the system to **traverse documents like a timeline.**

---

# Context Window Structure

Each retrieved chunk is presented to the model with **context stitching**.

```
PREVIOUS SUMMARY
ACTIVE CONTENT
NEXT SUMMARY
```

Example:

```
--- CONTEXT WINDOW: DOC1_009 ---

PREVIOUS SECTION SUMMARY
(summary of DOC1_008)

ACTIVE CONTENT
(actual text of DOC1_009)

FOLLOWING SECTION SUMMARY
(summary of DOC1_010)

--- END WINDOW ---
```

This helps the model understand **where the chunk sits in the document narrative.**

---

# Why Context Stitching Matters

Traditional chunking fragments information across boundaries.

THREAD-RAG restores continuity by attaching summaries of neighboring sections.

Benefits:

* better reasoning across chunk boundaries
* improved narrative coherence
* reduced hallucination risk
* stronger long-document understanding

---

# Retrieval Architecture

THREAD-RAG operates using two modes.

---

## 1. Jump Retrieval (Dart Mode)

Semantic search is used to find an entry point in the thread.

```
query → vector search → starting chunk
```

Example:

```
rag_search(query)
```

---

## 2. Sequential Traversal (Walk Mode)

Once inside a thread, the system can walk through adjacent chunks.

```
chunk_i → chunk_(i+1) → chunk_(i+2)
```

Traversal continues when summaries indicate the topic continues.

---

# Hybrid Retrieval Strategy

THREAD-RAG naturally supports patterns such as:

```
jump → walk → walk → jump
```

Example workflow:

```
rag_search(query)

fetch_chunks_by_id(DOC1_021)

fetch_chunks_by_id(DOC1_022)

rag_search(refined_query)
```

This enables **adaptive exploration of documents.**

---

# Retrieval vs Navigation Separation

THREAD-RAG explicitly separates **retrieval** from **navigation**.

### Retrieval Layer

Purpose: locate entry points.

Method:

```
vector search on chunk text
```

---

### Navigation Layer

Purpose: explore document structure.

Method:

```
sequential traversal across chunk threads
```

This separation prevents **embedding duplication and retrieval collapse.**

---

# Avoiding Top-K Poisoning

If contextual summaries are embedded into every chunk, adjacent chunks share identical semantic signals.

This causes **vector search clustering**, where neighboring chunks dominate the top-k results.

THREAD-RAG prevents this by:

* embedding **only the chunk text**
* using summaries **only for navigation**

This preserves **retrieval diversity.**

---

# Document Catalog System

THREAD-RAG includes a **document discovery layer**.

Example catalog:

```
DOC1 : project guidelines
DOC2 : grading policy
DOC3 : thesis regulations
```

Agents can first list available documents:

```
list_available_documents()
```

Then restrict retrieval scope.

---

# Multi-Document Thread Traversal

THREAD-RAG supports reasoning across documents.

Example:

```
DOC1_020 → evaluation rules
DOC2_015 → grading policy
```

The agent retrieves both chunks and compares them during reasoning.

---

# Offline Summary Pre-Heating

Chunk summaries are **pre-computed during ingestion** using a smaller model.

Pipeline:

```
document
   ↓
chunking
   ↓
summary generation
   ↓
embedding creation
   ↓
vector indexing
```

Benefits:

* lower runtime cost
* faster responses
* better traversal signals

---

# Cost Optimization Strategy

### Ingestion Phase

```
document chunking
summary generation
embedding creation
vector indexing
```

### Query Phase

```
vector search
thread traversal
LLM reasoning
```

Most heavy computation happens **offline**, reducing runtime latency.

---

# Thread Traversal Signals

Traversal decisions rely on:

* previous section summary
* next section summary
* semantic continuity

The agent decides whether to:

```
continue traversal
stop traversal
jump to another location
```

---

# Retrieval Diversity Preservation

Because embeddings include **only chunk text**, retrieval remains diverse.

Instead of returning many neighboring chunks, search results can include:

```
DOC1_023
DOC4_112
DOC2_045
DOC7_009
```

This increases **coverage across documents.**

---

# System Capabilities

THREAD-RAG enables:

* sequential document reading
* policy comparison
* long-range dependency tracing
* narrative reasoning
* cross-document analysis

---

# Comparison With Other RAG Approaches

### Vanilla RAG

```
query → top-k chunks → answer
```

Problem:

* fragmented context

THREAD-RAG advantage:

* retrieval + sequential traversal

---

### Parent Document Retrieval

Returns large sections.

Problem:

* high token cost

THREAD-RAG advantage:

* targeted traversal with smaller windows

---

### Contextual Retrieval

Uses document-level summaries.

Problem:

* weak local continuity

THREAD-RAG advantage:

* localized sequential context

---

### GraphRAG

Uses knowledge graphs.

Problem:

* heavy preprocessing

THREAD-RAG advantage:

* lightweight ordered structure

---

# Architectural Principles

THREAD-RAG follows these principles:

* structured document representation
* retrieval diversity
* guided navigation
* offline preprocessing
* agent-driven reasoning

---

# Typical Agent Workflow

```
1. list_available_documents
2. rag_search(query)
3. fetch_chunks_by_id(start_chunk)
4. sequential traversal
5. answer synthesis
```

---

# Advantages

THREAD-RAG provides:

* improved contextual coherence
* efficient token usage
* stronger reasoning across long documents
* reduced runtime cost
* flexible retrieval patterns

---

# Ideal Use Cases

THREAD-RAG works best for **long structured documents**:

* technical manuals
* legal documents
* academic papers
* compliance policies
* procedural documentation
* version comparisons

---

# Conceptual Model

```
Document
   ↓
Chunk Thread
   ↓
Semantic Entry Point
   ↓
Thread Traversal
   ↓
Answer Generation
```

---

# Summary

THREAD-RAG is a **thread-aware retrieval architecture** that allows LLM agents to enter, traverse, and reason over structured documents.

By combining:

* semantic search
* sequential traversal
* contextual stitching

THREAD-RAG preserves narrative continuity while maintaining efficient retrieval.
