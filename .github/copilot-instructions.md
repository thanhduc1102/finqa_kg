# Copilot Instructions for FinQA Knowledge Graph Project

## Project Architecture

This is a **modern Knowledge Graph implementation** for FinQA financial documents using advanced NLP. The system transforms financial reports (with tables, text, QA pairs) into queryable graph structures using transformers and semantic embeddings.

**Key Design**: Async-first architecture with GPU-accelerated NLP pipelines (Spacy, HuggingFace transformers, sentence-transformers).

### Package Structure

```
finqa_kg/src/
  ├── pipeline/         # 🆕 NEW: Single-sample processing (PREFERRED)
  │   ├── single_sample_processor.py  # Base: Per-sample mini KG + execution
  │   ├── advanced_processor.py       # Enhanced: Program synthesis from questions
  │   └── batch_processor.py          # Batch: Process multiple samples with stats
  ├── builder/          # Legacy: Full dataset KG construction
  │   ├── knowledge_graph_builder.py  # Main: ModernFinQAKnowledgeGraph
  │   ├── entity_extractor.py         # Multi-model entity extraction
  │   ├── relation_extractor.py       # Zero-shot relation classification
  │   └── text_processor.py           # Text segmentation & preprocessing
  ├── query/            # Legacy: Query engine with semantic search
  │   ├── knowledge_graph_query.py    # Main: ModernFinQAGraphQuery
  │   └── query_utils.py
  └── visualization/    # Interactive graph visualization
      └── graph_visualizer.py         # Plotly/Matplotlib outputs
```

**Architecture Decision**: The project has **two approaches** - use the pipeline approach for new work:

- **NEW Pipeline** (`src/pipeline/`): Processes individual samples with mini KGs (10MB, 0.1s/sample) ✅
- **Legacy System** (`src/builder/` + `src/query/`): Builds monolithic KG from entire dataset (10GB, 30-60 min) ⚠️

## Critical Workflows

### 🆕 1. NEW: Single Sample Processing (Recommended)

```python
from finqa_kg.src.pipeline import AdvancedSampleProcessor

async def process():
    # Load sample from FinQA dataset
    sample = {
        "id": "doc_123",
        "pre_text": ["Revenue: $637B"],
        "table": [[...]],
        "qa": {"question": "...", "program": "...", "exe_ans": 127.4}
    }

    processor = AdvancedSampleProcessor()
    result = await processor.process_sample(sample)

    # Access results
    print(f"Answer: {result.final_answer}")
    print(f"Correct: {result.is_correct}")
    print(f"Steps: {len(result.steps)}")
    print(result.explanation)  # Detailed explanation with KG evidence
```

**Pattern**: Each sample gets its own mini KG (10MB) → 1000x more memory efficient than legacy approach. Use `AdvancedSampleProcessor` when you need program synthesis from questions without provided programs. Use `SingleSampleProcessor` when programs are provided.

**Key Classes**:

- `SingleSampleProcessor`: Base processor with KG building + program execution
- `AdvancedSampleProcessor`: Extends base with question intent analysis + program synthesis
- `BatchProcessor`: Process multiple samples with accuracy statistics

### 2. Legacy: Building Full Dataset KG (Memory-Intensive)

```python
from finqa_kg.src.builder import ModernFinQAKnowledgeGraph

async def build():
    kg = ModernFinQAKnowledgeGraph()  # Auto-detects GPU
    await kg.build_from_json('path/to/data.json', max_samples=100)
    # Graph stored in kg.graph (NetworkX MultiDiGraph)
```

**Pattern**: All document processing methods are `async` - use `await` and run with `asyncio.run()`. The builder automatically initializes:

- Spacy `en_core_web_trf` for NLP
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- `dslim/bert-base-NER` for named entity recognition
- GPU utilization if available

⚠️ **Use legacy approach only when**: You need cross-document semantic search or relationship analysis across the entire dataset.

### 3. Legacy: Querying the Full Graph (Mixed Sync/Async)

```python
from finqa_kg.src.query import ModernFinQAGraphQuery

query = ModernFinQAGraphQuery(kg.graph)  # Takes NetworkX graph

# Semantic search (sync) - returns QueryResult dataclass
results = query.semantic_search("revenue growth", k=5)

# Question answering (async) - uses transformer QA model
answer = await query.answer_question("What was Q3 revenue?")

# Numerical analysis (sync)
trends = query.analyze_numerical_trends()
```

**Key Methods**: `semantic_search()`, `answer_question()`, `get_entity_context()`, `find_related_numbers()`, `trace_calculation_path()`, `analyze_numerical_trends()`, `get_table_context()`.

### 4. Testing & Validation

**CRITICAL**: Install dependencies in correct order to avoid binary incompatibility:

```bash
# Linux/WSL (recommended approach)
cd finqa_kg
chmod +x fix_environment.sh
./fix_environment.sh

# Windows PowerShell
cd finqa_kg
.\fix_environment.ps1

# Manual installation (if scripts don't work)
pip install numpy==1.24.3 scipy==1.10.1  # MUST install numpy first!
pip install spacy==3.7.2 spacy-transformers==1.3.4
python -m spacy download en_core_web_trf  # 320MB download
pip install -r requirements.txt
```

**Run tests:**

```bash
cd finqa_kg
python tests/quick_test.py              # Quick pipeline test (30s)
python tests/test_pipeline.py           # Pipeline functionality
python tests/test_basic.py              # Legacy: Basic functionality
pytest tests/test_basic.py -v           # With pytest
pytest tests/test_advanced.py -v        # Legacy: Advanced NLP features
pytest tests/test_performance.py -v     # Benchmarking
```

**Demo scripts:**

```bash
python examples/demo_advanced_pipeline.py  # NEW: Single sample demo
python examples/demo.py                    # Legacy: Full KG demo
```

**Test Pattern**: Tests use `test_data.json` (4-line financial doc) and verify async operations with `asyncio.run()`. Pipeline tests process individual samples in <1 second.

## FinQA Data Format (Critical)

Input JSON must follow FinQA schema:

```json
{
  "id": "doc_123",
  "filename": "report.pdf",
  "pre_text": ["Text before table..."],    // List of strings
  "post_text": ["Text after table..."],
  "table": [["Header1", "Header2"], [...]], // 2D array
  "qa": {"question": "...", "answer": "..."} // Optional
}
```

**Location**: Real data in `FinQA/dataset/{train,dev,test}.json` (thousands of documents). Test data in `finqa_kg/tests/test_data.json`.

## Code Conventions

### Entity Extraction Pattern

The system uses **entity deduplication by normalized text**:

```python
# In knowledge_graph_builder.py
async def _add_entity_node(self, entity: EntityMention) -> str:
    key = entity.text.strip().lower()  # Normalize
    if key in self.entity_index:
        return self.entity_index[key]  # Reuse existing
    # ... create new node
```

**Why**: Prevents duplicate entities like "Revenue" and "revenue". Index stored in `self.entity_index` dict.

### Node ID Convention

All nodes get unique IDs: `{TYPE}_{counter}` (e.g., `ENTITY_42`, `TEXT_7`, `QA_1`). Use `_create_node_id(prefix)` method - never create IDs manually.

### Graph Edge Types

- `has_pre_text`, `has_post_text`, `has_table`, `has_qa` - Document → Components
- `contains_entity` - Text → Entities (with start/end positions)
- `has_cell` - Table → Cells (with row/col)
- `semantic_link` - Text ↔ Text (similarity > 0.7)
- `relates_to` - Entity relationships from NLP

### Embedding Storage

Text nodes store embeddings as **lists** (JSON-serializable):

```python
embedding=self._get_text_embedding(text).tolist()  # numpy → list
```

Query engine rebuilds numpy arrays from lists when loaded.

## GPU Optimization

The system auto-detects CUDA:

```python
ModernFinQAKnowledgeGraph(use_gpu=True)  # Default: torch.cuda.is_available()
```

**Performance**: On GPU, processes ~50 docs/min. On CPU, ~5 docs/min. All transformer models move to device automatically.

## Common Pitfalls & Solutions

### 1. **Numpy Binary Incompatibility** (MOST COMMON)

**Error:** `ValueError: numpy.dtype size changed, may indicate binary incompatibility`

**Fix:** Run `./fix_environment.sh` (Linux/WSL) or `.\fix_environment.ps1` (Windows). This reinstalls packages in correct order: numpy → spacy → others.

**Why:** Spacy/thinc compiled against different numpy version. ALWAYS install numpy before spacy.

### 2. **Forgetting `await`**

All builder methods are async. Missing `await` returns a coroutine, not data.

```python
# Wrong
kg.build_from_json(path)  # Returns coroutine

# Correct
await kg.build_from_json(path)
```

### 3. **Spacy model not installed**

`en_core_web_trf` must be downloaded separately (320MB). Fails silently → empty entities.

```bash
python -m spacy download en_core_web_trf
```

### 4. **Memory on large datasets**

Use `max_samples` parameter when testing. Full FinQA train.json (6000+ docs) needs 16GB+ RAM.

```python
await kg.build_from_json(path, max_samples=100)  # Start small
```

### 5. **Query results structure**

`semantic_search()` returns list of `QueryResult` dataclasses with `.score`, `.content`, `.node_id`, `.metadata` - not raw dicts.

**Full troubleshooting:** See `finqa_kg/TROUBLESHOOTING.md`

## Extending the System

### Adding New Entity Types

Edit `entity_extractor.py` → `extract_all_entities()` method. Add new patterns to regex or additional NER models. Update `EntityMention` dataclass if new metadata needed.

### Adding Query Methods

Add to `ModernFinQAGraphQuery` class in `knowledge_graph_query.py`. Follow pattern:

- Sync for graph traversal
- Async for transformer inference
- Return `QueryResult` or typed dataclass

### Visualization Customization

Use `GraphVisualizer` with `VisualizationConfig`:

```python
from finqa_kg.src.visualization import GraphVisualizer, VisualizationConfig

vis = GraphVisualizer(kg.graph)
config = VisualizationConfig(layout='spring', node_size=100, show_labels=True)
vis.visualize_full_graph(config=config, output_path='graph.html')
```

## Integration with Original FinQA Code

- `FinQA/code/retriever/` - Original passage retrieval system
- `FinQA/code/generator/` - Original answer generation
- `finqa_kg/` - **New KG-based approach** (separate from original)

Don't mix modules - KG system is standalone replacement, not extension.

## Quick Reference

- Entry point example: `finqa_kg/examples/demo.py`
- Test with minimal data: `finqa_kg/tests/test_data.json`
- Performance tuning: `finqa_kg/OPTIMIZATION_GUIDE.md`
- Architecture diagrams: `finqa_kg/ARCHITECTURE.md`
