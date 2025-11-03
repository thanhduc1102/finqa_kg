# 🚀 Quick Reference - FinQA Knowledge Graph

## ⚡ Chạy Nhanh Trong 30 Giây

```bash
# 1. Fix dependencies
cd finqa_kg && ./fix_environment.sh

# 2. Test
python tests/test_basic.py

# 3. Done! ✅
```

---

## 📊 Hệ Thống Là Gì? (1 câu)

**Chuyển đổi tài liệu tài chính → Đồ thị tri thức → Tìm kiếm & trả lời câu hỏi bằng AI**

---

## 🏗️ Kiến Trúc (3 tầng)

```
1. BUILDER   → Xây dựng đồ thị từ JSON
2. QUERY     → Tìm kiếm và trả lời câu hỏi  
3. VISUALIZE → Vẽ đồ thị
```

---

## 📦 Cấu Trúc Đồ Thị (7 loại node)

```
📄 DOC
  ├─ 📝 TEXT → 🏷️ ENTITY
  ├─ 📊 TABLE
  │   ├─ 📋 HEADER
  │   └─ 📑 CELL → 🏷️ ENTITY
  └─ ❓ QA
```

---

## 🤖 AI Models (6 models)

1. **Spacy** - NLP cơ bản
2. **FinBERT** - Thuật ngữ tài chính
3. **RoBERTa** - Thực thể chung
4. **Sentence-Transformers** - Embeddings
5. **Zero-shot** - Phân loại quan hệ
6. **RoBERTa-Squad2** - Trả lời câu hỏi

---

## 💻 Code Examples

### Build Graph
```python
import asyncio
from finqa_kg.src.builder import ModernFinQAKnowledgeGraph

async def build():
    kg = ModernFinQAKnowledgeGraph()
    await kg.build_from_json('data.json')
    return kg

kg = asyncio.run(build())
```

### Query
```python
from finqa_kg.src.query import ModernFinQAGraphQuery

query = ModernFinQAGraphQuery(kg.graph)
results = query.semantic_search("revenue", k=5)
answer = await query.answer_question("What is revenue?")
```

### Visualize
```python
from finqa_kg.src.visualization import GraphVisualizer

vis = GraphVisualizer(kg.graph)
vis.create_interactive_visualization("graph.html")
```

---

## 🐛 Lỗi Phổ Biến & Fix

### Lỗi: `numpy.dtype size changed`
```bash
pip uninstall -y numpy spacy thinc
pip install numpy==1.24.3 scipy==1.10.1
pip install spacy==3.7.2 spacy-transformers==1.3.4
python -m spacy download en_core_web_trf
```

### Lỗi: `Can't find model 'en_core_web_trf'`
```bash
python -m spacy download en_core_web_trf
```

### Lỗi: `ModuleNotFoundError: No module named 'finqa_kg'`
```bash
cd /path/to/FinQA_research
python finqa_kg/tests/test_basic.py
```

---

## 📚 Đọc Gì Tiếp?

### Mới bắt đầu?
→ **GIAI_THICH_HE_THONG.md**

### Xem biểu đồ?
→ **MERMAID_DIAGRAMS.md**

### Gặp lỗi?
→ **TOM_TAT_FIX.md**

### Tìm file cụ thể?
→ **NAVIGATION.md**

---

## 🎯 Key Concepts

| Concept | Giải Thích |
|---------|------------|
| **Async** | Xử lý song song, nhanh hơn |
| **Entity** | Thực thể (số, ngày, tên) |
| **Embedding** | Vector 384 chiều đại diện text |
| **Semantic Search** | Tìm theo ý nghĩa, không cần từ khóa |
| **MultiDiGraph** | Đồ thị có hướng, nhiều edges |

---

## 📈 Performance

| Metric | CPU | GPU |
|--------|-----|-----|
| Docs/min | 5 | 50 |
| Query | <1s | <1s |
| Memory (100 docs) | 2GB | 2GB |

---

## 🔗 Important Files

| File | Mục Đích |
|------|----------|
| `src/builder/knowledge_graph_builder.py` | ⭐ Core builder |
| `src/query/knowledge_graph_query.py` | ⭐ Core query |
| `tests/test_basic.py` | 🧪 Test example |
| `examples/demo.py` | 📝 Demo code |

---

## 📞 Tìm Hiểu Thêm

```
📁 Xem NAVIGATION.md → Index đầy đủ
📊 Xem MERMAID_DIAGRAMS.md → 10 biểu đồ
📖 Xem GIAI_THICH_HE_THONG.md → Giải thích chi tiết
```

---

**That's it! 🎉**

*Print this page and keep it on your desk!*
