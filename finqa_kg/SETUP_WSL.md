# 🚀 Hướng Dẫn Cài Đặt & Chạy Hệ Thống - WSL

## 📋 Tóm Tắt Hệ Thống

Đây là hệ thống **Intelligent Knowledge Graph** cho FinQA với 5 phases:

1. **KG Builder** - Xây dựng KG từ text+table với Spacy NER
2. **Question Analyzer** - Phân tích question type, entities, operations
3. **Program Synthesizer** - Tự động sinh program từ question + KG
4. **Program Executor** - Execute với provenance tracking
5. **Pipeline Integration** - Tích hợp toàn bộ

## ✅ Kiến Trúc Mới (Đã Fix!)

```
Sample → [KG Builder với NLP] → Knowledge Graph
           ↓
Question → [Question Analyzer] → Question Type + Entities + Operations
           ↓
KG + Analysis → [Program Synthesizer] → Program với Arguments Ordered
           ↓
Program → [Program Executor] → Final Answer + Steps + Provenance
```

## 🛠️ Cài Đặt Trên WSL

### Bước 1: Cài đặt Python dependencies

```bash
# Vào thư mục project
cd /mnt/e/AI/FinQA_research/finqa_kg

# Cài đặt theo thứ tự CHÍNH XÁC (quan trọng!)
pip install numpy==1.24.3 scipy==1.10.1

pip install spacy==3.7.2 spacy-transformers==1.3.4

pip install networkx matplotlib plotly pandas tqdm

# Download Spacy model (320MB, cần internet)
python -m spacy download en_core_web_trf
```

### Bước 2: Kiểm tra cài đặt

```bash
python -c "import spacy; nlp = spacy.load('en_core_web_trf'); print('✓ Spacy OK')"
python -c "import networkx; print('✓ NetworkX OK')"
python -c "import matplotlib; print('✓ Matplotlib OK')"
```

### Bước 3: Chạy test

```bash
# Test với 1 sample
cd /mnt/e/AI/FinQA_research/finqa_kg
python tests/test_intelligent_pipeline.py --mode single

# Test với 5 samples
python tests/test_intelligent_pipeline.py --mode multiple --num_samples 5
```

## 📁 Files Đã Tạo

```
finqa_kg/src/pipeline/
├── intelligent_kg_builder.py      # KG construction với NLP
├── question_analyzer.py           # Question analysis
├── program_synthesizer.py         # Auto program synthesis
├── program_executor.py            # Execution với tracking
└── finqa_intelligent_pipeline.py  # Main pipeline

finqa_kg/tests/
└── test_intelligent_pipeline.py   # Test script

finqa_kg/
├── requirements.txt               # Dependencies
└── SETUP_WSL.md                   # File này
```

## 🎯 Cách Sử Dụng

### Example 1: Process một sample

```python
import asyncio
import json
from finqa_kg.src.pipeline import IntelligentFinQAPipeline

async def main():
    # Load data
    with open('FinQA/dataset/train.json') as f:
        data = json.load(f)

    # Initialize pipeline
    pipeline = IntelligentFinQAPipeline()

    # Process sample
    result = await pipeline.process_sample(data[0])

    # Print results
    print(f"Answer: {result.final_answer}")
    print(f"Correct: {result.is_correct}")
    print(f"Explanation:\n{result.full_explanation}")

asyncio.run(main())
```

### Example 2: Analyze nhiều samples

```bash
python tests/test_intelligent_pipeline.py --mode multiple --num_samples 10
```

## 🔍 Chi Tiết Components

### 1. IntelligentKGBuilder

- **Input**: pre_text, post_text, table
- **Process**:
  - Spacy NER để extract entities
  - Dependency parsing để extract relations
  - Semantic linking giữa text và table
- **Output**: NetworkX MultiDiGraph với entities, relations, numbers

### 2. QuestionAnalyzer

- **Input**: Question string
- **Process**:
  - Detect question type (percentage_change, ratio, average, etc)
  - Extract entities mentioned
  - Extract temporal information
  - Determine operation sequence
  - Resolve argument order
- **Output**: QuestionAnalysis object

### 3. ProgramSynthesizer

- **Input**: QuestionAnalysis + KG + EntityIndex
- **Process**:
  - Chọn template dựa vào question type
  - Query KG để retrieve arguments
  - Resolve argument order (old/new, numerator/denominator)
  - Generate program string
- **Output**: ProgramSynthesisResult với program + placeholders

### 4. ProgramExecutor

- **Input**: Program string + Placeholders
- **Process**:
  - Parse program thành tree
  - Execute recursively
  - Track provenance (which KG nodes used)
  - Build computation graph
- **Output**: ExecutionResult với answer + steps + graph

### 5. IntelligentFinQAPipeline

- **Tích hợp tất cả 4 components trên**
- **Thêm visualization và explanation**

## ⚠️ Lưu Ý Quan Trọng

1. **Thứ tự cài đặt**: PHẢI cài numpy trước spacy!
2. **Spacy model**: Cần download riêng `en_core_web_trf` (320MB)
3. **Memory**: Mỗi sample cần ~50-100MB RAM khi xử lý
4. **GPU**: Tự động detect CUDA, không bắt buộc

## 🐛 Troubleshooting

### Lỗi: "No module named 'spacy'"

```bash
pip install spacy==3.7.2
```

### Lỗi: "Can't find model 'en_core_web_trf'"

```bash
python -m spacy download en_core_web_trf
```

### Lỗi: "numpy.dtype size changed"

```bash
pip uninstall -y numpy spacy
pip install numpy==1.24.3
pip install spacy==3.7.2
```

## 📊 Expected Performance

- **Build KG**: 1-2s per sample
- **Analyze Question**: <0.1s
- **Synthesize Program**: 0.5-1s (depend on KG size)
- **Execute**: <0.1s
- **Total**: 2-3s per sample

## 🎓 Citation

Dựa trên paper: "FinQA: A Dataset of Numerical Reasoning over Financial Data" (EMNLP 2021)

## 📝 Next Steps

1. **Improve synthesis logic** - Add more question patterns
2. **Better entity linking** - Use BERT embeddings
3. **Multi-step reasoning** - Support complex programs
4. **Error analysis** - Debug failed cases
