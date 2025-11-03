# Mermaid Diagrams - FinQA Knowledge Graph System

## 1. Tổng Quan Kiến Trúc Hệ Thống

```mermaid
graph TB
    subgraph "Input Layer"
        JSON[FinQA JSON Data<br/>train.json, dev.json, test.json]
    end
    
    subgraph "Core Processing Layer"
        Builder[ModernFinQAKnowledgeGraph<br/>Builder]
        TextProc[Text Processor<br/>Cleaning & Normalization]
        EntityExt[Entity Extractor<br/>Multi-model NER]
        RelExt[Relation Extractor<br/>Zero-shot Classification]
    end
    
    subgraph "AI/ML Models"
        Spacy[Spacy Transformer<br/>en_core_web_trf]
        FinBERT[FinBERT NER<br/>Financial Entities]
        RoBERTa[RoBERTa NER<br/>General Entities]
        SentenceT[Sentence-Transformers<br/>all-MiniLM-L6-v2]
        ZeroShot[Zero-shot Classifier<br/>facebook/bart-large-mnli]
    end
    
    subgraph "Storage Layer"
        KG[Knowledge Graph<br/>NetworkX MultiDiGraph]
        EntityIdx[Entity Index<br/>Deduplication]
    end
    
    subgraph "Query Layer"
        QueryEngine[ModernFinQAGraphQuery<br/>Query Engine]
        SemanticSearch[Semantic Search]
        QA[Question Answering<br/>RoBERTa-Squad2]
        TrendAnalysis[Numerical Trend Analysis]
    end
    
    subgraph "Visualization Layer"
        Viz[GraphVisualizer]
        Interactive[Plotly Interactive<br/>HTML]
        Static[Matplotlib Static<br/>PNG/PDF]
    end
    
    subgraph "Output Layer"
        Results[Query Results]
        Charts[Visualizations]
        Stats[Graph Statistics]
    end
    
    JSON --> Builder
    Builder --> TextProc
    Builder --> EntityExt
    Builder --> RelExt
    
    TextProc --> Spacy
    EntityExt --> FinBERT
    EntityExt --> RoBERTa
    EntityExt --> Spacy
    RelExt --> ZeroShot
    
    Builder --> KG
    Builder --> EntityIdx
    
    KG --> QueryEngine
    QueryEngine --> SemanticSearch
    QueryEngine --> QA
    QueryEngine --> TrendAnalysis
    
    QueryEngine -.-> SentenceT
    QA -.-> RoBERTa
    
    KG --> Viz
    Viz --> Interactive
    Viz --> Static
    
    QueryEngine --> Results
    Viz --> Charts
    KG --> Stats
    
    style JSON fill:#e1f5fe
    style Builder fill:#fff3e0
    style KG fill:#f3e5f5
    style QueryEngine fill:#e8f5e9
    style Viz fill:#fce4ec
```

## 2. Quy Trình Xây Dựng Knowledge Graph (Chi Tiết)

```mermaid
sequenceDiagram
    participant User
    participant Builder as ModernFinQAKnowledgeGraph
    participant TextProc as Text Processor
    participant EntityExt as Entity Extractor
    participant RelExt as Relation Extractor
    participant KG as Knowledge Graph
    participant Models as AI Models
    
    User->>Builder: build_from_json(path)
    activate Builder
    
    Builder->>Builder: Load JSON data
    Builder->>Builder: Create DOC node
    
    par Process pre_text blocks
        loop For each pre_text
            Builder->>TextProc: Process text block
            activate TextProc
            TextProc->>TextProc: Clean & normalize
            TextProc->>Models: Get embeddings
            Models-->>TextProc: Vector embeddings
            TextProc-->>Builder: Processed text
            deactivate TextProc
            
            Builder->>EntityExt: Extract entities
            activate EntityExt
            EntityExt->>Models: FinBERT NER
            EntityExt->>Models: RoBERTa NER
            EntityExt->>Models: Spacy NER
            EntityExt->>EntityExt: Regex patterns (numbers, dates)
            Models-->>EntityExt: Entity mentions
            EntityExt-->>Builder: List of entities
            deactivate EntityExt
            
            Builder->>KG: Create TEXT node
            Builder->>KG: Create ENTITY nodes
            Builder->>KG: Add edges (contains_entity)
        end
    and Process table
        Builder->>Builder: Parse table structure
        Builder->>KG: Create TABLE node
        Builder->>KG: Create HEADER nodes
        
        loop For each cell
            Builder->>KG: Create CELL node
            Builder->>EntityExt: Extract entities from cell
            EntityExt-->>Builder: Cell entities
            Builder->>KG: Add edges (has_cell, contains_entity)
        end
    and Process post_text blocks
        loop For each post_text
            Builder->>TextProc: Process text block
            TextProc-->>Builder: Processed text
            Builder->>EntityExt: Extract entities
            EntityExt-->>Builder: Entities
            Builder->>KG: Create nodes & edges
        end
    end
    
    Builder->>RelExt: Create semantic links
    activate RelExt
    
    loop For pre_text × post_text pairs
        RelExt->>Models: Compute similarity
        Models-->>RelExt: Similarity score
        
        alt Similarity > threshold
            RelExt->>Models: Zero-shot classification
            Models-->>RelExt: Relation type & confidence
            RelExt->>KG: Add semantic edge
        end
    end
    deactivate RelExt
    
    Builder->>KG: Add QA node (if exists)
    Builder->>Builder: Build entity index
    
    Builder-->>User: Knowledge Graph complete
    deactivate Builder
```

## 3. Cấu Trúc Node và Edge trong Knowledge Graph

```mermaid
graph LR
    subgraph "Node Types"
        DOC[Document Node<br/>doc_id, filename]
        TEXT[Text Node<br/>content, block_type<br/>embedding]
        TABLE[Table Node<br/>rows, cols]
        HEADER[Header Node<br/>content, col_index<br/>embedding]
        CELL[Cell Node<br/>content, row, col<br/>embedding]
        ENTITY[Entity Node<br/>text, type, value<br/>metadata]
        QA[QA Node<br/>question, answer]
    end
    
    subgraph "Edge Types"
        E1[has_pre_text<br/>order]
        E2[has_post_text<br/>order]
        E3[has_table]
        E4[has_qa]
        E5[has_header<br/>col]
        E6[has_cell<br/>row, col]
        E7[contains_entity<br/>start, end]
        E8[semantic_link<br/>similarity, confidence]
        E9[header_to_cell]
    end
    
    DOC -->|E1| TEXT
    DOC -->|E2| TEXT
    DOC -->|E3| TABLE
    DOC -->|E4| QA
    TABLE -->|E5| HEADER
    TABLE -->|E6| CELL
    HEADER -->|E9| CELL
    TEXT -->|E7| ENTITY
    CELL -->|E7| ENTITY
    TEXT -.->|E8| TEXT
    
    style DOC fill:#1f77b4,color:#fff
    style TEXT fill:#ff7f0e,color:#fff
    style TABLE fill:#2ca02c,color:#fff
    style HEADER fill:#9467bd,color:#fff
    style CELL fill:#8c564b,color:#fff
    style ENTITY fill:#bcbd22,color:#000
    style QA fill:#17becf,color:#fff
```

## 4. Entity Extraction Pipeline (Multi-Model Approach)

```mermaid
flowchart TD
    Start([Input Text]) --> Clean[Text Cleaning<br/>Remove noise, normalize]
    
    Clean --> Parallel{Parallel Extraction}
    
    Parallel -->|Track 1| FinBERT[FinBERT NER<br/>Financial Terms]
    Parallel -->|Track 2| RoBERTa[RoBERTa NER<br/>General Entities]
    Parallel -->|Track 3| Spacy[Spacy Transformer<br/>Linguistic Analysis]
    Parallel -->|Track 4| Regex[Regex Patterns<br/>Numbers, Dates, Money]
    
    FinBERT --> Merge[Merge & Deduplicate]
    RoBERTa --> Merge
    Spacy --> Merge
    Regex --> Merge
    
    Merge --> Normalize[Normalize Entity Text<br/>Lowercase, trim]
    
    Normalize --> Check{Entity in Index?}
    
    Check -->|Yes| Reuse[Reuse Existing Node ID]
    Check -->|No| Create[Create New Entity Node]
    
    Create --> AddIndex[Add to Entity Index]
    AddIndex --> Return([Return Node ID])
    Reuse --> Return
    
    style Start fill:#e1f5fe
    style Parallel fill:#fff3e0
    style Merge fill:#f3e5f5
    style Check fill:#ffebee
    style Return fill:#e8f5e9
```

## 5. Query Engine - Semantic Search Flow

```mermaid
sequenceDiagram
    participant User
    participant Query as QueryEngine
    participant Index as Search Index
    participant Model as Sentence Transformer
    participant KG as Knowledge Graph
    
    User->>Query: semantic_search("revenue growth", k=5)
    activate Query
    
    Query->>Query: Normalize query
    Query->>Model: Encode query text
    activate Model
    Model-->>Query: Query embedding (384-dim)
    deactivate Model
    
    Query->>Index: Get all text embeddings
    Index-->>Query: Embedding matrix (N × 384)
    
    Query->>Query: Compute cosine similarities
    Note over Query: similarity = dot(query, texts) / norms
    
    Query->>Query: Sort by similarity (descending)
    Query->>Query: Filter by min_score threshold
    
    loop For top K results
        Query->>KG: Get node data
        KG-->>Query: Node content & metadata
        Query->>Query: Create QueryResult object
    end
    
    Query-->>User: List[QueryResult]
    deactivate Query
    
    Note over User,Query: QueryResult contains:<br/>- node_id<br/>- score<br/>- content<br/>- type<br/>- metadata
```

## 6. Question Answering Pipeline

```mermaid
flowchart TD
    Start([User Question]) --> Encode[Encode Question<br/>Using Sentence Transformer]
    
    Encode --> Search[Semantic Search<br/>Find top K similar nodes]
    
    Search --> Context[Build Context<br/>Combine retrieved texts]
    
    Context --> QAModel[RoBERTa QA Model<br/>deepset/roberta-base-squad2]
    
    QAModel --> Extract[Extract Answer Span<br/>with confidence score]
    
    Extract --> Check{Score > Threshold?}
    
    Check -->|Yes| Return[Return Answer]
    Check -->|No| NoAnswer[Return None]
    
    Return --> End([Answer String])
    NoAnswer --> End
    
    style Start fill:#e1f5fe
    style QAModel fill:#fff3e0
    style Check fill:#ffebee
    style End fill:#e8f5e9
```

## 7. Table Processing - Parallel Cell Processing

```mermaid
graph TB
    subgraph "Table Structure"
        Table[Table<br/>rows × cols]
        
        subgraph "Headers Row"
            H1[Header 1<br/>col=0]
            H2[Header 2<br/>col=1]
            H3[Header 3<br/>col=2]
            H4[Header N<br/>col=N]
        end
        
        subgraph "Data Rows"
            C11[Cell 1,1]
            C12[Cell 1,2]
            C13[Cell 1,3]
            C21[Cell 2,1]
            C22[Cell 2,2]
            C23[Cell 2,3]
        end
    end
    
    Table --> H1
    Table --> H2
    Table --> H3
    Table --> H4
    
    H1 -.header_to_cell.-> C11
    H2 -.header_to_cell.-> C12
    H3 -.header_to_cell.-> C13
    
    H1 -.header_to_cell.-> C21
    H2 -.header_to_cell.-> C22
    H3 -.header_to_cell.-> C23
    
    Table -->|has_cell<br/>row=1, col=1| C11
    Table -->|has_cell<br/>row=1, col=2| C12
    Table -->|has_cell<br/>row=1, col=3| C13
    
    subgraph "Cell Processing (Async)"
        C11 --> E1[Extract Entities]
        C12 --> E2[Extract Entities]
        C13 --> E3[Extract Entities]
        
        E1 --> EN1[Entity Nodes]
        E2 --> EN2[Entity Nodes]
        E3 --> EN3[Entity Nodes]
    end
    
    style Table fill:#2ca02c,color:#fff
    style H1 fill:#9467bd,color:#fff
    style H2 fill:#9467bd,color:#fff
    style H3 fill:#9467bd,color:#fff
    style C11 fill:#8c564b,color:#fff
    style C12 fill:#8c564b,color:#fff
    style EN1 fill:#bcbd22
```

## 8. Visualization Pipeline

```mermaid
flowchart LR
    KG[Knowledge Graph<br/>NetworkX] --> Config[Visualization Config<br/>Colors, Sizes, Layout]
    
    Config --> Layout{Layout Algorithm}
    
    Layout -->|spring| Spring[Force-directed<br/>Spring Layout]
    Layout -->|circular| Circular[Circular<br/>Layout]
    Layout -->|shell| Shell[Shell<br/>Layout]
    Layout -->|random| Random[Random<br/>Layout]
    
    Spring --> Render{Render Engine}
    Circular --> Render
    Shell --> Render
    Random --> Render
    
    Render -->|Static| MPL[Matplotlib<br/>PNG/PDF]
    Render -->|Interactive| Plotly[Plotly<br/>HTML]
    
    MPL --> Output1[Static Image<br/>test_graph.png]
    Plotly --> Output2[Interactive HTML<br/>test_graph.html]
    
    style KG fill:#f3e5f5
    style Render fill:#fff3e0
    style Output1 fill:#e8f5e9
    style Output2 fill:#e1f5fe
```

## 9. Data Flow - From JSON to Query Results

```mermaid
graph TD
    subgraph "Input"
        JSON["FinQA JSON<br/>{<br/>  id: 'doc_1',<br/>  pre_text: [...],<br/>  table: [...],<br/>  post_text: [...],<br/>  qa: {...}<br/>}"]
    end
    
    subgraph "Parsing"
        Parse[Parse JSON<br/>Extract fields]
        Validate[Validate Structure]
    end
    
    subgraph "NLP Processing"
        Clean[Clean Text]
        Tokenize[Tokenize]
        NER[Named Entity Recognition]
        Embed[Generate Embeddings]
    end
    
    subgraph "Graph Construction"
        CreateNodes[Create Nodes<br/>DOC, TEXT, TABLE, etc.]
        CreateEdges[Create Edges<br/>Relations]
        IndexEntities[Build Entity Index<br/>Deduplication]
    end
    
    subgraph "Storage"
        NetworkX[NetworkX MultiDiGraph<br/>In-Memory Storage]
    end
    
    subgraph "Querying"
        QueryInterface[Query Interface]
        Semantic[Semantic Search]
        QA[Question Answering]
        Analysis[Trend Analysis]
    end
    
    subgraph "Output"
        Results["Query Results<br/>[<br/>  {node_id, score, content},<br/>  ...<br/>]"]
    end
    
    JSON --> Parse
    Parse --> Validate
    Validate --> Clean
    Clean --> Tokenize
    Tokenize --> NER
    Tokenize --> Embed
    
    NER --> CreateNodes
    Embed --> CreateNodes
    CreateNodes --> CreateEdges
    CreateEdges --> IndexEntities
    
    IndexEntities --> NetworkX
    
    NetworkX --> QueryInterface
    QueryInterface --> Semantic
    QueryInterface --> QA
    QueryInterface --> Analysis
    
    Semantic --> Results
    QA --> Results
    Analysis --> Results
    
    style JSON fill:#e1f5fe
    style NetworkX fill:#f3e5f5
    style Results fill:#e8f5e9
```

## 10. Async Processing Model

```mermaid
sequenceDiagram
    participant Main as Main Thread
    participant Builder as KG Builder (Async)
    participant Task1 as Pre-text Task
    participant Task2 as Table Task
    participant Task3 as Post-text Task
    participant Models as AI Models
    
    Main->>Builder: await build_from_json()
    activate Builder
    
    par Parallel Processing
        Builder->>Task1: Process pre_text blocks
        activate Task1
        loop For each block
            Task1->>Models: Extract entities
            Models-->>Task1: Entities
            Task1->>Task1: Create nodes & edges
        end
        Task1-->>Builder: Pre-text complete
        deactivate Task1
        
    and
        Builder->>Task2: Process table
        activate Task2
        Task2->>Task2: Create table & header nodes
        
        par Process cells in parallel
            Task2->>Models: Cell 1 entities
            Task2->>Models: Cell 2 entities
            Task2->>Models: Cell N entities
        end
        
        Task2-->>Builder: Table complete
        deactivate Task2
        
    and
        Builder->>Task3: Process post_text blocks
        activate Task3
        loop For each block
            Task3->>Models: Extract entities
            Models-->>Task3: Entities
            Task3->>Task3: Create nodes & edges
        end
        Task3-->>Builder: Post-text complete
        deactivate Task3
    end
    
    Builder->>Builder: Create semantic links
    Builder->>Builder: Add QA node
    Builder-->>Main: Graph built successfully
    deactivate Builder
    
    Note over Main,Builder: All I/O and NLP operations<br/>run asynchronously for performance
```

---

## Chú Thích

### Node Types (Các loại Node)
- **DOC**: Document node - điểm bắt đầu cho mỗi tài liệu
- **TEXT**: Text block node - đoạn văn bản (pre/post)
- **TABLE**: Table structure node - cấu trúc bảng
- **HEADER**: Table header node - tiêu đề cột
- **CELL**: Table cell node - ô trong bảng
- **ENTITY**: Entity node - thực thể (số, ngày, tên, v.v.)
- **QA**: Question-Answer node - cặp hỏi đáp

### Edge Types (Các loại Edge)
- **has_pre_text/has_post_text**: Document → Text
- **has_table**: Document → Table
- **has_qa**: Document → QA
- **has_header**: Table → Header
- **has_cell**: Table → Cell
- **contains_entity**: Text/Cell → Entity
- **semantic_link**: Text ↔ Text (có similarity cao)
- **header_to_cell**: Header → Cell

### AI Models Used
1. **Spacy Transformer** (`en_core_web_trf`): NLP cơ bản
2. **FinBERT** (`yiyanghkust/finbert-pretrained-ner`): Financial NER
3. **RoBERTa** (`jean-baptiste/roberta-large-ner-english`): General NER
4. **Sentence Transformer** (`all-MiniLM-L6-v2`): Text embeddings
5. **Zero-shot Classifier** (`facebook/bart-large-mnli`): Relation classification
6. **QA Model** (`deepset/roberta-base-squad2`): Question answering
```
