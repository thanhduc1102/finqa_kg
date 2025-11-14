# QUY TRÌNH XÂY DỰNG KNOWLEDGE GRAPH

## 1. OVERVIEW - Luồng Dữ Liệu Tổng Quan

```mermaid
graph TB
    Start[FinQA Sample] --> Extract[Trích xuất dữ liệu]
    Extract --> Table[Table Data]
    Extract --> PreText[Pre Text]
    Extract --> PostText[Post Text]
    Extract --> QA[Question & Program]
    
    Table --> KG[Knowledge Graph Builder]
    PreText --> KG
    PostText --> KG
    
    KG --> Graph[Structured Knowledge Graph]
    
    QA --> Retriever[Argument Retriever]
    Graph --> Retriever
    
    Retriever --> Program[Program Executor]
    Program --> Answer[Final Answer]
    
    style Start fill:#e1f5ff
    style Graph fill:#c8e6c9
    style Answer fill:#fff9c4
```

## 2. CHI TIẾT XÂY DỰNG KNOWLEDGE GRAPH

```mermaid
flowchart TD
    subgraph Input["📥 INPUT DATA"]
        S[Sample] --> T[Table 2D Array]
        S --> PT[Pre Text List]
        S --> POST[Post Text List]
    end
    
    subgraph TableProcessing["📊 TABLE PROCESSING"]
        T --> ParseHeader[Parse Header Row]
        ParseHeader --> NormHeader[Normalize Column Names]
        NormHeader --> CreateTable[Create TABLE Node]
        
        CreateTable --> IterRows[For each row]
        IterRows --> CreateRow[Create ROW Node]
        CreateRow --> IterCells[For each cell in row]
        
        IterCells --> ParseCell[Parse Cell Value]
        ParseCell --> CheckNum{Is Number?}
        
        CheckNum -->|Yes| ExtractMeta[Extract Metadata]
        ExtractMeta --> IsPct{Has %}
        ExtractMeta --> IsCurr{Has $}
        ExtractMeta --> IsNeg{Parentheses?}
        
        IsPct -->|Yes| IdxPct[Index as Percentage]
        IsCurr -->|Yes| IdxCurr[Mark as Currency]
        IsNeg -->|Yes| Negate[Make Negative]
        
        CheckNum -->|No| TextCell[Text Cell]
        
        IdxPct --> CreateCell[Create CELL Node]
        IdxCurr --> CreateCell
        Negate --> CreateCell
        TextCell --> CreateCell
        
        CreateCell --> IndexValue[Add to Value Index]
        IndexValue --> LinkRow[Link ROW -> CELL]
        LinkRow --> LinkSameRow[Link to other CELLs in same ROW]
        LinkSameRow --> LinkSameCol[Link to CELLs in same COLUMN]
    end
    
    subgraph TextProcessing["📝 TEXT PROCESSING"]
        PT --> CombineText[Combine Pre/Post Text]
        POST --> CombineText
        
        CombineText --> IterText[For each sentence]
        IterText --> CreateText[Create TEXT Node]
        CreateText --> ExtractNum[Extract numbers from text]
        ExtractNum --> LinkText[Link TEXT -> TABLE]
    end
    
    subgraph GraphOutput["🎯 OUTPUT GRAPH"]
        LinkSameCol --> FinalKG[Complete Knowledge Graph]
        LinkText --> FinalKG
        
        FinalKG --> Nodes[Nodes with Metadata]
        FinalKG --> Edges[Typed Relationships]
        FinalKG --> Index[Value Index Map]
    end
    
    style Input fill:#e3f2fd
    style TableProcessing fill:#f3e5f5
    style TextProcessing fill:#fff3e0
    style GraphOutput fill:#e8f5e9
```

## 3. CẤU TRÚC NODE TRONG GRAPH

```mermaid
classDiagram
    class TableNode {
        +String type = "table"
        +String label = "TABLE"
        +int rows
        +int cols
    }
    
    class RowNode {
        +String type = "row"
        +String label = "ROW"
        +int row_index
        +bool is_header
        +List row_data
    }
    
    class CellNode {
        +String type = "cell"
        +String label = "NUMBER|TEXT|MONEY|PERCENT"
        +int row_index
        +int col_index
        +String column_name
        +String raw_value
        +float value
        +bool is_header
        +String context
        ---
        +bool is_percent
        +bool is_currency
        +bool is_negative
        +String original_format
    }
    
    class TextNode {
        +String type = "text"
        +String label = "TEXT"
        +String content
    }
    
    TableNode "1" --> "*" RowNode : HAS_ROW
    RowNode "1" --> "*" CellNode : HAS_CELL
    CellNode --> CellNode : SAME_ROW
    CellNode --> CellNode : SAME_COLUMN
    TextNode --> TableNode : REFERS_TO
```

## 4. PARSE CELL VALUE - Chi Tiết

```mermaid
flowchart TD
    Start[Cell Value String] --> Trim[Trim whitespace]
    Trim --> CheckEmpty{Empty or - or N/A?}
    
    CheckEmpty -->|Yes| ReturnNull[Return: None, TEXT, {}]
    CheckEmpty -->|No| DetectFormat[Detect Format]
    
    DetectFormat --> HasPct{Contains %}
    DetectFormat --> HasDollar{Contains $}
    DetectFormat --> HasParen{Has parentheses?}
    
    HasPct -->|Yes| MarkPct[metadata.is_percent = True]
    HasDollar -->|Yes| MarkCurr[metadata.is_currency = True]
    HasParen -->|Yes| MarkNeg[metadata.is_negative = True]
    
    MarkPct --> Clean[Remove formatting]
    MarkCurr --> Clean
    MarkNeg --> Clean
    HasPct -->|No| Clean
    HasDollar -->|No| Clean
    HasParen -->|No| Clean
    
    Clean --> RemoveComma[Remove commas]
    RemoveComma --> RemoveDollar[Remove $]
    RemoveDollar --> RemovePct[Remove %]
    RemovePct --> HandleParen[Handle parentheses]
    
    HandleParen --> TryParse{Can parse as float?}
    
    TryParse -->|Yes| ParseNum[value = float parsed]
    TryParse -->|No| ReturnText[Return: None, TEXT, metadata]
    
    ParseNum --> ApplyNeg{Is negative?}
    ApplyNeg -->|Yes| Negate[value = -value]
    ApplyNeg -->|No| DetermineLabel
    Negate --> DetermineLabel
    
    DetermineLabel --> CheckLabel{Format type?}
    CheckLabel -->|is_currency| LabelMoney[label = MONEY]
    CheckLabel -->|is_percent| LabelPct[label = PERCENT]
    CheckLabel -->|neither| LabelNum[label = NUMBER]
    
    LabelMoney --> Return[Return: value, label, metadata]
    LabelPct --> Return
    LabelNum --> Return
    
    style Start fill:#e1f5ff
    style Return fill:#c8e6c9
    style ReturnNull fill:#ffcdd2
    style ReturnText fill:#ffcdd2
```

## 5. VALUE INDEXING - Tạo Index để Tìm Nhanh

```mermaid
flowchart LR
    subgraph Creation["Node Creation"]
        Cell[Create Cell Node] --> GetValue[Get numeric value]
        GetValue --> CheckPct{Is percentage?}
    end
    
    subgraph Indexing["Value Indexing"]
        CheckPct -->|No| IdxNormal[value_index[value] = node_id]
        CheckPct -->|Yes| IdxBoth[Index both forms]
        
        IdxBoth --> Idx1[value_index[23.6] = node_id]
        IdxBoth --> Idx2[value_index[0.236] = node_id]
    end
    
    subgraph Lookup["Fast Lookup"]
        Query[Search for value] --> HashLookup[O1 Hash Lookup]
        HashLookup --> GetNodes[value_index[target]]
        GetNodes --> Instant[Instant Results!]
    end
    
    style Indexing fill:#fff9c4
    style Lookup fill:#c8e6c9
```

## 6. ARGUMENT RETRIEVAL - Từ Program đến Data

```mermaid
sequenceDiagram
    participant P as Program
    participant R as Retriever
    participant KG as Knowledge Graph
    participant VI as Value Index
    
    P->>R: Need argument "9896"
    R->>R: Parse: is number, not #ref
    R->>VI: find_nodes_by_value(9896)
    VI->>VI: Hash lookup O(1)
    VI->>R: Return [CELL_42, CELL_89]
    
    R->>R: Multiple matches, need context
    R->>P: Extract context from question
    P->>R: Keywords: ["revenue", "2018"]
    
    R->>KG: filter by column_name contains "revenue"
    KG->>R: CELL_42 matches
    
    R->>KG: filter by row contains "2018"
    KG->>R: CELL_42 matches both!
    
    R->>P: Return CELL_42 with value 9896
    
    Note over P,R: If percentage: also try value/100
```

## 7. ĐẦY ĐỦ PIPELINE - End to End

```mermaid
graph TB
    subgraph DataPrep["1️⃣ DATA PREPARATION"]
        Raw[Raw Sample] --> Parse[Parse JSON]
        Parse --> Table[Extract Table]
        Parse --> Text[Extract Text]
        Parse --> Q[Extract Question]
        Parse --> Prog[Extract Program]
    end
    
    subgraph KGBuild["2️⃣ KNOWLEDGE GRAPH BUILD"]
        Table --> BuildTable[Build Table Topology]
        BuildTable --> TableNodes[TABLE→ROW→CELL nodes]
        
        Text --> ExtractText[Extract Text Entities]
        ExtractText --> TextNodes[TEXT nodes]
        
        TableNodes --> Link[Link Relationships]
        TextNodes --> Link
        Link --> IndexVal[Create Value Index]
    end
    
    subgraph Query["3️⃣ ARGUMENT RETRIEVAL"]
        Prog --> ParseProg[Parse Program]
        ParseProg --> ExtractArgs[Extract Arguments]
        
        ExtractArgs --> CheckType{Argument Type}
        CheckType -->|#0,#1| Intermediate[From Previous Step]
        CheckType -->|const_XXX| Scaling[Scaling Factor]
        CheckType -->|number| LookupVal[Lookup in KG]
        
        Q --> ExtractCtx[Extract Context]
        ExtractCtx --> Keywords[Keywords for Filtering]
        
        IndexVal --> LookupVal
        Keywords --> LookupVal
        LookupVal --> Matched[Matched Nodes]
    end
    
    subgraph Execute["4️⃣ PROGRAM EXECUTION"]
        Matched --> Resolve[Resolve All Arguments]
        Intermediate --> Resolve
        Scaling --> Resolve
        
        Resolve --> ExecOps[Execute Operations]
        ExecOps --> Step1[Step 1: op1 args → #0]
        Step1 --> Step2[Step 2: op2 args,#0 → #1]
        Step2 --> Result[Final Result]
    end
    
    style DataPrep fill:#e3f2fd
    style KGBuild fill:#f3e5f5
    style Query fill:#fff3e0
    style Execute fill:#e8f5e9
```

## 8. EXAMPLE - Concrete Sample

**Input:**
```json
{
  "table": [
    ["Year", "Revenue", "Percent"],
    ["2018", "$9896", "23.6%"],
    ["2017", "$7510", "19.6%"]
  ],
  "qa": {
    "question": "what was the total operating expenses in 2018 in millions",
    "program": "divide(9896, 23.6%)",
    "answer": "41932"
  }
}
```

**KG Construction Flow:**

```mermaid
graph TD
    subgraph Input
        T[Table 3x3]
        Q[Question]
        P[Program]
    end
    
    subgraph KGNodes["Built Nodes"]
        TN[TABLE_1]
        R0[ROW_1 header]
        R1[ROW_2 data]
        R2[ROW_3 data]
        
        C1["CELL_1 [0,0] = 'Year'"]
        C2["CELL_2 [0,1] = 'Revenue'"]
        C3["CELL_3 [0,2] = 'Percent'"]
        C4["CELL_4 [1,0] = '2018'"]
        C5["CELL_5 [1,1] = '$9896'<br/>value=9896<br/>is_currency=true"]
        C6["CELL_6 [1,2] = '23.6%'<br/>value=23.6<br/>is_percent=true"]
    end
    
    subgraph Index["Value Index"]
        I1["9896 → [CELL_5]"]
        I2["23.6 → [CELL_6]"]
        I3["0.236 → [CELL_6]"]
    end
    
    subgraph Retrieval["Argument Retrieval"]
        P --> A1["Need: 9896"]
        P --> A2["Need: 23.6%"]
        
        A1 --> I1
        A2 --> I2
        
        I1 --> Found1[✓ CELL_5]
        I2 --> Found2[✓ CELL_6]
    end
    
    subgraph Execution
        Found1 --> E["divide(9896, 23.6)"]
        Found2 --> E
        E --> Ans["= 41932"]
    end
    
    T --> TN
    TN --> R0
    TN --> R1
    R0 --> C1
    R0 --> C2
    R1 --> C5
    R1 --> C6
    
    C5 --> I1
    C6 --> I2
    C6 --> I3
```

---

## KEY POINTS

1. **Table → Structured topology** (TABLE → ROW → CELL)
2. **Cell parsing → Rich metadata** (%, $, negative)
3. **Value indexing → O(1) lookup** (instant search)
4. **Context filtering → Accurate retrieval** (column/row keywords)
5. **Program execution → State management** (#0, #1 intermediate results)

Đây là quy trình hoàn chỉnh để xây dựng Knowledge Graph từ dữ liệu FinQA!
