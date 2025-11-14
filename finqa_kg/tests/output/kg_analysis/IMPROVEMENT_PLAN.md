# IMPROVEMENT PLAN FOR KNOWLEDGE GRAPH SYSTEM
## Actionable Steps to Increase Accuracy from 15% to 70%+

**Priority Order:** Fix biggest bottlenecks first for maximum impact

---

## PHASE 1: CRITICAL FIXES (Week 1) - Target: 35-40% Accuracy

### 1.1 Fix Percentage Extraction & Matching ⭐ HIGH IMPACT

**Problem:** 
- Table cell "23.6%" not matched to program argument `23.6%`
- Number extraction strips % symbol, breaks matching

**Solution:**
```python
# In structured_kg_builder.py: _parse_cell_value()

def _parse_cell_value(self, cell_value: str) -> Tuple[Optional[float], str, Dict]:
    """Enhanced to preserve format metadata"""
    cell_value = cell_value.strip()
    metadata = {'original_format': cell_value}
    
    # Check for percentage
    is_percent = '%' in cell_value
    cleaned = cell_value.replace('%', '').replace(',', '').replace('$', '').strip()
    
    # Handle parentheses (negative)
    is_negative = cleaned.startswith('(') and cleaned.endswith(')')
    if is_negative:
        cleaned = '-' + cleaned[1:-1]
    
    try:
        value = float(cleaned)
        metadata['is_percent'] = is_percent
        metadata['is_negative'] = is_negative
        metadata['normalized_value'] = value
        
        # Store BOTH forms in node
        return value, 'NUMBER', metadata
    except ValueError:
        return None, 'TEXT', metadata

# In node creation:
self.kg.add_node(
    cell_id,
    value=numeric_value,
    value_percent=numeric_value if is_percent else None,  # ← NEW
    original_format=cell_value,  # ← NEW
    ...
)
```

**In argument retrieval:**
```python
def find_constant(self, target_value: float, allow_percent=True) -> List[Node]:
    """Find nodes matching value, handling % ambiguity"""
    matches = []
    
    for node_id, data in self.kg.nodes(data=True):
        node_value = data.get('value')
        if node_value is None:
            continue
        
        # Direct match
        if abs(node_value - target_value) < 0.01:
            matches.append((node_id, data))
        
        # Percentage match (23.6 vs 0.236)
        if allow_percent and data.get('is_percent'):
            if abs(node_value - target_value) < 0.01:  # 23.6 == 23.6
                matches.append((node_id, data))
            elif abs(node_value/100 - target_value) < 0.0001:  # 23.6/100 == 0.236
                matches.append((node_id, data))
    
    return matches
```

**Expected Impact:** +10-15% accuracy

---

### 1.2 Add Value-Based Index ⭐ HIGH IMPACT

**Problem:**
- Semantic search is slow and unreliable
- No fast lookup by numeric value

**Solution:**
```python
# In structured_kg_builder.py

class StructuredKGBuilder:
    def __init__(self):
        self.kg = nx.MultiDiGraph()
        self.node_counter = 0
        self.value_index = defaultdict(list)  # ← NEW: value → [node_ids]
    
    def _create_cell_node(self, ...):
        cell_id = self._create_node_id("CELL")
        
        # ... create node ...
        
        # Index by value
        if numeric_value is not None:
            # Index normalized value
            self.value_index[numeric_value].append(cell_id)
            
            # Also index percentage forms
            if is_percent:
                self.value_index[numeric_value / 100].append(cell_id)
        
        return cell_id
    
    def find_nodes_by_value(self, target_value: float, tolerance=0.01) -> List[str]:
        """Fast O(1) lookup by value"""
        matches = []
        
        # Exact match
        if target_value in self.value_index:
            matches.extend(self.value_index[target_value])
        
        # Tolerance match (for floating point)
        for value, nodes in self.value_index.items():
            if abs(value - target_value) < tolerance:
                matches.extend(nodes)
        
        return matches
```

**Expected Impact:** +5-10% accuracy, much faster retrieval

---

### 1.3 Handle Scaling Factors (const_XXX) ⭐ MEDIUM IMPACT

**Problem:**
- `const_1000`, `const_1000000` treated as missing constants
- These are unit conversions, not data values

**Solution:**
```python
# In program_executor.py or program_synthesizer.py

SCALING_CONSTANTS = {
    'const_100': 100,
    'const_1000': 1000,
    'const_1000000': 1000000,
    'const_m1': -1,
}

def resolve_argument(arg: str, kg, intermediate_results) -> float:
    """Resolve argument to numeric value"""
    
    # Case 1: Intermediate result
    if arg.startswith('#'):
        ref_idx = int(arg[1:])
        return intermediate_results[ref_idx]
    
    # Case 2: Scaling constant
    if arg in SCALING_CONSTANTS:
        return SCALING_CONSTANTS[arg]
    
    # Case 3: Direct number
    try:
        # Handle percentage
        if arg.endswith('%'):
            return float(arg[:-1])
        return float(arg)
    except ValueError:
        pass
    
    # Case 4: Look up in KG
    nodes = kg.find_nodes_by_value(float(arg))
    if nodes:
        return kg.nodes[nodes[0]]['value']
    
    raise ValueError(f"Cannot resolve argument: {arg}")
```

**Expected Impact:** +5% accuracy

---

### 1.4 Improve Number Extraction from Text ⭐ MEDIUM IMPACT

**Problem:**
- Parentheses notation not always parsed
- Multi-format numbers missed

**Solution:**
```python
def extract_numbers_from_text(text: str) -> List[Dict]:
    """Enhanced number extraction"""
    numbers = []
    
    patterns = [
        # Currency with negatives: $ -23,158 or $(23,158)
        r'\$\s*-?\d+(?:,\d{3})*(?:\.\d+)?|\$\s*\(\s*\d+(?:,\d{3})*(?:\.\d+)?\s*\)',
        
        # Percentages: 23.6% or (23.6%)
        r'-?\d+(?:,\d{3})*(?:\.\d+)?%|\(\s*\d+(?:,\d{3})*(?:\.\d+)?%\s*\)',
        
        # Plain numbers: 1234, 1,234, or (1,234)
        r'-?\d+(?:,\d{3})*(?:\.\d+)?|\(\s*\d+(?:,\d{3})*(?:\.\d+)?\s*\)',
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            original = match.group(0)
            cleaned = original.replace(',', '').replace('$', '').replace('%', '').strip()
            
            # Handle parentheses
            is_negative = '(' in original and ')' in original
            if is_negative:
                cleaned = cleaned.replace('(', '').replace(')', '').strip()
                cleaned = '-' + cleaned if not cleaned.startswith('-') else cleaned
            
            try:
                value = float(cleaned)
                numbers.append({
                    'value': value,
                    'original': original,
                    'position': match.start(),
                    'is_percent': '%' in original,
                    'is_currency': '$' in original
                })
            except ValueError:
                continue
    
    return numbers
```

**Expected Impact:** +3-5% accuracy

---

## PHASE 2: SEMANTIC ENHANCEMENTS (Week 2) - Target: 50-55% Accuracy

### 2.1 Add Temporal Entity Extraction ⭐ MEDIUM IMPACT

**Problem:**
- 10% of questions need temporal context ("in 2018", "Q1 2012")
- Years/periods not extracted as entities

**Solution:**
```python
def extract_temporal_entities(text: str) -> List[Dict]:
    """Extract years, quarters, fiscal periods"""
    temporal = []
    
    patterns = {
        'year': r'\b(19\d{2}|20\d{2})\b',
        'quarter': r'\bQ[1-4]\s+(19\d{2}|20\d{2})\b',
        'fiscal_year': r'\bfiscal\s+(year\s+)?(19\d{2}|20\d{2})\b',
        'month_year': r'\b(January|February|...|December)\s+(19\d{2}|20\d{2})\b',
    }
    
    for entity_type, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            temporal.append({
                'type': entity_type,
                'value': match.group(0),
                'year': int(re.search(r'(19\d{2}|20\d{2})', match.group(0)).group(0)),
                'position': match.start()
            })
    
    return temporal

# In KG builder:
def _add_temporal_entities(self, text_list):
    """Add temporal nodes and link to nearby numbers"""
    for text in text_list:
        temporals = extract_temporal_entities(text)
        numbers = extract_numbers_from_text(text)
        
        for temp in temporals:
            temp_id = self._create_node_id("TIME")
            self.kg.add_node(temp_id, type='temporal', **temp)
            
            # Link to numbers in same sentence
            for num in numbers:
                if abs(num['position'] - temp['position']) < 100:  # within 100 chars
                    num_id = self._find_number_node(num['value'])
                    if num_id:
                        self.kg.add_edge(num_id, temp_id, relation='DURING')
```

**Expected Impact:** +5-8% accuracy

---

### 2.2 Link Text Numbers to Table Cells ⭐ HIGH IMPACT

**Problem:**
- Same value appears in text and table, but disconnected
- No semantic links between mentions

**Solution:**
```python
def _link_text_to_table(self, text_node_ids, table_id):
    """Semantic linking of text mentions to table cells"""
    
    # Get all text numbers
    text_numbers = []
    for text_id in text_node_ids:
        text_content = self.kg.nodes[text_id]['content']
        numbers = extract_numbers_from_text(text_content)
        for num in numbers:
            text_numbers.append({
                'text_node': text_id,
                'value': num['value'],
                'context': text_content
            })
    
    # Get all table cells
    table_cells = []
    for node_id, data in self.kg.nodes(data=True):
        if data.get('type') == 'cell' and data.get('value') is not None:
            table_cells.append((node_id, data))
    
    # Match and link
    for text_num in text_numbers:
        for cell_id, cell_data in table_cells:
            cell_value = cell_data['value']
            
            # Value match with tolerance
            if abs(cell_value - text_num['value']) < 0.01:
                # Create bidirectional link
                self.kg.add_edge(
                    text_num['text_node'], 
                    cell_id, 
                    relation='REFERS_TO_VALUE',
                    matched_value=text_num['value']
                )
                self.kg.add_edge(
                    cell_id,
                    text_num['text_node'],
                    relation='MENTIONED_IN',
                    matched_value=text_num['value']
                )
```

**Expected Impact:** +8-12% accuracy (helps disambiguation)

---

### 2.3 Column-based Filtering ⭐ MEDIUM IMPACT

**Problem:**
- Question asks for "2018 revenue" but multiple revenue values exist
- Need to filter by column/row context

**Solution:**
```python
def find_cell_by_column_and_row_keywords(
    self,
    value: float,
    column_keywords: List[str] = None,
    row_keywords: List[str] = None
) -> Optional[str]:
    """Find cell matching value + context"""
    
    candidates = self.find_nodes_by_value(value)
    
    if not column_keywords and not row_keywords:
        return candidates[0] if candidates else None
    
    # Filter by column
    if column_keywords:
        filtered = []
        for cell_id in candidates:
            col_name = self.kg.nodes[cell_id].get('column_name', '').lower()
            if any(kw.lower() in col_name for kw in column_keywords):
                filtered.append(cell_id)
        candidates = filtered
    
    # Filter by row
    if row_keywords:
        filtered = []
        for cell_id in candidates:
            row_idx = self.kg.nodes[cell_id].get('row_index')
            # Get row data
            row_cells = self.query_cells_by_row(row_idx)
            row_text = ' '.join(str(c.get('value', '')) for c in row_cells)
            
            if any(kw.lower() in row_text.lower() for kw in row_keywords):
                filtered.append(cell_id)
        candidates = filtered
    
    return candidates[0] if candidates else None

# Usage in argument retrieval:
# Question: "What was 2018 revenue?"
# → find_cell_by_column_and_row_keywords(
#       value=9896, 
#       column_keywords=['revenue', 'expense'],
#       row_keywords=['2018']
#   )
```

**Expected Impact:** +5-8% accuracy

---

## PHASE 3: ADVANCED FEATURES (Week 3) - Target: 65-70% Accuracy

### 3.1 Program Execution State Management

**Solution:**
```python
class ProgramExecutor:
    def __init__(self, kg):
        self.kg = kg
        self.intermediate_results = {}  # #0, #1, etc.
        self.execution_trace = []
    
    def execute(self, program: str) -> float:
        """Execute program with state tracking"""
        operations = self._parse_program(program)
        
        for idx, op in enumerate(operations):
            # Resolve arguments
            resolved_args = []
            for arg in op['args']:
                value = self._resolve_argument(arg)
                resolved_args.append(value)
            
            # Execute operation
            result = self._execute_operation(op['op'], resolved_args)
            
            # Store intermediate result
            self.intermediate_results[idx] = result
            self.execution_trace.append({
                'step': idx,
                'operation': op['op'],
                'args': resolved_args,
                'result': result
            })
        
        return result
```

### 3.2 Multi-hop Query Support

**Solution:**
```python
def query_with_path(self, start_keywords: List[str], path: List[str], target_type: str):
    """
    Example: Find "revenue" (start) → "2018" (path) → cell value (target)
    """
    # Start nodes
    start_nodes = []
    for keyword in start_keywords:
        nodes = self._find_nodes_by_keyword(keyword)
        start_nodes.extend(nodes)
    
    # Follow path
    current_nodes = start_nodes
    for relation in path:
        next_nodes = []
        for node in current_nodes:
            neighbors = self.kg.neighbors(node)
            for neighbor in neighbors:
                if self.kg.edges[node, neighbor]['relation'] == relation:
                    next_nodes.append(neighbor)
        current_nodes = next_nodes
    
    # Filter by target type
    targets = [n for n in current_nodes if self.kg.nodes[n]['type'] == target_type]
    return targets
```

---

## IMPLEMENTATION PRIORITY

**Week 1 (Critical):**
1. ✅ Percentage handling (#1.1)
2. ✅ Value index (#1.2)
3. ✅ Scaling factors (#1.3)
4. ✅ Better number extraction (#1.4)

**Week 2 (Important):**
5. ✅ Temporal entities (#2.1)
6. ✅ Text-table linking (#2.2)
7. ✅ Column filtering (#2.3)

**Week 3 (Advanced):**
8. ✅ Execution state (#3.1)
9. ✅ Multi-hop queries (#3.2)

---

## TESTING STRATEGY

After each phase:
1. Run on 50 samples
2. Measure accuracy improvement
3. Analyze new failure modes
4. Adjust priorities

**Metrics to track:**
- Overall accuracy
- Argument retrieval success rate
- Program execution success rate
- Average confidence scores

---

## EXPECTED OUTCOMES

**Current:** 15% accuracy (3/20 correct)

**After Phase 1:** 35-40% (7-8/20)
- Fixed percentage matching
- Fast value lookup
- Scaling factors handled

**After Phase 2:** 50-55% (10-11/20)
- Temporal context
- Better disambiguation
- Text-table links

**After Phase 3:** 65-70% (13-14/20)
- Complex queries
- Multi-step programs
- Edge cases handled

**Remaining 30% issues:**
- Truly ambiguous questions
- Complex reasoning beyond graph
- Data quality issues

---

## CODE FILES TO MODIFY

1. `structured_kg_builder.py` - Core improvements
2. `program_executor.py` - Execution engine
3. `semantic_retriever.py` - Argument retrieval
4. `finqa_intelligent_pipeline.py` - Integration

---

*This plan provides concrete, actionable steps with code examples*  
*Each improvement targets specific failure modes identified in analysis*
