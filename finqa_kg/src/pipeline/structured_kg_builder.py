"""
STRUCTURED KNOWLEDGE GRAPH BUILDER

Thay vì flat graph, build structured graph với table topology:
- TableNode: Đại diện cho table
- RowNode: Đại diện cho mỗi row
- CellNode: Đại diện cho mỗi cell (intersection of row x column)

Relations:
- HAS_ROW: Table → Row
- HAS_CELL: Row → Cell  
- IN_COLUMN: Cell → Column
- SAME_ROW: Cell ↔ Cell
- SAME_COLUMN: Cell ↔ Cell

Benefits:
- Biết entity nào trong row nào, column nào
- Query chính xác dựa trên structure
- Validate constraints dễ dàng
"""

import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
import re


class StructuredKGBuilder:
    """Build structured KG with explicit table topology"""
    
    def __init__(self):
        self.node_counter = 0
        self.kg = nx.MultiDiGraph()
        
    def _create_node_id(self, prefix: str) -> str:
        """Generate unique node ID"""
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"
    
    def build_from_sample(self, sample: Dict[str, Any]) -> nx.MultiDiGraph:
        """
        Build structured KG from FinQA sample
        
        Args:
            sample: Dict with 'table', 'pre_text', 'post_text', 'qa'
            
        Returns:
            Structured knowledge graph
        """
        print("\n[StructuredKG] Building...")
        
        # Reset
        self.kg = nx.MultiDiGraph()
        self.node_counter = 0
        
        # Step 1: Build table structure
        table_data = sample.get('table', [])
        if table_data:
            table_node = self._build_table_structure(table_data)
            print(f"  ✓ Table built: {len(table_data)} rows")
        else:
            table_node = None
            print(f"  ⚠ No table data")
        
        # Step 2: Add text entities (pre_text, post_text)
        text_entities = self._extract_text_entities(
            sample.get('pre_text', []),
            sample.get('post_text', [])
        )
        print(f"  ✓ Text entities: {len(text_entities)}")
        
        # Step 3: Link text entities to table cells
        if table_node:
            self._link_text_to_table(text_entities, table_node)
            print(f"  ✓ Linked text to table")
        
        print(f"  ✓ KG complete: {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges")
        
        return self.kg
    
    def _build_table_structure(self, table_data: List[List[str]]) -> str:
        """
        Build explicit table structure
        
        Structure:
          TableNode
            ├─ RowNode_0 (header)
            │   ├─ CellNode (col=0, value="Year")
            │   ├─ CellNode (col=1, value="2009")
            │   └─ CellNode (col=2, value="2010")
            ├─ RowNode_1 (data)
            │   ├─ CellNode (col=0, value="Revenue")
            │   ├─ CellNode (col=1, value=100)
            │   └─ CellNode (col=2, value=150)
            ...
        
        Returns:
            Table node ID
        """
        if not table_data or len(table_data) == 0:
            return None
        
        # Create table node
        table_id = self._create_node_id("TABLE")
        self.kg.add_node(table_id, type='table', label='TABLE')
        
        # Extract header (first row is usually header)
        header = table_data[0] if table_data else []
        header_normalized = [self._normalize_header(h) for h in header]
        
        # Create row nodes
        for row_idx, row_data in enumerate(table_data):
            row_id = self._create_node_id("ROW")
            is_header = (row_idx == 0)
            
            self.kg.add_node(
                row_id,
                type='row',
                label='ROW',
                row_index=row_idx,
                is_header=is_header,
                row_data=row_data  # Store raw data
            )
            
            # Link: Table HAS_ROW Row
            self.kg.add_edge(table_id, row_id, relation='HAS_ROW')
            
            # Create cell nodes for this row
            for col_idx, cell_value in enumerate(row_data):
                cell_id = self._create_cell_node(
                    row_id=row_id,
                    row_idx=row_idx,
                    col_idx=col_idx,
                    cell_value=cell_value,
                    column_name=header_normalized[col_idx] if col_idx < len(header_normalized) else f"col_{col_idx}",
                    is_header=is_header
                )
        
        return table_id
    
    def _create_cell_node(self,
                          row_id: str,
                          row_idx: int,
                          col_idx: int,
                          cell_value: str,
                          column_name: str,
                          is_header: bool) -> str:
        """Create a cell node with rich metadata"""
        
        cell_id = self._create_node_id("CELL")
        
        # Parse value (number or text)
        numeric_value, label = self._parse_cell_value(cell_value)
        
        # Build context string
        context = f"table[{row_idx},{col_idx}]: column={column_name}, value={cell_value}"
        
        # Add node
        self.kg.add_node(
            cell_id,
            type='cell',
            label=label,
            row_index=row_idx,
            col_index=col_idx,
            column_name=column_name,
            raw_value=cell_value,
            value=numeric_value if numeric_value is not None else cell_value,
            is_header=is_header,
            context=context
        )
        
        # Link: Row HAS_CELL Cell
        self.kg.add_edge(row_id, cell_id, relation='HAS_CELL')
        
        # Link cells in same row
        for other_cell_id in self.kg.nodes():
            if (self.kg.nodes[other_cell_id].get('type') == 'cell' and
                self.kg.nodes[other_cell_id].get('row_index') == row_idx and
                other_cell_id != cell_id):
                self.kg.add_edge(cell_id, other_cell_id, relation='SAME_ROW')
        
        # Link cells in same column
        for other_cell_id in self.kg.nodes():
            if (self.kg.nodes[other_cell_id].get('type') == 'cell' and
                self.kg.nodes[other_cell_id].get('col_index') == col_idx and
                other_cell_id != cell_id):
                self.kg.add_edge(cell_id, other_cell_id, relation='SAME_COLUMN')
        
        return cell_id
    
    def _parse_cell_value(self, cell_value: str) -> Tuple[Optional[float], str]:
        """
        Parse cell value to extract number and determine label
        
        Returns:
            (numeric_value, label)
        """
        cell_value = cell_value.strip()
        
        if not cell_value or cell_value == '-' or cell_value.lower() in ['n/a', 'na', '']:
            return None, 'TEXT'
        
        # Remove common formatting
        cleaned = cell_value.replace(',', '').replace('$', '').replace('%', '').strip()
        
        # Try to parse as number
        try:
            # Handle parentheses (negative numbers)
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]
            
            value = float(cleaned)
            
            # Determine label based on original format
            if '$' in cell_value:
                return value, 'MONEY'
            elif '%' in cell_value:
                return value, 'PERCENT'
            else:
                return value, 'NUMBER'
                
        except ValueError:
            # Not a number
            return None, 'TEXT'
    
    def _normalize_header(self, header: str) -> str:
        """Normalize column header"""
        # Remove special chars, lowercase
        normalized = re.sub(r'[^a-z0-9\s]', '', header.lower())
        normalized = normalized.strip().replace(' ', '_')
        return normalized if normalized else 'unnamed'
    
    def _extract_text_entities(self,
                                pre_text: List[str],
                                post_text: List[str]) -> List[str]:
        """Extract entities from text (simplified - just store text for now)"""
        
        # For now, just create text nodes
        # Later can enhance with NER
        text_node_ids = []
        
        for text in pre_text + post_text:
            if text.strip():
                text_id = self._create_node_id("TEXT")
                self.kg.add_node(
                    text_id,
                    type='text',
                    label='TEXT',
                    content=text
                )
                text_node_ids.append(text_id)
        
        return text_node_ids
    
    def _link_text_to_table(self, text_node_ids: List[str], table_id: str):
        """Link text entities to table cells (future enhancement)"""
        # For now, just create general links
        # Later can add semantic matching
        for text_id in text_node_ids:
            self.kg.add_edge(text_id, table_id, relation='REFERS_TO')
    
    def query_cells_by_row(self, row_index: int) -> List[Dict]:
        """Get all cells in a specific row"""
        cells = []
        for node_id, data in self.kg.nodes(data=True):
            if (data.get('type') == 'cell' and
                data.get('row_index') == row_index):
                cells.append({
                    'node_id': node_id,
                    **data
                })
        return cells
    
    def query_cells_by_column(self, column_name: str) -> List[Dict]:
        """Get all cells in a specific column"""
        cells = []
        for node_id, data in self.kg.nodes(data=True):
            if (data.get('type') == 'cell' and
                data.get('column_name') == column_name):
                cells.append({
                    'node_id': node_id,
                    **data
                })
        return cells
    
    def query_cells_by_value_in_column(self,
                                       column_name: str,
                                       value_keywords: List[str]) -> List[Dict]:
        """
        Find cells in a specific column that match keywords
        
        Example:
            column_name = "item"
            value_keywords = ["available", "sale", "investments"]
            → Returns cells with "available-for-sale investments"
        """
        column_cells = self.query_cells_by_column(column_name)
        
        matched_cells = []
        for cell in column_cells:
            raw_value = cell.get('raw_value', '').lower()
            # Check if all keywords present
            if all(kw.lower() in raw_value for kw in value_keywords):
                matched_cells.append(cell)
        
        return matched_cells
    
    def get_cell_neighbors_in_row(self, cell_node_id: str) -> List[Dict]:
        """Get all other cells in the same row as given cell"""
        cell_data = self.kg.nodes[cell_node_id]
        row_idx = cell_data.get('row_index')
        
        if row_idx is None:
            return []
        
        return self.query_cells_by_row(row_idx)
