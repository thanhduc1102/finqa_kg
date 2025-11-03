"""
Intelligent Knowledge Graph Builder
Xây dựng KG từ text + table sử dụng NLP thực sự:
- Entity extraction với NER
- Relation extraction với dependency parsing
- Text-table semantic linking
"""

import networkx as nx
import spacy
from typing import Dict, List, Tuple, Any, Optional
import re
from dataclasses import dataclass
import numpy as np

@dataclass
class Entity:
    """Entity được extract từ text hoặc table"""
    text: str
    label: str  # ORG, PERSON, MONEY, DATE, PERCENT, etc
    start: int
    end: int
    value: Optional[float] = None  # Nếu là số
    context: str = ""
    source_type: str = ""  # 'text' or 'table'
    source_id: str = ""

@dataclass
class Relation:
    """Relation giữa 2 entities"""
    head: str  # entity ID
    relation: str  # relation type
    tail: str  # entity ID
    confidence: float = 1.0

class IntelligentKGBuilder:
    """
    Build Knowledge Graph với NLP:
    1. Extract entities từ text và table
    2. Extract relations qua dependency parsing
    3. Link text và table qua semantic similarity
    4. Tạo graph structure với NetworkX
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize với Spacy models
        """
        print("Loading Spacy model...")
        try:
            # Load transformer-based model (tốt nhất)
            self.nlp = spacy.load("en_core_web_trf")
            print("✓ Loaded en_core_web_trf")
        except:
            try:
                # Fallback to large model
                self.nlp = spacy.load("en_core_web_lg")
                print("✓ Loaded en_core_web_lg")
            except:
                # Fallback to small model
                self.nlp = spacy.load("en_core_web_sm")
                print("⚠ Using en_core_web_sm (less accurate)")
        
        self.entity_counter = 0
        self.node_counter = 0
        
    def build_kg(self, sample: Dict[str, Any]) -> nx.MultiDiGraph:
        """
        Main function: Build KG từ một sample
        
        Args:
            sample: Dict với keys: pre_text, post_text, table, qa
            
        Returns:
            NetworkX MultiDiGraph
        """
        kg = nx.MultiDiGraph()
        
        # Document root node
        doc_id = sample.get('id', f'doc_{self.node_counter}')
        self.node_counter += 1
        
        kg.add_node(doc_id, 
                   type='document',
                   filename=sample.get('filename', ''))
        
        # Step 1: Process text và extract entities
        print("  [1/5] Processing text...")
        text_entities, text_relations, text_nodes = self._process_text(
            sample.get('pre_text', []),
            sample.get('post_text', []),
            kg, doc_id
        )
        
        # Step 2: Process table và extract entities
        print("  [2/5] Processing table...")
        table_entities, table_nodes = self._process_table(
            sample.get('table', []),
            kg, doc_id
        )
        
        # Step 3: Link text và table qua semantic similarity
        print("  [3/5] Linking text and table...")
        self._link_text_table(
            text_entities, table_entities,
            text_nodes, table_nodes,
            kg
        )
        
        # Step 4: Add intra-text relations
        print("  [4/5] Adding text relations...")
        for rel in text_relations:
            if rel.head in kg.nodes and rel.tail in kg.nodes:
                kg.add_edge(rel.head, rel.tail,
                           relation=rel.relation,
                           confidence=rel.confidence)
        
        # Step 5: Process QA if exists
        print("  [5/5] Processing QA...")
        if 'qa' in sample and sample['qa']:
            self._process_qa(sample['qa'], kg, doc_id)
        
        print(f"  ✓ KG built: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
        return kg
    
    def _process_text(self, pre_text: List[str], post_text: List[str], 
                     kg: nx.MultiDiGraph, doc_id: str) -> Tuple[List[Entity], List[Relation], List[str]]:
        """
        Process text với Spacy:
        - NER để extract entities
        - Dependency parsing để extract relations
        """
        all_entities = []
        all_relations = []
        text_nodes = []
        
        all_texts = [(t, 'pre', idx) for idx, t in enumerate(pre_text)] + \
                    [(t, 'post', idx) for idx, t in enumerate(post_text)]
        
        for text, position, idx in all_texts:
            # Tạo text node
            text_node_id = f"text_{position}_{idx}"
            text_nodes.append(text_node_id)
            
            kg.add_node(text_node_id,
                       type='text',
                       content=text,
                       position=position,
                       index=idx)
            kg.add_edge(doc_id, text_node_id, relation=f'has_{position}_text')
            
            # Process với Spacy
            doc = self.nlp(text)
            
            # Extract entities
            for ent in doc.ents:
                entity_id = f"entity_{self.entity_counter}"
                self.entity_counter += 1
                
                # Extract numerical value nếu có
                num_value = self._extract_number(ent.text)
                
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    value=num_value,
                    context=text[max(0, ent.start_char-50):min(len(text), ent.end_char+50)],
                    source_type='text',
                    source_id=text_node_id
                )
                all_entities.append(entity)
                
                # Add entity node
                kg.add_node(entity_id,
                           type='entity',
                           text=ent.text,
                           label=ent.label_,
                           value=num_value,
                           context=entity.context)
                
                kg.add_edge(text_node_id, entity_id,
                           relation='contains_entity',
                           start=ent.start_char,
                           end=ent.end_char)
            
            # Extract relations qua dependency parsing
            for token in doc:
                # Nếu token có dependency với subject/object
                if token.dep_ in ['nsubj', 'dobj', 'pobj'] and token.head != token:
                    head_text = token.head.text
                    tail_text = token.text
                    relation_type = token.dep_
                    
                    # Tìm entity nodes tương ứng
                    head_ids = [e_id for e_id in kg.nodes() 
                               if kg.nodes[e_id].get('type') == 'entity' 
                               and kg.nodes[e_id].get('text', '').lower() == head_text.lower()]
                    tail_ids = [e_id for e_id in kg.nodes() 
                               if kg.nodes[e_id].get('type') == 'entity'
                               and kg.nodes[e_id].get('text', '').lower() == tail_text.lower()]
                    
                    for h_id in head_ids:
                        for t_id in tail_ids:
                            all_relations.append(Relation(
                                head=h_id,
                                relation=relation_type,
                                tail=t_id,
                                confidence=0.8
                            ))
        
        return all_entities, all_relations, text_nodes
    
    def _process_table(self, table: List[List[str]], 
                      kg: nx.MultiDiGraph, doc_id: str) -> Tuple[List[Entity], List[str]]:
        """
        Process table:
        - Extract table structure
        - Extract entities từ cells
        - Tạo header-cell relationships
        """
        table_entities = []
        table_nodes = []
        
        if not table or len(table) == 0:
            return table_entities, table_nodes
        
        # Table node
        table_id = f"table_{self.node_counter}"
        self.node_counter += 1
        table_nodes.append(table_id)
        
        headers = table[0] if len(table) > 0 else []
        
        kg.add_node(table_id,
                   type='table',
                   headers=headers,
                   num_rows=len(table)-1,
                   num_cols=len(headers))
        kg.add_edge(doc_id, table_id, relation='has_table')
        
        # Process each cell
        for row_idx, row in enumerate(table[1:], start=1):
            row_label = row[0] if len(row) > 0 else ""
            
            for col_idx, cell_value in enumerate(row):
                cell_id = f"cell_{row_idx}_{col_idx}"
                
                # Extract number và entity
                num_value = self._extract_number(str(cell_value))
                
                # Process cell với Spacy để extract entities
                doc = self.nlp(str(cell_value))
                
                cell_entities = []
                for ent in doc.ents:
                    entity_id = f"entity_{self.entity_counter}"
                    self.entity_counter += 1
                    
                    entity = Entity(
                        text=ent.text,
                        label=ent.label_,
                        start=0,
                        end=len(ent.text),
                        value=self._extract_number(ent.text),
                        context=f"Table[{row_idx},{col_idx}]: {headers[col_idx] if col_idx < len(headers) else ''} = {cell_value}",
                        source_type='table',
                        source_id=cell_id
                    )
                    table_entities.append(entity)
                    cell_entities.append(entity_id)
                    
                    # Add entity node
                    kg.add_node(entity_id,
                               type='entity',
                               text=ent.text,
                               label=ent.label_,
                               value=entity.value,
                               context=entity.context)
                
                # Add cell node
                kg.add_node(cell_id,
                           type='cell',
                           value=cell_value,
                           number=num_value,
                           row=row_idx,
                           col=col_idx,
                           header=headers[col_idx] if col_idx < len(headers) else '',
                           row_label=row_label)
                
                kg.add_edge(table_id, cell_id,
                           relation='has_cell',
                           row=row_idx,
                           col=col_idx)
                
                # Link cell to entities
                for ent_id in cell_entities:
                    kg.add_edge(cell_id, ent_id,
                               relation='contains_entity')
        
        return table_entities, table_nodes
    
    def _link_text_table(self, text_entities: List[Entity], table_entities: List[Entity],
                        text_nodes: List[str], table_nodes: List[str],
                        kg: nx.MultiDiGraph):
        """
        Link text và table qua:
        - Exact match entities
        - Number matching
        - Semantic similarity của context
        """
        # Exact entity match
        for t_ent in text_entities:
            for tab_ent in table_entities:
                # Match by text hoặc number
                if t_ent.text.lower() == tab_ent.text.lower():
                    # Tìm node IDs
                    text_ent_ids = [n for n in kg.nodes() 
                                   if kg.nodes[n].get('type') == 'entity'
                                   and kg.nodes[n].get('text', '').lower() == t_ent.text.lower()
                                   and kg.nodes[n].get('context', '').find(t_ent.source_id) >= 0]
                    
                    table_ent_ids = [n for n in kg.nodes()
                                    if kg.nodes[n].get('type') == 'entity'
                                    and kg.nodes[n].get('text', '').lower() == tab_ent.text.lower()
                                    and kg.nodes[n].get('context', '').find(tab_ent.source_id) >= 0]
                    
                    for t_id in text_ent_ids:
                        for tab_id in table_ent_ids:
                            kg.add_edge(t_id, tab_id,
                                       relation='same_entity',
                                       confidence=1.0)
                
                # Match by value
                elif (t_ent.value is not None and tab_ent.value is not None 
                      and abs(t_ent.value - tab_ent.value) < 0.01):
                    text_ent_ids = [n for n in kg.nodes() 
                                   if kg.nodes[n].get('type') == 'entity'
                                   and kg.nodes[n].get('value') == t_ent.value]
                    
                    table_ent_ids = [n for n in kg.nodes()
                                    if kg.nodes[n].get('type') == 'entity'
                                    and kg.nodes[n].get('value') == tab_ent.value]
                    
                    for t_id in text_ent_ids[:1]:  # Limit to avoid too many edges
                        for tab_id in table_ent_ids[:1]:
                            kg.add_edge(t_id, tab_id,
                                       relation='same_value',
                                       confidence=0.9)
    
    def _process_qa(self, qa: Dict[str, Any], kg: nx.MultiDiGraph, doc_id: str):
        """Process QA pair và add vào KG"""
        qa_node = f"qa_{self.node_counter}"
        self.node_counter += 1
        
        kg.add_node(qa_node,
                   type='qa',
                   question=qa.get('question', ''),
                   answer=qa.get('answer', ''),
                   program=qa.get('program', ''),
                   exe_ans=qa.get('exe_ans', None))
        
        kg.add_edge(doc_id, qa_node, relation='has_qa')
        
        # Extract entities từ question
        question = qa.get('question', '')
        if question:
            doc = self.nlp(question)
            for ent in doc.ents:
                # Link question entities với document entities
                matching_nodes = [n for n in kg.nodes()
                                 if kg.nodes[n].get('type') == 'entity'
                                 and kg.nodes[n].get('text', '').lower() == ent.text.lower()]
                
                for match_id in matching_nodes:
                    kg.add_edge(qa_node, match_id,
                               relation='mentions_entity',
                               entity_text=ent.text)
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract number từ text"""
        if not isinstance(text, str):
            try:
                return float(text)
            except:
                return None
        
        # Remove common symbols
        clean = text.replace('$', '').replace(',', '').replace('%', '').strip()
        clean = re.sub(r'\(.*?\)', '', clean).strip()  # Remove parentheses
        
        # Extract first number
        numbers = re.findall(r'-?\d+\.?\d*', clean)
        if numbers:
            try:
                return float(numbers[0])
            except:
                pass
        
        return None
    
    def get_entity_index(self, kg: nx.MultiDiGraph) -> Dict[str, List[Dict]]:
        """
        Tạo index của entities theo text và value để lookup nhanh
        """
        entity_index = {
            'by_text': {},
            'by_value': {},
            'by_label': {}
        }
        
        for node_id, data in kg.nodes(data=True):
            if data.get('type') == 'entity':
                text = data.get('text', '').lower()
                value = data.get('value')
                label = data.get('label', '')
                
                # Index by text
                if text not in entity_index['by_text']:
                    entity_index['by_text'][text] = []
                entity_index['by_text'][text].append({
                    'id': node_id,
                    'data': data
                })
                
                # Index by value
                if value is not None:
                    if value not in entity_index['by_value']:
                        entity_index['by_value'][value] = []
                    entity_index['by_value'][value].append({
                        'id': node_id,
                        'data': data
                    })
                
                # Index by label
                if label not in entity_index['by_label']:
                    entity_index['by_label'][label] = []
                entity_index['by_label'][label].append({
                    'id': node_id,
                    'data': data
                })
        
        return entity_index
