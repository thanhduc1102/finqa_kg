"""
KNOWLEDGE GRAPH QUALITY ANALYSIS
=================================

Script toàn diện để đánh giá chất lượng Knowledge Graph:
1. Entity Extraction Quality (số, ngày tháng, NER)
2. Relation Extraction Quality
3. Table Structure Coverage
4. Text-to-Table Integration
5. Information Completeness (có đủ thông tin để trả lời câu hỏi không?)
6. Retrieval Accuracy (có tìm đúng entities cần thiết không?)

Chạy trên 50 samples để có đánh giá thống kê đáng tin cậy.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import re
import pandas as pd
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.intelligent_kg_builder import IntelligentKGBuilder
from src.pipeline.structured_kg_builder import StructuredKGBuilder
from src.visualization.graph_visualizer import GraphVisualizer
import networkx as nx


class KGQualityAnalyzer:
    """Phân tích chất lượng Knowledge Graph"""
    
    def __init__(self):
        self.results = []
        self.visualizer = GraphVisualizer()
        
    def analyze_sample(self, sample: Dict[str, Any], kg: nx.MultiDiGraph, sample_idx: int) -> Dict[str, Any]:
        """
        Phân tích toàn diện một sample
        
        Returns:
            Dict với các metrics đánh giá
        """
        analysis = {
            'sample_id': sample.get('id', f'sample_{sample_idx}'),
            'sample_idx': sample_idx,
            'filename': sample.get('filename', 'N/A'),
        }
        
        # 1. Phân tích Entity Extraction
        analysis['entity_metrics'] = self._analyze_entities(sample, kg)
        
        # 2. Phân tích Table Structure
        analysis['table_metrics'] = self._analyze_table_structure(sample, kg)
        
        # 3. Phân tích Text Processing
        analysis['text_metrics'] = self._analyze_text_processing(sample, kg)
        
        # 4. Phân tích Relations
        analysis['relation_metrics'] = self._analyze_relations(kg)
        
        # 5. Phân tích Coverage (có đủ thông tin không?)
        analysis['coverage_metrics'] = self._analyze_coverage(sample, kg)
        
        # 6. KG Statistics
        analysis['kg_stats'] = {
            'total_nodes': kg.number_of_nodes(),
            'total_edges': kg.number_of_edges(),
            'node_types': self._count_node_types(kg),
            'edge_types': self._count_edge_types(kg),
        }
        
        return analysis
    
    def _analyze_entities(self, sample: Dict[str, Any], kg: nx.MultiDiGraph) -> Dict[str, Any]:
        """
        Đánh giá Entity Extraction
        
        Kiểm tra:
        - Có extract đủ số không? (so với ground truth)
        - Có extract đúng loại entities không? (ORG, DATE, MONEY, etc)
        - Có deduplicate entities không?
        """
        metrics = {}
        
        # Extract ground truth numbers từ QA
        qa = sample.get('qa', {})
        gt_program = qa.get('program', '')
        gt_numbers = self._extract_numbers_from_program(gt_program)
        
        # Extract numbers from KG
        kg_numbers = []
        kg_entities = []
        
        for node_id, node_data in kg.nodes(data=True):
            if node_data.get('type') == 'entity':
                kg_entities.append(node_data)
                if 'value' in node_data and node_data['value'] is not None:
                    kg_numbers.append(node_data['value'])
        
        # Metrics
        metrics['total_entities'] = len(kg_entities)
        metrics['entities_with_values'] = len(kg_numbers)
        metrics['gt_numbers_count'] = len(gt_numbers)
        metrics['kg_numbers_count'] = len(kg_numbers)
        
        # Coverage: Bao nhiêu % GT numbers có trong KG?
        matched_numbers = 0
        for gt_num in gt_numbers:
            if any(abs(kg_num - gt_num) < 0.01 for kg_num in kg_numbers):
                matched_numbers += 1
        
        metrics['number_coverage'] = matched_numbers / len(gt_numbers) if gt_numbers else 0.0
        metrics['matched_numbers'] = matched_numbers
        
        # Entity type distribution
        entity_types = Counter(e.get('label', 'UNKNOWN') for e in kg_entities)
        metrics['entity_types'] = dict(entity_types)
        
        # Deduplication check (có bao nhiêu entities trùng text?)
        entity_texts = [e.get('text', '').strip().lower() for e in kg_entities]
        unique_texts = set(entity_texts)
        metrics['unique_entity_texts'] = len(unique_texts)
        metrics['duplicate_ratio'] = 1.0 - (len(unique_texts) / len(entity_texts)) if entity_texts else 0.0
        
        return metrics
    
    def _analyze_table_structure(self, sample: Dict[str, Any], kg: nx.MultiDiGraph) -> Dict[str, Any]:
        """
        Đánh giá Table Structure trong KG
        
        Kiểm tra:
        - Có build đúng structure TABLE -> ROW -> CELL không?
        - Có preserve table topology không?
        - Cells có đầy đủ metadata không?
        """
        metrics = {}
        
        table_data = sample.get('table', [])
        metrics['gt_table_rows'] = len(table_data)
        metrics['gt_table_cells'] = sum(len(row) for row in table_data) if table_data else 0
        
        # Count KG table nodes
        table_nodes = [n for n, d in kg.nodes(data=True) if d.get('type') == 'table']
        row_nodes = [n for n, d in kg.nodes(data=True) if d.get('type') == 'row']
        cell_nodes = [n for n, d in kg.nodes(data=True) if d.get('type') == 'cell']
        
        metrics['kg_table_nodes'] = len(table_nodes)
        metrics['kg_row_nodes'] = len(row_nodes)
        metrics['kg_cell_nodes'] = len(cell_nodes)
        
        # Row coverage
        metrics['row_coverage'] = len(row_nodes) / metrics['gt_table_rows'] if metrics['gt_table_rows'] > 0 else 0.0
        
        # Cell coverage
        metrics['cell_coverage'] = len(cell_nodes) / metrics['gt_table_cells'] if metrics['gt_table_cells'] > 0 else 0.0
        
        # Check structure completeness
        has_proper_structure = False
        if table_nodes:
            table_id = table_nodes[0]
            # Check if table has rows
            has_rows = any(kg.has_edge(table_id, row_id) for row_id in row_nodes)
            # Check if rows have cells
            has_cells = any(kg.has_edge(row_id, cell_id) for row_id in row_nodes for cell_id in cell_nodes)
            has_proper_structure = has_rows and has_cells
        
        metrics['has_proper_structure'] = has_proper_structure
        
        # Cell metadata completeness
        cells_with_position = sum(1 for n, d in kg.nodes(data=True) 
                                   if d.get('type') == 'cell' and 'row_index' in d and 'col_index' in d)
        cells_with_column_name = sum(1 for n, d in kg.nodes(data=True) 
                                      if d.get('type') == 'cell' and 'column_name' in d)
        
        metrics['cells_with_position'] = cells_with_position
        metrics['cells_with_column_name'] = cells_with_column_name
        
        if cell_nodes:
            metrics['position_coverage'] = cells_with_position / len(cell_nodes)
            metrics['column_name_coverage'] = cells_with_column_name / len(cell_nodes)
        else:
            metrics['position_coverage'] = 0.0
            metrics['column_name_coverage'] = 0.0
        
        return metrics
    
    def _analyze_text_processing(self, sample: Dict[str, Any], kg: nx.MultiDiGraph) -> Dict[str, Any]:
        """
        Đánh giá Text Processing
        
        Kiểm tra:
        - Có process pre_text và post_text không?
        - Có extract entities từ text không?
        - Có link text với table không?
        """
        metrics = {}
        
        pre_text = sample.get('pre_text', [])
        post_text = sample.get('post_text', [])
        
        metrics['gt_pre_text_count'] = len(pre_text)
        metrics['gt_post_text_count'] = len(post_text)
        metrics['gt_total_text_count'] = len(pre_text) + len(post_text)
        
        # Count text nodes in KG
        text_nodes = [n for n, d in kg.nodes(data=True) if d.get('type') == 'text']
        metrics['kg_text_nodes'] = len(text_nodes)
        
        metrics['text_coverage'] = (len(text_nodes) / metrics['gt_total_text_count'] 
                                    if metrics['gt_total_text_count'] > 0 else 0.0)
        
        # Check for text-to-table links
        text_table_edges = 0
        for node_id, node_data in kg.nodes(data=True):
            if node_data.get('type') == 'text':
                # Count edges from text to table/cell
                for neighbor in kg.neighbors(node_id):
                    neighbor_type = kg.nodes[neighbor].get('type', '')
                    if neighbor_type in ['table', 'cell', 'row']:
                        text_table_edges += 1
                        break  # Count each text node once
        
        metrics['text_to_table_links'] = text_table_edges
        metrics['text_to_table_link_ratio'] = (text_table_edges / len(text_nodes) 
                                                if text_nodes else 0.0)
        
        return metrics
    
    def _analyze_relations(self, kg: nx.MultiDiGraph) -> Dict[str, Any]:
        """
        Đánh giá Relations
        
        Kiểm tra:
        - Có bao nhiêu loại relations?
        - Relations có meaningful không?
        - Có bidirectional relations không?
        """
        metrics = {}
        
        # Count edge types
        edge_types = Counter()
        for u, v, data in kg.edges(data=True):
            rel_type = data.get('relation', 'UNKNOWN')
            edge_types[rel_type] += 1
        
        metrics['total_relations'] = kg.number_of_edges()
        metrics['unique_relation_types'] = len(edge_types)
        metrics['relation_type_distribution'] = dict(edge_types)
        
        # Check for important relation types
        important_relations = ['HAS_ROW', 'HAS_CELL', 'IN_COLUMN', 'CONTAINS', 'RELATED_TO']
        metrics['has_important_relations'] = {
            rel: rel in edge_types for rel in important_relations
        }
        
        # Average edges per node
        metrics['avg_edges_per_node'] = (kg.number_of_edges() / kg.number_of_nodes() 
                                         if kg.number_of_nodes() > 0 else 0.0)
        
        return metrics
    
    def _analyze_coverage(self, sample: Dict[str, Any], kg: nx.MultiDiGraph) -> Dict[str, Any]:
        """
        Đánh giá Coverage: KG có đủ thông tin để trả lời câu hỏi không?
        
        Đây là metric quan trọng nhất!
        """
        metrics = {}
        
        qa = sample.get('qa', {})
        question = qa.get('question', '')
        program = qa.get('program', '')
        
        # Extract required numbers from program
        required_numbers = self._extract_numbers_from_program(program)
        
        # Extract available numbers from KG
        kg_numbers = []
        for node_id, node_data in kg.nodes(data=True):
            if 'value' in node_data and node_data['value'] is not None:
                kg_numbers.append(node_data['value'])
        
        # Check coverage
        found_numbers = []
        missing_numbers = []
        
        for req_num in required_numbers:
            found = False
            for kg_num in kg_numbers:
                if abs(kg_num - req_num) < 0.01:  # Tolerance for floating point
                    found = True
                    found_numbers.append(req_num)
                    break
            if not found:
                missing_numbers.append(req_num)
        
        metrics['required_numbers_count'] = len(required_numbers)
        metrics['found_numbers_count'] = len(found_numbers)
        metrics['missing_numbers_count'] = len(missing_numbers)
        metrics['number_coverage_rate'] = (len(found_numbers) / len(required_numbers) 
                                           if required_numbers else 1.0)
        metrics['missing_numbers'] = missing_numbers
        
        # Extract keywords from question
        question_keywords = self._extract_keywords(question)
        
        # Check if keywords appear in KG
        kg_text_content = []
        for node_id, node_data in kg.nodes(data=True):
            if 'text' in node_data:
                kg_text_content.append(node_data['text'].lower())
            if 'content' in node_data:
                kg_text_content.append(node_data['content'].lower())
        
        kg_text_combined = ' '.join(kg_text_content)
        
        found_keywords = []
        for keyword in question_keywords:
            if keyword.lower() in kg_text_combined:
                found_keywords.append(keyword)
        
        metrics['question_keywords'] = question_keywords
        metrics['found_keywords'] = found_keywords
        metrics['keyword_coverage_rate'] = (len(found_keywords) / len(question_keywords) 
                                            if question_keywords else 1.0)
        
        # Overall coverage score
        metrics['overall_coverage_score'] = (
            metrics['number_coverage_rate'] * 0.7 +  # Numbers are most important
            metrics['keyword_coverage_rate'] * 0.3   # Keywords are supporting
        )
        
        return metrics
    
    def _extract_numbers_from_program(self, program: str) -> List[float]:
        """Extract numbers from program string"""
        if not program:
            return []
        
        numbers = []
        # Pattern: const_X or const_X.X
        const_pattern = r'const_(-?\d+(?:\.\d+)?)'
        matches = re.findall(const_pattern, program)
        
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract important keywords from question"""
        # Remove common words
        stop_words = {'what', 'is', 'the', 'was', 'were', 'are', 'in', 'of', 'for', 'to', 'a', 'an', 
                     'and', 'or', 'but', 'on', 'at', 'by', 'from', 'with', 'as', 'be', 'been', 'being',
                     'how', 'many', 'much', 'did', 'do', 'does', 'would', 'could', 'should', '?'}
        
        words = question.lower().split()
        keywords = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:') not in stop_words and len(w) > 2]
        
        return keywords
    
    def _count_node_types(self, kg: nx.MultiDiGraph) -> Dict[str, int]:
        """Count nodes by type"""
        type_counts = Counter()
        for node_id, node_data in kg.nodes(data=True):
            node_type = node_data.get('type', 'UNKNOWN')
            type_counts[node_type] += 1
        return dict(type_counts)
    
    def _count_edge_types(self, kg: nx.MultiDiGraph) -> Dict[str, int]:
        """Count edges by type"""
        type_counts = Counter()
        for u, v, data in kg.edges(data=True):
            edge_type = data.get('relation', 'UNKNOWN')
            type_counts[edge_type] += 1
        return dict(type_counts)
    
    def generate_report(self, all_analyses: List[Dict[str, Any]], output_path: Path):
        """
        Tạo báo cáo tổng hợp
        """
        report = []
        report.append("="*100)
        report.append("KNOWLEDGE GRAPH QUALITY ANALYSIS REPORT")
        report.append("="*100)
        report.append(f"Analyzed {len(all_analyses)} samples")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 1. Entity Extraction Summary
        report.append("\n" + "="*100)
        report.append("1. ENTITY EXTRACTION QUALITY")
        report.append("="*100)
        
        avg_entities = sum(a['entity_metrics']['total_entities'] for a in all_analyses) / len(all_analyses)
        avg_number_coverage = sum(a['entity_metrics']['number_coverage'] for a in all_analyses) / len(all_analyses)
        avg_duplicate_ratio = sum(a['entity_metrics']['duplicate_ratio'] for a in all_analyses) / len(all_analyses)
        
        report.append(f"Average entities per sample: {avg_entities:.1f}")
        report.append(f"Average number coverage: {avg_number_coverage:.2%}")
        report.append(f"Average duplicate ratio: {avg_duplicate_ratio:.2%}")
        
        # Entity type distribution
        all_entity_types = Counter()
        for a in all_analyses:
            for entity_type, count in a['entity_metrics']['entity_types'].items():
                all_entity_types[entity_type] += count
        
        report.append(f"\nEntity Type Distribution:")
        for entity_type, count in all_entity_types.most_common():
            report.append(f"  {entity_type}: {count}")
        
        # 2. Table Structure Summary
        report.append("\n" + "="*100)
        report.append("2. TABLE STRUCTURE QUALITY")
        report.append("="*100)
        
        avg_row_coverage = sum(a['table_metrics']['row_coverage'] for a in all_analyses) / len(all_analyses)
        avg_cell_coverage = sum(a['table_metrics']['cell_coverage'] for a in all_analyses) / len(all_analyses)
        proper_structure_count = sum(1 for a in all_analyses if a['table_metrics']['has_proper_structure'])
        
        report.append(f"Average row coverage: {avg_row_coverage:.2%}")
        report.append(f"Average cell coverage: {avg_cell_coverage:.2%}")
        report.append(f"Samples with proper structure: {proper_structure_count}/{len(all_analyses)} ({proper_structure_count/len(all_analyses):.2%})")
        
        avg_position_coverage = sum(a['table_metrics']['position_coverage'] for a in all_analyses) / len(all_analyses)
        avg_column_name_coverage = sum(a['table_metrics']['column_name_coverage'] for a in all_analyses) / len(all_analyses)
        
        report.append(f"Average cell position coverage: {avg_position_coverage:.2%}")
        report.append(f"Average column name coverage: {avg_column_name_coverage:.2%}")
        
        # 3. Text Processing Summary
        report.append("\n" + "="*100)
        report.append("3. TEXT PROCESSING QUALITY")
        report.append("="*100)
        
        avg_text_coverage = sum(a['text_metrics']['text_coverage'] for a in all_analyses) / len(all_analyses)
        avg_text_table_link_ratio = sum(a['text_metrics']['text_to_table_link_ratio'] for a in all_analyses) / len(all_analyses)
        
        report.append(f"Average text coverage: {avg_text_coverage:.2%}")
        report.append(f"Average text-to-table link ratio: {avg_text_table_link_ratio:.2%}")
        
        # 4. Relation Summary
        report.append("\n" + "="*100)
        report.append("4. RELATION EXTRACTION QUALITY")
        report.append("="*100)
        
        avg_relations = sum(a['relation_metrics']['total_relations'] for a in all_analyses) / len(all_analyses)
        avg_unique_relation_types = sum(a['relation_metrics']['unique_relation_types'] for a in all_analyses) / len(all_analyses)
        avg_edges_per_node = sum(a['relation_metrics']['avg_edges_per_node'] for a in all_analyses) / len(all_analyses)
        
        report.append(f"Average relations per sample: {avg_relations:.1f}")
        report.append(f"Average unique relation types: {avg_unique_relation_types:.1f}")
        report.append(f"Average edges per node: {avg_edges_per_node:.2f}")
        
        # Relation type distribution
        all_relation_types = Counter()
        for a in all_analyses:
            for rel_type, count in a['relation_metrics']['relation_type_distribution'].items():
                all_relation_types[rel_type] += count
        
        report.append(f"\nRelation Type Distribution:")
        for rel_type, count in all_relation_types.most_common(10):
            report.append(f"  {rel_type}: {count}")
        
        # 5. Coverage Summary (MOST IMPORTANT!)
        report.append("\n" + "="*100)
        report.append("5. INFORMATION COVERAGE (KEY METRIC!)")
        report.append("="*100)
        
        avg_number_coverage_rate = sum(a['coverage_metrics']['number_coverage_rate'] for a in all_analyses) / len(all_analyses)
        avg_keyword_coverage_rate = sum(a['coverage_metrics']['keyword_coverage_rate'] for a in all_analyses) / len(all_analyses)
        avg_overall_coverage = sum(a['coverage_metrics']['overall_coverage_score'] for a in all_analyses) / len(all_analyses)
        
        report.append(f"Average number coverage rate: {avg_number_coverage_rate:.2%} ← CRITICAL!")
        report.append(f"Average keyword coverage rate: {avg_keyword_coverage_rate:.2%}")
        report.append(f"Average overall coverage score: {avg_overall_coverage:.2%}")
        
        # Samples with perfect coverage
        perfect_coverage = sum(1 for a in all_analyses if a['coverage_metrics']['number_coverage_rate'] >= 1.0)
        report.append(f"\nSamples with 100% number coverage: {perfect_coverage}/{len(all_analyses)} ({perfect_coverage/len(all_analyses):.2%})")
        
        # Samples with issues
        low_coverage_samples = [a for a in all_analyses if a['coverage_metrics']['number_coverage_rate'] < 0.5]
        report.append(f"Samples with <50% number coverage: {len(low_coverage_samples)}/{len(all_analyses)} ({len(low_coverage_samples)/len(all_analyses):.2%})")
        
        if low_coverage_samples:
            report.append(f"\nSamples with coverage issues:")
            for a in low_coverage_samples[:5]:  # Show first 5
                report.append(f"  - {a['sample_id']}: {a['coverage_metrics']['number_coverage_rate']:.2%} coverage")
                report.append(f"    Missing {len(a['coverage_metrics']['missing_numbers'])} numbers: {a['coverage_metrics']['missing_numbers']}")
        
        # 6. Overall KG Statistics
        report.append("\n" + "="*100)
        report.append("6. OVERALL KG STATISTICS")
        report.append("="*100)
        
        avg_nodes = sum(a['kg_stats']['total_nodes'] for a in all_analyses) / len(all_analyses)
        avg_edges = sum(a['kg_stats']['total_edges'] for a in all_analyses) / len(all_analyses)
        
        report.append(f"Average nodes per KG: {avg_nodes:.1f}")
        report.append(f"Average edges per KG: {avg_edges:.1f}")
        
        # Node type distribution
        all_node_types = Counter()
        for a in all_analyses:
            for node_type, count in a['kg_stats']['node_types'].items():
                all_node_types[node_type] += count
        
        report.append(f"\nNode Type Distribution (Total):")
        for node_type, count in all_node_types.most_common():
            report.append(f"  {node_type}: {count}")
        
        # 7. Recommendations
        report.append("\n" + "="*100)
        report.append("7. RECOMMENDATIONS")
        report.append("="*100)
        
        if avg_number_coverage_rate < 0.8:
            report.append("❌ CRITICAL: Number coverage is LOW (<80%)!")
            report.append("   → Improve entity extraction from tables")
            report.append("   → Check number parsing and normalization")
            report.append("   → Verify table structure is correctly built")
        else:
            report.append("✓ Number coverage is acceptable (>80%)")
        
        if avg_duplicate_ratio > 0.3:
            report.append("⚠ WARNING: High duplicate ratio (>30%)!")
            report.append("   → Improve entity deduplication")
            report.append("   → Use normalized keys for entity index")
        else:
            report.append("✓ Entity deduplication is working well")
        
        if avg_text_table_link_ratio < 0.3:
            report.append("⚠ WARNING: Low text-to-table linking (<30%)!")
            report.append("   → Improve semantic matching between text and table")
            report.append("   → Add more relation types")
        else:
            report.append("✓ Text-to-table linking is working")
        
        if avg_cell_coverage < 0.9:
            report.append("❌ CRITICAL: Cell coverage is LOW (<90%)!")
            report.append("   → Check table parsing logic")
            report.append("   → Verify all cells are being created")
        else:
            report.append("✓ Table structure is complete")
        
        # Write report
        report_text = '\n'.join(report)
        output_path.write_text(report_text, encoding='utf-8')
        
        return report_text


async def analyze_kg_with_intelligent_builder(num_samples: int = 50):
    """Analyze KG quality using IntelligentKGBuilder"""
    
    print("="*100)
    print("ANALYZING KG QUALITY WITH INTELLIGENT BUILDER")
    print("="*100)
    
    # Load data
    train_path = Path(__file__).parent.parent.parent / "FinQA" / "dataset" / "train.json"
    
    if not train_path.exists():
        print(f"✗ Error: train.json not found at {train_path}")
        return
    
    print(f"\nLoading data from: {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    print(f"Will analyze first {num_samples} samples\n")
    
    # Initialize
    kg_builder = IntelligentKGBuilder(use_gpu=False)
    analyzer = KGQualityAnalyzer()
    
    # Process samples
    all_analyses = []
    
    for i, sample in enumerate(data[:num_samples]):
        print(f"\n{'='*100}")
        print(f"Processing sample {i+1}/{num_samples}: {sample.get('id', 'N/A')}")
        print(f"{'='*100}")
        
        try:
            # Build KG
            kg = kg_builder.build_kg(sample)
            
            # Analyze
            analysis = analyzer.analyze_sample(sample, kg, i)
            all_analyses.append(analysis)
            
            # Print quick stats
            print(f"\n  KG Stats:")
            print(f"    Nodes: {analysis['kg_stats']['total_nodes']}")
            print(f"    Edges: {analysis['kg_stats']['total_edges']}")
            print(f"    Entities: {analysis['entity_metrics']['total_entities']}")
            print(f"  Coverage:")
            print(f"    Number coverage: {analysis['coverage_metrics']['number_coverage_rate']:.2%}")
            print(f"    Overall coverage: {analysis['coverage_metrics']['overall_coverage_score']:.2%}")
            
            # Visualize first 3 samples
            if i < 3:
                output_dir = Path(__file__).parent / "output" / "kg_analysis"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                safe_id = sample.get('id', f'sample_{i}').replace('/', '_').replace('\\', '_').replace(':', '_')
                viz_path = output_dir / f"kg_visualization_{safe_id}.html"
                
                try:
                    analyzer.visualizer.visualize_kg(kg, str(viz_path), 
                                                     title=f"Sample {i+1}: {sample.get('id', 'N/A')}")
                    print(f"  ✓ Visualization saved: {viz_path}")
                except Exception as e:
                    print(f"  ⚠ Could not visualize: {e}")
            
        except Exception as e:
            print(f"  ✗ Error processing sample: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate report
    print(f"\n{'='*100}")
    print("GENERATING FINAL REPORT")
    print(f"{'='*100}")
    
    output_dir = Path(__file__).parent / "output" / "kg_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"kg_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_text = analyzer.generate_report(all_analyses, report_path)
    
    print(f"\n✓ Report saved to: {report_path}")
    print("\n" + "="*100)
    print("REPORT PREVIEW:")
    print("="*100)
    print(report_text)
    
    # Save detailed results as JSON
    json_path = output_dir / f"kg_quality_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_analyses, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Detailed data saved to: {json_path}")


async def analyze_kg_with_structured_builder(num_samples: int = 50):
    """Analyze KG quality using StructuredKGBuilder"""
    
    print("="*100)
    print("ANALYZING KG QUALITY WITH STRUCTURED BUILDER")
    print("="*100)
    
    # Load data
    train_path = Path(__file__).parent.parent.parent / "FinQA" / "dataset" / "train.json"
    
    if not train_path.exists():
        print(f"✗ Error: train.json not found at {train_path}")
        return
    
    print(f"\nLoading data from: {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    print(f"Will analyze first {num_samples} samples\n")
    
    # Initialize
    kg_builder = StructuredKGBuilder()
    analyzer = KGQualityAnalyzer()
    
    # Process samples
    all_analyses = []
    
    for i, sample in enumerate(data[:num_samples]):
        print(f"\n{'='*100}")
        print(f"Processing sample {i+1}/{num_samples}: {sample.get('id', 'N/A')}")
        print(f"{'='*100}")
        
        try:
            # Build KG
            kg = kg_builder.build_from_sample(sample)
            
            # Analyze
            analysis = analyzer.analyze_sample(sample, kg, i)
            all_analyses.append(analysis)
            
            # Print quick stats
            print(f"\n  KG Stats:")
            print(f"    Nodes: {analysis['kg_stats']['total_nodes']}")
            print(f"    Edges: {analysis['kg_stats']['total_edges']}")
            print(f"    Entities: {analysis['entity_metrics']['total_entities']}")
            print(f"  Coverage:")
            print(f"    Number coverage: {analysis['coverage_metrics']['number_coverage_rate']:.2%}")
            print(f"    Overall coverage: {analysis['coverage_metrics']['overall_coverage_score']:.2%}")
            
            # Visualize first 3 samples
            if i < 3:
                output_dir = Path(__file__).parent / "output" / "kg_analysis"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                safe_id = sample.get('id', f'sample_{i}').replace('/', '_').replace('\\', '_').replace(':', '_')
                viz_path = output_dir / f"structured_kg_visualization_{safe_id}.html"
                
                try:
                    analyzer.visualizer.visualize_kg(kg, str(viz_path), 
                                                     title=f"Structured Sample {i+1}: {sample.get('id', 'N/A')}")
                    print(f"  ✓ Visualization saved: {viz_path}")
                except Exception as e:
                    print(f"  ⚠ Could not visualize: {e}")
            
        except Exception as e:
            print(f"  ✗ Error processing sample: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate report
    print(f"\n{'='*100}")
    print("GENERATING FINAL REPORT")
    print(f"{'='*100}")
    
    output_dir = Path(__file__).parent / "output" / "kg_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"structured_kg_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_text = analyzer.generate_report(all_analyses, report_path)
    
    print(f"\n✓ Report saved to: {report_path}")
    print("\n" + "="*100)
    print("REPORT PREVIEW:")
    print("="*100)
    print(report_text)
    
    # Save detailed results as JSON
    json_path = output_dir / f"structured_kg_quality_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_analyses, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Detailed data saved to: {json_path}")


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Knowledge Graph Quality')
    parser.add_argument('--builder', choices=['intelligent', 'structured', 'both'], default='both',
                       help='Which KG builder to analyze')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to analyze (default: 50)')
    
    args = parser.parse_args()
    
    if args.builder in ['intelligent', 'both']:
        print("\n\nANALYZING INTELLIGENT KG BUILDER...")
        asyncio.run(analyze_kg_with_intelligent_builder(args.num_samples))
    
    if args.builder in ['structured', 'both']:
        print("\n\nANALYZING STRUCTURED KG BUILDER...")
        asyncio.run(analyze_kg_with_structured_builder(args.num_samples))


if __name__ == "__main__":
    main()
