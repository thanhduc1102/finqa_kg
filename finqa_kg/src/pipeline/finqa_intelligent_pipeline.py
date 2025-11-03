"""
Intelligent FinQA Pipeline
Tích hợp toàn bộ components:
1. Build KG từ sample
2. Analyze question
3. Synthesize program
4. Execute program
5. Generate explanation
"""

import asyncio
import networkx as nx
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .intelligent_kg_builder import IntelligentKGBuilder
from .question_analyzer import QuestionAnalyzer, QuestionAnalysis
from .program_synthesizer import ProgramSynthesizer, ProgramSynthesisResult
from .program_executor import ProgramExecutor, ExecutionResult

@dataclass
class PipelineResult:
    """Kết quả cuối cùng của pipeline"""
    # Input
    sample_id: str
    question: str
    
    # KG
    kg: nx.MultiDiGraph
    kg_stats: Dict[str, int]
    
    # Analysis
    question_analysis: QuestionAnalysis
    
    # Synthesis
    synthesized_program: str
    synthesis_confidence: float
    
    # Execution
    final_answer: float
    computation_steps: list
    is_correct: bool
    ground_truth: Optional[float]
    
    # Explanation
    full_explanation: str
    
    # Metadata
    processing_time: float
    error: Optional[str] = None

class IntelligentFinQAPipeline:
    """
    Main pipeline class
    
    Usage:
        pipeline = IntelligentFinQAPipeline()
        result = await pipeline.process_sample(sample)
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize all components
        """
        print("Initializing Intelligent FinQA Pipeline...")
        
        self.kg_builder = IntelligentKGBuilder(use_gpu=use_gpu)
        self.question_analyzer = QuestionAnalyzer()
        self.program_synthesizer = ProgramSynthesizer()
        self.program_executor = ProgramExecutor()
        
        print("✓ Pipeline ready!")
    
    async def process_sample(self, sample: Dict[str, Any]) -> PipelineResult:
        """
        Process một sample hoàn chỉnh
        
        Args:
            sample: Dict với keys:
                - id: sample ID
                - pre_text: list of strings
                - post_text: list of strings
                - table: 2D list
                - qa: dict với 'question', 'answer', 'exe_ans', etc
        
        Returns:
            PipelineResult
        """
        import time
        start_time = time.time()
        
        sample_id = sample.get('id', 'unknown')
        print(f"\n{'='*60}")
        print(f"Processing Sample: {sample_id}")
        print(f"{'='*60}")
        
        try:
            # Extract QA info
            qa = sample.get('qa', {})
            question = qa.get('question', '')
            ground_truth = qa.get('exe_ans', None)
            
            if not question:
                raise ValueError("No question found in sample")
            
            print(f"Question: {question}")
            if ground_truth is not None:
                print(f"Ground Truth: {ground_truth}")
            
            # Step 1: Build Knowledge Graph
            print(f"\n[STEP 1/5] Building Knowledge Graph...")
            kg = self.kg_builder.build_kg(sample)
            entity_index = self.kg_builder.get_entity_index(kg)
            
            kg_stats = {
                'nodes': kg.number_of_nodes(),
                'edges': kg.number_of_edges(),
                'entities': len([n for n in kg.nodes() if kg.nodes[n].get('type') == 'entity'])
            }
            print(f"  KG Stats: {kg_stats}")
            
            # Step 2: Analyze Question
            print(f"\n[STEP 2/5] Analyzing Question...")
            qa_analysis = self.question_analyzer.analyze(question)
            print(f"  Type: {qa_analysis.question_type}")
            print(f"  Operations: {qa_analysis.operations}")
            print(f"  Entities: {qa_analysis.entities_mentioned[:5]}")
            
            # Step 3: Synthesize Program
            print(f"\n[STEP 3/5] Synthesizing Program...")
            synthesis_result = self.program_synthesizer.synthesize(
                qa_analysis,
                kg,
                entity_index
            )
            print(f"  Program: {synthesis_result.program}")
            print(f"  Confidence: {synthesis_result.confidence:.2f}")
            
            # Step 4: Execute Program
            print(f"\n[STEP 4/5] Executing Program...")
            exec_result = self.program_executor.execute(
                synthesis_result.program,
                synthesis_result.placeholders,
                ground_truth
            )
            
            if exec_result.error:
                print(f"  ✗ Execution Error: {exec_result.error}")
            else:
                print(f"  Answer: {exec_result.final_answer}")
                print(f"  Steps: {len(exec_result.steps)}")
                if ground_truth is not None:
                    print(f"  Correct: {'✓ YES' if exec_result.is_correct else '✗ NO'}")
            
            # Step 5: Generate Full Explanation
            print(f"\n[STEP 5/5] Generating Explanation...")
            full_explanation = self._generate_full_explanation(
                question,
                qa_analysis,
                synthesis_result,
                exec_result,
                kg
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            print(f"\n✓ Processing complete in {processing_time:.2f}s")
            
            # Create result
            return PipelineResult(
                sample_id=sample_id,
                question=question,
                kg=kg,
                kg_stats=kg_stats,
                question_analysis=qa_analysis,
                synthesized_program=synthesis_result.program,
                synthesis_confidence=synthesis_result.confidence,
                final_answer=exec_result.final_answer,
                computation_steps=exec_result.steps,
                is_correct=exec_result.is_correct,
                ground_truth=ground_truth,
                full_explanation=full_explanation,
                processing_time=processing_time,
                error=exec_result.error
            )
        
        except Exception as e:
            import traceback
            error_msg = f"Pipeline error: {str(e)}\n{traceback.format_exc()}"
            print(f"\n✗ {error_msg}")
            
            processing_time = time.time() - start_time
            
            return PipelineResult(
                sample_id=sample_id,
                question=question if 'question' in locals() else "",
                kg=nx.MultiDiGraph(),
                kg_stats={'nodes': 0, 'edges': 0, 'entities': 0},
                question_analysis=None,
                synthesized_program="",
                synthesis_confidence=0.0,
                final_answer=0.0,
                computation_steps=[],
                is_correct=False,
                ground_truth=ground_truth if 'ground_truth' in locals() else None,
                full_explanation="",
                processing_time=processing_time,
                error=error_msg
            )
    
    def _generate_full_explanation(self,
                                   question: str,
                                   qa_analysis: QuestionAnalysis,
                                   synthesis: ProgramSynthesisResult,
                                   execution: ExecutionResult,
                                   kg: nx.MultiDiGraph) -> str:
        """
        Generate comprehensive explanation
        """
        lines = []
        
        lines.append("=" * 60)
        lines.append("DETAILED EXPLANATION")
        lines.append("=" * 60)
        
        # Question
        lines.append(f"\nQuestion: {question}")
        
        # Question Analysis
        lines.append(f"\n--- Question Analysis ---")
        lines.append(f"Type: {qa_analysis.question_type}")
        lines.append(f"Required Operations: {', '.join(qa_analysis.operations)}")
        if qa_analysis.entities_mentioned:
            lines.append(f"Entities Mentioned: {', '.join(qa_analysis.entities_mentioned[:10])}")
        if qa_analysis.temporal_entities:
            lines.append(f"Temporal Info: {', '.join(qa_analysis.temporal_entities)}")
        
        # Knowledge Graph
        lines.append(f"\n--- Knowledge Graph ---")
        lines.append(f"Total Nodes: {kg.number_of_nodes()}")
        lines.append(f"Total Edges: {kg.number_of_edges()}")
        entity_count = len([n for n in kg.nodes() if kg.nodes[n].get('type') == 'entity'])
        lines.append(f"Entities Extracted: {entity_count}")
        
        # Program Synthesis
        lines.append(f"\n--- Program Synthesis ---")
        lines.append(f"Synthesized Program: {synthesis.program}")
        lines.append(f"Confidence: {synthesis.confidence:.2%}")
        lines.append(f"\nArguments Retrieved:")
        for ph_id, ph_data in synthesis.placeholders.items():
            value = ph_data.get('value', 'N/A')
            context = ph_data.get('context', '')[:80]
            lines.append(f"  {ph_id}: {value}")
            lines.append(f"       Context: {context}...")
        
        # Execution
        lines.append(f"\n--- Execution Steps ---")
        for step in execution.steps:
            if step.arg2 is not None:
                lines.append(f"Step {step.step_num}: {step.operation}({step.arg1}, {step.arg2}) = {step.result}")
            else:
                lines.append(f"Step {step.step_num}: {step.operation}({step.arg1}) = {step.result}")
            
            if step.source_nodes:
                sources = [s for s in step.source_nodes if s]
                if sources:
                    lines.append(f"         Sources: {', '.join(sources[:3])}")
        
        # Result
        lines.append(f"\n--- Final Result ---")
        lines.append(f"Computed Answer: {execution.final_answer}")
        
        if execution.ground_truth is not None:
            lines.append(f"Ground Truth: {execution.ground_truth}")
            lines.append(f"Match: {'✓ CORRECT' if execution.is_correct else '✗ INCORRECT'}")
            
            if not execution.is_correct:
                error = abs(execution.final_answer - execution.ground_truth)
                error_pct = error / execution.ground_truth * 100 if execution.ground_truth != 0 else 100
                lines.append(f"Error: {error:.4f} ({error_pct:.2f}%)")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def visualize_result(self, result: PipelineResult, output_path: str = None):
        """
        Visualize KG và computation graph
        """
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(20, 10))
        
        # Left: Knowledge Graph
        ax1 = plt.subplot(121)
        pos1 = nx.spring_layout(result.kg, k=0.5, iterations=50)
        
        # Color nodes by type
        node_colors = []
        for node in result.kg.nodes():
            node_type = result.kg.nodes[node].get('type', '')
            if node_type == 'entity':
                node_colors.append('lightblue')
            elif node_type == 'cell':
                node_colors.append('lightgreen')
            elif node_type == 'text':
                node_colors.append('lightyellow')
            elif node_type == 'table':
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightgray')
        
        nx.draw(result.kg, pos1, ax=ax1, node_color=node_colors,
               node_size=300, font_size=6, with_labels=False, arrows=True)
        ax1.set_title(f"Knowledge Graph\n{result.kg_stats['nodes']} nodes, {result.kg_stats['edges']} edges")
        
        # Right: Computation Graph
        ax2 = plt.subplot(122)
        if result.computation_steps:
            comp_graph = nx.DiGraph()
            for step in result.computation_steps:
                step_node = f"Step{step.step_num}"
                comp_graph.add_node(step_node, 
                                   label=f"{step.operation}\n={step.result:.2f}")
            
            pos2 = nx.spring_layout(comp_graph)
            nx.draw(comp_graph, pos2, ax=ax2, node_color='lightgreen',
                   node_size=1000, font_size=10, with_labels=True, arrows=True)
            ax2.set_title(f"Computation Flow\n{len(result.computation_steps)} steps")
        else:
            ax2.text(0.5, 0.5, "No computation graph", 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Computation Flow")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
