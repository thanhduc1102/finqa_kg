"""
Pipeline Module - Intelligent FinQA Processing
"""

# New Intelligent Pipeline
from .intelligent_kg_builder import IntelligentKGBuilder, Entity, Relation
from .question_analyzer import QuestionAnalyzer, QuestionAnalysis
from .program_synthesizer import ProgramSynthesizer, ProgramSynthesisResult
from .program_executor import ProgramExecutor, ExecutionResult, ComputationStep
from .finqa_intelligent_pipeline import IntelligentFinQAPipeline, PipelineResult

# Legacy components (kept for backward compatibility)
from .single_sample_processor import (
    SingleSampleProcessor,
    ProgramStep,
    OperatorType
)

from .batch_processor import (
    BatchProcessor,
    BatchStatistics
)

from .advanced_processor import (
    AdvancedSampleProcessor,
    QuestionIntent
)

__all__ = [
    # New Intelligent System
    'IntelligentKGBuilder',
    'Entity',
    'Relation',
    'QuestionAnalyzer',
    'QuestionAnalysis',
    'ProgramSynthesizer',
    'ProgramSynthesisResult',
    'ProgramExecutor',
    'ExecutionResult',
    'ComputationStep',
    'IntelligentFinQAPipeline',
    'PipelineResult',
    
    # Legacy
    'SingleSampleProcessor',
    'ProgramStep',
    'OperatorType',
    'BatchProcessor',
    'BatchStatistics',
    'AdvancedSampleProcessor',
    'QuestionIntent'
]
