"""
Batch Processor - Xử lý nhiều samples và đánh giá kết quả
"""

import json
import asyncio
from typing import List, Dict, Any
from pathlib import Path
import logging
from tqdm import tqdm
from dataclasses import dataclass, asdict

from .single_sample_processor import SingleSampleProcessor, ExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class BatchStatistics:
    """Thống kê kết quả batch processing"""
    total_samples: int
    successful: int
    failed: int
    correct_answers: int
    incorrect_answers: int
    accuracy: float
    avg_execution_time: float
    
    def to_dict(self):
        return asdict(self)

class BatchProcessor:
    """
    Xử lý batch samples từ FinQA dataset
    """
    
    def __init__(self, max_workers: int = 1):
        """
        Args:
            max_workers: Số samples xử lý đồng thời (1 = sequential)
        """
        self.max_workers = max_workers
        self.results = []
        
    async def process_dataset(self, 
                             data_path: str, 
                             max_samples: int = None,
                             output_path: str = None) -> BatchStatistics:
        """
        Xử lý toàn bộ dataset
        
        Args:
            data_path: Path to FinQA JSON file
            max_samples: Giới hạn số samples (None = all)
            output_path: Path to save results
            
        Returns:
            BatchStatistics
        """
        logger.info(f"Loading dataset from {data_path}")
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        logger.info(f"Processing {len(data)} samples...")
        
        # Process samples
        import time
        start_time = time.time()
        
        if self.max_workers == 1:
            # Sequential processing
            for sample in tqdm(data, desc="Processing samples"):
                result = await self._process_single_sample(sample)
                self.results.append(result)
        else:
            # Parallel processing (if needed in future)
            tasks = [self._process_single_sample(sample) for sample in data]
            self.results = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # Calculate statistics
        stats = self._calculate_statistics(elapsed_time)
        
        logger.info(f"\n{self._format_statistics(stats)}")
        
        # Save results if requested
        if output_path:
            self._save_results(output_path, stats)
        
        return stats
    
    async def _process_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Xử lý một sample và return result dict"""
        processor = SingleSampleProcessor()
        
        try:
            result = await processor.process_sample(sample)
            
            return {
                'sample_id': sample.get('id', 'unknown'),
                'question': sample.get('qa', {}).get('question', ''),
                'predicted_answer': result.final_answer,
                'ground_truth': result.ground_truth,
                'is_correct': result.is_correct,
                'program': sample.get('qa', {}).get('program', ''),
                'steps': [
                    {
                        'operator': step.operator,
                        'arg1': step.arg1,
                        'arg2': step.arg2,
                        'result': step.result
                    } for step in result.steps
                ],
                'explanation': result.explanation,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {sample.get('id')}: {e}")
            return {
                'sample_id': sample.get('id', 'unknown'),
                'question': sample.get('qa', {}).get('question', ''),
                'success': False,
                'error': str(e),
                'predicted_answer': None,
                'ground_truth': sample.get('qa', {}).get('exe_ans'),
                'is_correct': False
            }
    
    def _calculate_statistics(self, elapsed_time: float) -> BatchStatistics:
        """Tính toán statistics từ results"""
        total = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        failed = total - successful
        
        correct = sum(1 for r in self.results if r.get('is_correct', False))
        incorrect = successful - correct
        
        accuracy = correct / successful if successful > 0 else 0.0
        avg_time = elapsed_time / total if total > 0 else 0.0
        
        return BatchStatistics(
            total_samples=total,
            successful=successful,
            failed=failed,
            correct_answers=correct,
            incorrect_answers=incorrect,
            accuracy=accuracy,
            avg_execution_time=avg_time
        )
    
    def _format_statistics(self, stats: BatchStatistics) -> str:
        """Format statistics for display"""
        return f"""
╔══════════════════════════════════════╗
║      BATCH PROCESSING RESULTS        ║
╠══════════════════════════════════════╣
║ Total Samples:        {stats.total_samples:>6}       ║
║ Successful:           {stats.successful:>6}       ║
║ Failed:               {stats.failed:>6}       ║
║                                      ║
║ Correct Answers:      {stats.correct_answers:>6}       ║
║ Incorrect Answers:    {stats.incorrect_answers:>6}       ║
║                                      ║
║ Accuracy:            {stats.accuracy:>6.2%}       ║
║ Avg Time/Sample:     {stats.avg_execution_time:>6.2f}s      ║
╚══════════════════════════════════════╝
        """
    
    def _save_results(self, output_path: str, stats: BatchStatistics):
        """Save results to JSON file"""
        output_data = {
            'statistics': stats.to_dict(),
            'results': self.results
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Phân tích các lỗi thường gặp"""
        errors = [r for r in self.results if not r['success']]
        incorrect = [r for r in self.results if r['success'] and not r.get('is_correct', False)]
        
        error_types = {}
        for error in errors:
            error_msg = str(error.get('error', 'Unknown'))
            error_type = error_msg.split(':')[0]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(errors),
            'total_incorrect': len(incorrect),
            'error_types': error_types,
            'example_errors': errors[:5],
            'example_incorrect': incorrect[:5]
        }
