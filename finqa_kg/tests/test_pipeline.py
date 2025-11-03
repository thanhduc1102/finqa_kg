"""
Test Script - Single Sample Processing Pipeline
Demo việc xử lý từng sample riêng lẻ
"""

import asyncio
import json
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import SingleSampleProcessor, BatchProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_single_sample():
    """Test với một sample duy nhất"""
    print("\n" + "="*60)
    print("TEST 1: SINGLE SAMPLE PROCESSING")
    print("="*60 + "\n")
    
    # Load test data
    test_file = Path(__file__).parent / "test_data.json"
    
    if not test_file.exists():
        # Create simple test data
        test_data = {
            "id": "test_sample_1",
            "filename": "test.pdf",
            "pre_text": [
                "Company XYZ reported Q3 2023 results.",
                "Revenue increased significantly."
            ],
            "post_text": [
                "Operating costs decreased by 5%.",
                "Net profit margin improved."
            ],
            "table": [
                ["Metric", "Q3 2022", "Q3 2023"],
                ["Revenue", "1000", "1200"],
                ["Costs", "842", "800"],
                ["Profit", "158", "400"]
            ],
            "qa": {
                "question": "What is the average payment volume per transaction for American Express?",
                "program": "divide(637, const_5)",
                "exe_ans": 127.4,
                "answer": "127.40"
            }
        }
        
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"Created test data at {test_file}")
    
    with open(test_file, 'r') as f:
        sample = json.load(f)
    
    # Process sample
    processor = SingleSampleProcessor()
    result = await processor.process_sample(sample)
    
    # Display results
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}\n")
    
    print(f"Question: {sample['qa']['question']}")
    print(f"\nProgram: {sample['qa']['program']}")
    print(f"\nExecution Steps:")
    for idx, step in enumerate(result.steps, 1):
        print(f"  {idx}. {step.operator}({step.arg1}, {step.arg2}) = {step.result}")
    
    print(f"\nFinal Answer: {result.final_answer}")
    print(f"Ground Truth: {result.ground_truth}")
    print(f"Correct: {'✓' if result.is_correct else '✗'}")
    
    print(f"\n{'='*60}")
    print("KNOWLEDGE GRAPH INFO:")
    print(f"{'='*60}\n")
    print(f"Nodes: {processor.kg.number_of_nodes()}")
    print(f"Edges: {processor.kg.number_of_edges()}")
    print(f"Numbers indexed: {len(processor.number_index)}")
    
    # Visualize
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        processor.visualize_computation(
            result, 
            str(output_dir / "computation_graph.png")
        )
        print(f"\nVisualization saved to {output_dir / 'computation_graph.png'}")
    except Exception as e:
        logger.warning(f"Could not create visualization: {e}")
    
    print(f"\n{'='*60}\n")
    
    return result

async def test_real_finqa_sample():
    """Test với sample thực từ FinQA dataset"""
    print("\n" + "="*60)
    print("TEST 2: REAL FINQA SAMPLE")
    print("="*60 + "\n")
    
    # Load real FinQA data
    finqa_path = Path(__file__).parent.parent.parent / "FinQA" / "dataset" / "dev.json"
    
    if not finqa_path.exists():
        print(f"FinQA dataset not found at {finqa_path}")
        print("Skipping real sample test...")
        return None
    
    with open(finqa_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take first sample
    sample = data[0]
    
    print(f"Sample ID: {sample['id']}")
    print(f"Question: {sample['qa']['question']}")
    print(f"Program: {sample['qa']['program']}")
    print(f"Expected Answer: {sample['qa']['exe_ans']}")
    
    # Process
    processor = SingleSampleProcessor()
    result = await processor.process_sample(sample)
    
    print(f"\n{'='*60}")
    print("EXECUTION:")
    print(f"{'='*60}\n")
    
    for idx, step in enumerate(result.steps, 1):
        print(f"{idx}. {step.operator}({step.arg1}, {step.arg2}) = {step.result}")
        print(f"   Source nodes: {step.source_nodes}")
    
    print(f"\nPredicted: {result.final_answer}")
    print(f"Expected:  {result.ground_truth}")
    print(f"Correct:   {'✓ YES' if result.is_correct else '✗ NO'}")
    
    print(f"\n{'='*60}")
    print("EXPLANATION:")
    print(f"{'='*60}\n")
    print(result.explanation)
    
    return result

async def test_batch_processing():
    """Test batch processing với nhiều samples"""
    print("\n" + "="*60)
    print("TEST 3: BATCH PROCESSING")
    print("="*60 + "\n")
    
    finqa_path = Path(__file__).parent.parent.parent / "FinQA" / "dataset" / "dev.json"
    
    if not finqa_path.exists():
        print(f"FinQA dataset not found at {finqa_path}")
        print("Skipping batch test...")
        return None
    
    # Process 10 samples
    processor = BatchProcessor(max_workers=1)
    
    output_path = Path(__file__).parent / "test_output" / "batch_results.json"
    
    stats = await processor.process_dataset(
        str(finqa_path),
        max_samples=10,
        output_path=str(output_path)
    )
    
    print(f"\nResults saved to {output_path}")
    
    # Error analysis
    print("\n" + "="*60)
    print("ERROR ANALYSIS:")
    print("="*60 + "\n")
    
    error_analysis = processor.get_error_analysis()
    print(f"Total Errors: {error_analysis['total_errors']}")
    print(f"Total Incorrect: {error_analysis['total_incorrect']}")
    print(f"\nError Types:")
    for error_type, count in error_analysis['error_types'].items():
        print(f"  - {error_type}: {count}")
    
    return stats

async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "FINQA SINGLE SAMPLE PIPELINE TEST")
    print("="*70)
    
    try:
        # Test 1: Simple sample
        await test_single_sample()
        
        # Test 2: Real FinQA sample
        await test_real_finqa_sample()
        
        # Test 3: Batch processing
        await test_batch_processing()
        
        print("\n" + "="*70)
        print(" "*25 + "ALL TESTS COMPLETED")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
