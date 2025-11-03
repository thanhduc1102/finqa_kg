"""
Test Intelligent Pipeline với dữ liệu thực từ train.json
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.finqa_intelligent_pipeline import IntelligentFinQAPipeline

async def test_single_sample():
    """Test với một sample từ train.json"""
    
    print("="*80)
    print("TESTING INTELLIGENT FINQA PIPELINE")
    print("="*80)
    
    # Load data
    train_path = Path(__file__).parent.parent.parent / "FinQA" / "dataset" / "train.json"
    
    if not train_path.exists():
        print(f"✗ Error: train.json not found at {train_path}")
        return False
    
    print(f"\nLoading data from: {train_path}")
    
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = IntelligentFinQAPipeline(use_gpu=False)
    
    # Test với sample đầu tiên
    sample = data[0]
    
    print(f"\n{'='*80}")
    print("TEST SAMPLE INFO:")
    print(f"{'='*80}")
    print(f"ID: {sample.get('id', 'N/A')}")
    print(f"Filename: {sample.get('filename', 'N/A')}")
    print(f"Pre-text sentences: {len(sample.get('pre_text', []))}")
    print(f"Post-text sentences: {len(sample.get('post_text', []))}")
    print(f"Table rows: {len(sample.get('table', []))}")
    print(f"Question: {sample.get('qa', {}).get('question', 'N/A')}")
    print(f"Expected Answer: {sample.get('qa', {}).get('exe_ans', 'N/A')}")
    print(f"Original Program: {sample.get('qa', {}).get('program', 'N/A')}")
    
    # Process
    result = await pipeline.process_sample(sample)
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}")
    
    if result.error:
        print(f"✗ Error: {result.error}")
        return False
    
    print(f"\nQuestion Type: {result.question_analysis.question_type if result.question_analysis else 'N/A'}")
    print(f"Synthesized Program: {result.synthesized_program}")
    print(f"Synthesis Confidence: {result.synthesis_confidence:.2%}")
    print(f"\nFinal Answer: {result.final_answer}")
    print(f"Ground Truth: {result.ground_truth}")
    print(f"Match: {'✓ CORRECT' if result.is_correct else '✗ INCORRECT'}")
    
    print(f"\nKG Stats:")
    print(f"  Nodes: {result.kg_stats['nodes']}")
    print(f"  Edges: {result.kg_stats['edges']}")
    print(f"  Entities: {result.kg_stats['entities']}")
    
    print(f"\nComputation Steps: {len(result.computation_steps)}")
    for i, step in enumerate(result.computation_steps, 1):
        if step.arg2 is not None:
            print(f"  {i}. {step.operation}({step.arg1}, {step.arg2}) = {step.result}")
        else:
            print(f"  {i}. {step.operation}({step.arg1}) = {step.result}")
    
    print(f"\nProcessing Time: {result.processing_time:.2f}s")
    
    # Save full explanation
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Sanitize filename - replace / and \ with _
    safe_id = sample.get('id', 'test').replace('/', '_').replace('\\', '_').replace(':', '_')
    
    explanation_path = output_dir / f"explanation_{safe_id}.txt"
    with open(explanation_path, 'w', encoding='utf-8') as f:
        f.write(result.full_explanation)
    print(f"\nFull explanation saved to: {explanation_path}")
    
    # Try to visualize
    try:
        viz_path = output_dir / f"visualization_{safe_id}.png"
        pipeline.visualize_result(result, str(viz_path))
    except Exception as e:
        print(f"Note: Could not create visualization: {e}")
    
    return result.is_correct

async def test_multiple_samples(num_samples: int = 5):
    """Test với nhiều samples"""
    
    print(f"\n{'='*80}")
    print(f"TESTING WITH {num_samples} SAMPLES")
    print(f"{'='*80}")
    
    # Load data
    train_path = Path(__file__).parent.parent.parent / "FinQA" / "dataset" / "train.json"
    
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize pipeline
    pipeline = IntelligentFinQAPipeline(use_gpu=False)
    
    # Process samples
    results = []
    correct_count = 0
    
    for i, sample in enumerate(data[:num_samples]):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'='*80}")
        
        result = await pipeline.process_sample(sample)
        results.append(result)
        
        if result.is_correct:
            correct_count += 1
        
        print(f"Result: {'✓ CORRECT' if result.is_correct else '✗ INCORRECT'}")
    
    # Summary
    accuracy = correct_count / num_samples
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total Samples: {num_samples}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {num_samples - correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Detailed statistics
    avg_kg_nodes = sum(r.kg_stats['nodes'] for r in results) / len(results)
    avg_kg_entities = sum(r.kg_stats['entities'] for r in results) / len(results)
    avg_steps = sum(len(r.computation_steps) for r in results) / len(results)
    avg_time = sum(r.processing_time for r in results) / len(results)
    avg_confidence = sum(r.synthesis_confidence for r in results) / len(results)
    
    print(f"\nAverage Statistics:")
    print(f"  KG Nodes: {avg_kg_nodes:.1f}")
    print(f"  KG Entities: {avg_kg_entities:.1f}")
    print(f"  Computation Steps: {avg_steps:.1f}")
    print(f"  Synthesis Confidence: {avg_confidence:.2%}")
    print(f"  Processing Time: {avg_time:.2f}s")
    
    # Question type breakdown
    question_types = {}
    for r in results:
        if r.question_analysis:
            q_type = r.question_analysis.question_type
            if q_type not in question_types:
                question_types[q_type] = {'total': 0, 'correct': 0}
            question_types[q_type]['total'] += 1
            if r.is_correct:
                question_types[q_type]['correct'] += 1
    
    print(f"\nBy Question Type:")
    for q_type, stats in question_types.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {q_type}: {stats['correct']}/{stats['total']} ({acc:.1%})")
    
    return accuracy

def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Intelligent FinQA Pipeline')
    parser.add_argument('--mode', choices=['single', 'multiple'], default='single',
                       help='Test mode: single sample or multiple samples')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to test (for multiple mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        success = asyncio.run(test_single_sample())
        sys.exit(0 if success else 1)
    else:
        accuracy = asyncio.run(test_multiple_samples(args.num_samples))
        # Exit with 0 if accuracy >= 50%
        sys.exit(0 if accuracy >= 0.5 else 1)

if __name__ == "__main__":
    main()
