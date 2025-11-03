"""
Comprehensive Demo - Advanced Single Sample Processing
Demonstration của toàn bộ pipeline mới
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import AdvancedSampleProcessor, BatchProcessor

async def demo_advanced_processing():
    """Demo với advanced processor"""
    
    print("\n" + "="*80)
    print(" "*20 + "FINQA ADVANCED PIPELINE DEMO")
    print("="*80 + "\n")
    
    # Load real FinQA sample
    finqa_path = Path(__file__).parent.parent / "FinQA" / "dataset" / "dev.json"
    
    if not finqa_path.exists():
        print(f"FinQA dataset not found at {finqa_path}")
        print("Please ensure FinQA dataset is in the correct location.")
        return
    
    with open(finqa_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process first 3 samples để demo
    print(f"Loaded {len(data)} samples from dataset")
    print(f"Processing first 3 samples for demonstration...\n")
    
    processor = AdvancedSampleProcessor()
    
    for idx, sample in enumerate(data[:3], 1):
        print(f"\n{'█'*80}")
        print(f"{'█'*30} SAMPLE {idx} {'█'*30}")
        print(f"{'█'*80}\n")
        
        print(f"ID: {sample['id']}")
        print(f"File: {sample['filename']}")
        print(f"\nQuestion: {sample['qa']['question']}")
        print(f"Expected Answer: {sample['qa']['exe_ans']}")
        print(f"Provided Program: {sample['qa']['program']}")
        
        print(f"\n{'-'*80}")
        print("PROCESSING...")
        print(f"{'-'*80}\n")
        
        try:
            result = await processor.process_sample(sample)
            
            # Print detailed explanation
            print(result.explanation)
            
            # Show KG stats
            print(f"\n{'-'*80}")
            print("KNOWLEDGE GRAPH STATISTICS")
            print(f"{'-'*80}")
            print(f"Total Nodes: {processor.kg.number_of_nodes()}")
            print(f"Total Edges: {processor.kg.number_of_edges()}")
            print(f"Numbers Indexed: {len(processor.number_index)}")
            
            # Node breakdown
            node_types = {}
            for node_id, data in processor.kg.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            print(f"\nNode Types:")
            for node_type, count in sorted(node_types.items()):
                print(f"  {node_type:15s}: {count:4d}")
            
            # Visualize if possible
            output_dir = Path(__file__).parent / "demo_output"
            output_dir.mkdir(exist_ok=True)
            
            viz_path = output_dir / f"sample_{idx}_computation.png"
            try:
                processor.visualize_computation(result, str(viz_path))
                print(f"\nVisualization saved: {viz_path}")
            except Exception as e:
                print(f"\nVisualization skipped: {e}")
            
        except Exception as e:
            print(f"❌ ERROR processing sample: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'█'*80}\n")
        
        # Pause between samples
        if idx < 3:
            await asyncio.sleep(0.5)
    
    print(f"\n{'='*80}")
    print(" "*25 + "DEMO COMPLETED")
    print(f"{'='*80}\n")

async def demo_batch_with_analysis():
    """Demo batch processing với error analysis"""
    
    print("\n" + "="*80)
    print(" "*20 + "BATCH PROCESSING WITH ANALYSIS")
    print("="*80 + "\n")
    
    finqa_path = Path(__file__).parent.parent / "FinQA" / "dataset" / "dev.json"
    
    if not finqa_path.exists():
        print("Dataset not found. Skipping batch demo.")
        return
    
    # Process 20 samples
    print("Processing 20 samples from dev set...")
    print("This may take a few minutes...\n")
    
    processor = BatchProcessor(max_workers=1)
    
    output_path = Path(__file__).parent / "demo_output" / "batch_results_20.json"
    
    stats = await processor.process_dataset(
        str(finqa_path),
        max_samples=20,
        output_path=str(output_path)
    )
    
    # Detailed analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80 + "\n")
    
    error_analysis = processor.get_error_analysis()
    
    print(f"Total Errors: {error_analysis['total_errors']}")
    print(f"Total Incorrect (but successful): {error_analysis['total_incorrect']}")
    
    print(f"\nError Types:")
    for error_type, count in error_analysis['error_types'].items():
        print(f"  {error_type}: {count}")
    
    # Show examples of incorrect answers
    print(f"\n{'-'*80}")
    print("EXAMPLES OF INCORRECT ANSWERS:")
    print(f"{'-'*80}\n")
    
    for idx, example in enumerate(error_analysis['example_incorrect'][:3], 1):
        print(f"\n{idx}. Sample: {example['sample_id']}")
        print(f"   Question: {example['question']}")
        print(f"   Predicted: {example['predicted_answer']}")
        print(f"   Ground Truth: {example['ground_truth']}")
        print(f"   Program: {example['program']}")
    
    print(f"\n{'='*80}\n")

async def main():
    """Run all demos"""
    
    try:
        # Demo 1: Advanced processing
        await demo_advanced_processing()
        
        # Demo 2: Batch with analysis
        await demo_batch_with_analysis()
        
        print("\n" + "🎉"*40)
        print("\n" + " "*25 + "ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("\n" + "🎉"*40 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
