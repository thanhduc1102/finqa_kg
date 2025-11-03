"""
Simple Quick Test - Verify pipeline works
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Importing modules...")
try:
    from src.pipeline.single_sample_processor import SingleSampleProcessor
    print("✓ SingleSampleProcessor imported")
except Exception as e:
    print(f"✗ Error importing: {e}")
    sys.exit(1)

async def quick_test():
    print("\n" + "="*60)
    print("QUICK TEST - Single Sample Processor")
    print("="*60 + "\n")
    
    # Create test sample
    sample = {
        "id": "quick_test_1",
        "filename": "test.pdf",
        "pre_text": ["Revenue in 2023 was $637 billion."],
        "post_text": ["Total transactions: 5 billion."],
        "table": [
            ["Metric", "Value"],
            ["Revenue", "637"],
            ["Transactions", "5"]
        ],
        "qa": {
            "question": "What is the average revenue per transaction?",
            "program": "divide(637, 5)",
            "exe_ans": 127.4,
            "answer": "127.4"
        }
    }
    
    print("Test Sample:")
    print(f"  Question: {sample['qa']['question']}")
    print(f"  Program: {sample['qa']['program']}")
    print(f"  Expected: {sample['qa']['exe_ans']}")
    
    print("\nProcessing...")
    
    try:
        processor = SingleSampleProcessor()
        result = await processor.process_sample(sample)
        print(result)
        
        print("\n" + "-"*60)
        print("RESULTS:")
        print("-"*60)
        print(f"Computed Answer: {result.final_answer}")
        print(f"Expected Answer: {result.ground_truth}")
        print(f"Match: {'✓ YES' if result.is_correct else '✗ NO'}")
        
        print(f"\nKnowledge Graph:")
        print(f"  Nodes: {processor.kg.number_of_nodes()}")
        print(f"  Edges: {processor.kg.number_of_edges()}")
        print(f"  Numbers indexed: {len(processor.number_index)}")
        
        print(f"\nComputation Steps:")
        for idx, step in enumerate(result.steps, 1):
            print(f"  {idx}. {step.operator}({step.arg1}, {step.arg2}) = {step.result}")
        
        print("\n" + "="*60)
        print("✓ TEST PASSED!" if result.is_correct else "✗ TEST FAILED!")
        print("="*60 + "\n")
        
        return result.is_correct
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)
