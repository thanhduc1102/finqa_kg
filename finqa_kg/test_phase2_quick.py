"""Quick test of Phase 2 fixes on 5 samples"""
import json
import time
from src.pipeline.intelligent_kg_builder import IntelligentKGBuilder
from src.pipeline.question_analyzer import QuestionAnalyzer
from src.pipeline.program_synthesizer import ProgramSynthesizer

# Load first 5 samples
with open('../FinQA/dataset/train.json', 'r') as f:
    data = json.load(f)
samples = data[:5]

print("="*80)
print("PHASE 2 TESTING - 5 SAMPLES")
print("="*80)

# Initialize components once
print("\nInitializing components...")
builder = IntelligentKGBuilder()
analyzer = QuestionAnalyzer()
synthesizer = ProgramSynthesizer()
print("✓ Ready!\n")

# Define arithmetic operations for program execution
def multiply(a, b):
    return a * b
def divide(a, b):
    return a / b if b != 0 else 0
def subtract(a, b):
    return a - b
def add(a, b):
    return a + b

results = []
correct = 0
total = 0

for idx, sample in enumerate(samples, 1):
    print("="*80)
    print(f"Sample {idx}/5: {sample['id']}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Extract info
        question = sample['qa']['question']
        expected_answer = float(sample['qa']['exe_ans'])
        
        print(f"Question: {question}")
        print(f"Expected: {expected_answer}")
        
        # Build KG
        print("\n[1/3] Building KG...", end=" ", flush=True)
        kg = builder.build_kg(sample)
        entity_index = builder.get_entity_index(kg)
        print(f"✓ ({kg.number_of_nodes()} nodes)")
        
        # Analyze question
        print("[2/3] Analyzing question...", end=" ", flush=True)
        qa = analyzer.analyze(question)
        print(f"✓ (type: {qa.question_type})")
        
        # Synthesize program
        print("[3/3] Synthesizing program...", end=" ", flush=True)
        result = synthesizer.synthesize(qa, kg, entity_index)
        print(f"✓")
        
        # Execute program
        if result.program:
            prog = result.program.replace('const_100', '100')
            try:
                answer = eval(prog)
                
                # Check correctness
                tolerance = 0.01
                is_correct = abs(answer - expected_answer) < tolerance
                
                elapsed = time.time() - start_time
                
                print(f"\nProgram: {result.program}")
                print(f"Answer: {answer}")
                print(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                print(f"Time: {elapsed:.2f}s")
                
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    'id': sample['id'],
                    'question_type': qa.question_type,
                    'correct': is_correct,
                    'expected': expected_answer,
                    'got': answer,
                    'time': elapsed
                })
                
            except Exception as e:
                print(f"\n❌ Execution error: {e}")
                total += 1
                results.append({
                    'id': sample['id'],
                    'question_type': qa.question_type,
                    'correct': False,
                    'error': str(e)
                })
        else:
            print("\n❌ No program generated")
            total += 1
            results.append({
                'id': sample['id'],
                'question_type': qa.question_type,
                'correct': False,
                'error': 'No program'
            })
            
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        total += 1
        results.append({
            'id': sample['id'],
            'correct': False,
            'error': str(e)
        })
    
    print()

# Print summary
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Total Samples: {total}")
print(f"Correct: {correct}")
print(f"Incorrect: {total - correct}")
print(f"Accuracy: {100 * correct / total if total > 0 else 0:.1f}%")
print()

# Breakdown by question type
print("By Question Type:")
type_stats = {}
for r in results:
    qtype = r.get('question_type', 'unknown')
    if qtype not in type_stats:
        type_stats[qtype] = {'correct': 0, 'total': 0}
    type_stats[qtype]['total'] += 1
    if r.get('correct', False):
        type_stats[qtype]['correct'] += 1

for qtype, stats in sorted(type_stats.items()):
    acc = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    print(f"  {qtype}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

print("\n" + "="*80)
print("PHASE 2 COMPLETE!")
print("="*80)
