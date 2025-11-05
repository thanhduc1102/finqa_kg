"""Quick test for Sample 3 (direct_lookup with DATE filtering)"""
import json
import time
from src.pipeline.intelligent_kg_builder import IntelligentKGBuilder
from src.pipeline.question_analyzer import QuestionAnalyzer
from src.pipeline.program_synthesizer import ProgramSynthesizer

# Load Sample 3
with open('../FinQA/dataset/train.json', 'r') as f:
    data = json.load(f)
sample = data[2]

print("="*80)
print("SAMPLE 3 TEST - Phase 3 Fix")
print("="*80)
print(f"ID: {sample['id']}")
print(f"Question: {sample['qa']['question']}")
print(f"Expected: {sample['qa']['exe_ans']}")
print()

# Initialize
builder = IntelligentKGBuilder()
analyzer = QuestionAnalyzer()
synthesizer = ProgramSynthesizer()

# Build KG
print("[1/3] Building KG...", flush=True)
kg = builder.build_kg(sample)
entity_index = builder.get_entity_index(kg)
print(f"  ✓ {kg.number_of_nodes()} nodes")

# Analyze
print("[2/3] Analyzing question...")
qa = analyzer.analyze(sample['qa']['question'])
print(f"  ✓ Type: {qa.question_type}")
print(f"  ✓ Entities: {qa.entities_mentioned}")

# Synthesize
print("[3/3] Synthesizing program...")
result = synthesizer.synthesize(qa, kg, entity_index)
print(f"  ✓ Program: {result.program}")

# Execute
def divide(a, b):
    return a / b if b != 0 else 0
def multiply(a, b):
    return a * b
def subtract(a, b):
    return a - b
def add(a, b):
    return a + b

if result.program:
    prog = result.program.replace('const_100', '100')
    try:
        answer = eval(prog)
        expected = float(sample['qa']['exe_ans'])
        is_correct = abs(answer - expected) < 0.01
        
        print()
        print("="*80)
        print(f"Answer: {answer}")
        print(f"Expected: {expected}")
        print(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        print("="*80)
    except Exception as e:
        print(f"\n❌ Error: {e}")
