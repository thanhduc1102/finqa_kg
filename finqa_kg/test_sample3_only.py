"""Test Sample 3 specifically"""
import json
from src.pipeline.intelligent_kg_builder import IntelligentKGBuilder
from src.pipeline.question_analyzer import QuestionAnalyzer
from src.pipeline.program_synthesizer import ProgramSynthesizer

# Load Sample 3
with open('../FinQA/dataset/train.json', 'r') as f:
    data = json.load(f)
sample = data[2]  # Sample 3 (0-indexed)

print("="*80)
print("TESTING SAMPLE 3 ONLY")
print("="*80)

# Initialize
builder = IntelligentKGBuilder()
analyzer = QuestionAnalyzer()
synthesizer = ProgramSynthesizer()

# Process
question = sample['qa']['question']
expected_answer = float(sample['qa']['exe_ans'])

print(f"Question: {question}")
print(f"Expected: {expected_answer}")
print(f"Ground Truth Program: {sample['qa']['program']}")

# Build KG
print("\n[1/3] Building KG...")
kg = builder.build_kg(sample)
entity_index = builder.get_entity_index(kg)
print(f"  ✓ {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")

# Analyze
print("[2/3] Analyzing question...")
qa = analyzer.analyze(question)
print(f"  ✓ Type: {qa.question_type}")

# Synthesize
print("[3/3] Synthesizing program...")
result = synthesizer.synthesize(qa, kg, entity_index)
print(f"  ✓ Done")

# Define operations
def multiply(a, b):
    return a * b
def divide(a, b):
    return a / b if b != 0 else 0
def subtract(a, b):
    return a - b
def add(a, b):
    return a + b

# Execute
if result.program:
    prog = result.program.replace('const_100', '100')
    try:
        answer = eval(prog)
        
        # Check
        tolerance = 0.01
        is_correct = abs(answer - expected_answer) < tolerance
        
        print(f"\nProgram: {result.program}")
        print(f"Answer: {answer}")
        print(f"Expected: {expected_answer}")
        print(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        
    except Exception as e:
        print(f"\n❌ Execution error: {e}")
else:
    print("\n❌ No program generated")
