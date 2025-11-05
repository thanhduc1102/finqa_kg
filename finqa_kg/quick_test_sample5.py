"""Quick test for Sample 5 (percentage_change with revenue)"""
import json
from src.pipeline.intelligent_kg_builder import IntelligentKGBuilder
from src.pipeline.question_analyzer import QuestionAnalyzer
from src.pipeline.program_synthesizer import ProgramSynthesizer

# Load Sample 5
with open('../FinQA/dataset/train.json', 'r') as f:
    data = json.load(f)
sample = data[4]

print("Sample:", sample['id'])
print("Question:", sample['qa']['question'])
print("Ground Truth:", sample['qa']['exe_ans'])
print("="*60)

# Build KG
print("\n[1] Building KG...")
builder = IntelligentKGBuilder()
kg = builder.build_kg(sample)
print(f"  ✓ KG: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")

# Analyze question
print("\n[2] Analyzing Question...")
analyzer = QuestionAnalyzer()
qa = analyzer.analyze(sample['qa']['question'])
print(f"  Type: {qa.question_type}")
print(f"  Entities: {qa.entities_mentioned}")
print(f"  Temporal: {qa.temporal_entities}")

# Get entity index
entity_index = builder.get_entity_index(kg)

# Synthesize program
print("\n[3] Synthesizing Program...")
synthesizer = ProgramSynthesizer()
result = synthesizer.synthesize(qa, kg, entity_index)

print("\n" + "="*60)
print(f"Program: {result.program}")
print(f"Placeholders: {result.placeholders}")

# Calculate answer manually
if result.program:
    # Execute program - already has values in it
    prog = result.program.replace('const_100', '100')
    
    # Define arithmetic operations as functions
    def multiply(a, b):
        return a * b
    def divide(a, b):
        return a / b if b != 0 else 0
    def subtract(a, b):
        return a - b
    def add(a, b):
        return a + b
    
    try:
        answer = eval(prog)
        print(f"Answer: {answer}")
        print(f"Expected: {sample['qa']['exe_ans']}")
        
        # Check if correct (within tolerance)
        tolerance = 0.01
        expected = float(sample['qa']['exe_ans'])
        is_correct = abs(answer - expected) < tolerance
        print(f"Correct: {'✅ YES' if is_correct else '❌ NO'}")
    except Exception as e:
        print(f"Error evaluating: {e}")
