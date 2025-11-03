"""
Basic test for FinQA Knowledge Graph
"""
import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from finqa_kg.src.builder import ModernFinQAKnowledgeGraph
from finqa_kg.src.query import ModernFinQAGraphQuery
from finqa_kg.src.visualization import GraphVisualizer, VisualizationConfig

async def test_basic():
    print("1. Testing basic graph building...")
    # Initialize graph
    kg = ModernFinQAKnowledgeGraph()
    
    # Build from test data
    test_file = Path(__file__).parent / "test_data.json"
    await kg.build_from_json(str(test_file))
    
    print(f"Graph built successfully!")
    print(f"Total nodes: {kg.graph.number_of_nodes()}")
    print(f"Total edges: {kg.graph.number_of_edges()}")
    
    print("\n2. Testing basic queries...")
    # Initialize query engine
    query = ModernFinQAGraphQuery(kg.graph)
    
    # Try semantic search
    print("\nTesting semantic search:")
    results = query.semantic_search("revenue growth", k=2)
    for idx, result in enumerate(results, 1):
        print(f"\nResult {idx}:")
        print(f"Score: {result.score:.3f}")
        print(f"Content: {result.content}")
    
    # Try question answering
    print("\nTesting question answering:")
    question = "What was the revenue growth?"
    answer = await query.answer_question(question)
    print(f"Q: {question}")
    print(f"A: {answer}")
    
    print("\n3. Testing visualization...")
    # Initialize visualizer
    vis = GraphVisualizer(kg.graph)
    
    # Create visualizations
    config = VisualizationConfig(
        node_size=800,
        font_size=8,
        width=1600,
        height=1000
    )
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("Generating static visualization...")
    vis.visualize_full_graph(
        config=config,
        output_path=str(output_dir / "test_graph.png")
    )
    
    print("Generating interactive visualization...")
    vis.create_interactive_visualization(
        str(output_dir / "test_graph.html"),
        config=config
    )
    
    print("\nTest completed! Check the output directory for visualizations.")

if __name__ == "__main__":
    asyncio.run(test_basic())