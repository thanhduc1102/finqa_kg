"""
Comprehensive demo of FinQA Knowledge Graph functionality
"""

import asyncio
import json
from pathlib import Path
import logging
from typing import Dict, Any

from finqa_kg.src.builder.knowledge_graph_builder import ModernFinQAKnowledgeGraph
from finqa_kg.src.query.knowledge_graph_query import ModernFinQAGraphQuery
from finqa_kg.src.visualization.graph_visualizer import (
    GraphVisualizer,
    VisualizationConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def build_knowledge_graph(data_path: str, max_samples: int = None) -> ModernFinQAKnowledgeGraph:
    """Build knowledge graph from FinQA dataset"""
    logger.info("Initializing Knowledge Graph Builder...")
    kg = ModernFinQAKnowledgeGraph()
    
    logger.info(f"Building graph from {data_path}")
    await kg.build_from_json(data_path, max_samples=max_samples)
    
    return kg

async def demonstrate_querying(kg: ModernFinQAKnowledgeGraph):
    """Demonstrate various query capabilities"""
    logger.info("\nDemonstrating query capabilities...")
    
    # Initialize query engine
    query_engine = ModernFinQAGraphQuery(kg.graph)
    
    # 1. Semantic Search
    logger.info("\n1. Semantic Search Example:")
    query = "revenue growth in the last quarter"
    results = query_engine.semantic_search(query, k=3)
    for idx, result in enumerate(results, 1):
        logger.info(f"Result {idx}:")
        logger.info(f"Score: {result.score:.3f}")
        logger.info(f"Content: {result.content}")
        logger.info("---")
        
    # 2. Question Answering
    logger.info("\n2. Question Answering Example:")
    question = "What was the total revenue?"
    answer = await query_engine.answer_question(question)
    logger.info(f"Q: {question}")
    logger.info(f"A: {answer}")
    
    # 3. Find Related Numbers
    logger.info("\n3. Finding Related Numbers Example:")
    number = 1000000  # 1 million
    related = query_engine.find_related_numbers(number, tolerance=0.1)
    for idx, result in enumerate(related[:3], 1):
        logger.info(f"Related Number {idx}:")
        logger.info(f"Value: {result.metadata['number_value']}")
        logger.info(f"Context: {result.content}")
        logger.info("---")
        
    # 4. Analyze Numerical Trends
    logger.info("\n4. Numerical Trend Analysis Example:")
    trends = query_engine.analyze_numerical_trends(min_count=3)
    for idx, trend in enumerate(trends[:2], 1):
        logger.info(f"Trend {idx}:")
        logger.info(f"Context: {trend['context']}")
        logger.info(f"Direction: {trend.get('trend_direction', 'N/A')}")
        logger.info(f"Values: {trend['values']}")
        logger.info("---")

def demonstrate_visualization(kg: ModernFinQAKnowledgeGraph):
    """Demonstrate visualization capabilities"""
    logger.info("\nDemonstrating visualization capabilities...")
    
    # Initialize visualizer
    vis = GraphVisualizer(kg.graph)
    
    # Custom configuration
    config = VisualizationConfig(
        node_size=800,
        font_size=6,
        width=1600,
        height=1000,
        show_labels=True,
        show_edge_labels=True,
        layout="spring"
    )
    
    # 1. Full Graph Visualization
    logger.info("1. Creating full graph visualization...")
    vis.visualize_full_graph(
        config=config,
        output_path="full_graph.png"
    )
    
    # 2. Subgraph Visualization
    logger.info("2. Creating subgraph visualization...")
    # Get a document node for center
    doc_nodes = [n for n, d in kg.graph.nodes(data=True) if d.get('type') == 'document']
    if doc_nodes:
        vis.visualize_subgraph(
            doc_nodes[0],
            depth=2,
            config=config,
            output_path="subgraph.png"
        )
        
    # 3. Interactive Visualization
    logger.info("3. Creating interactive visualization...")
    vis.create_interactive_visualization(
        "interactive_graph.html",
        config=config
    )

async def main():
    # Set paths
    data_path = "data/finqa/train.json"
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return
        
    try:
        # Build Knowledge Graph
        kg = await build_knowledge_graph(data_path, max_samples=5)
        
        # Save the graph
        kg.save_graph("finqa_knowledge_graph.pkl")
        
        # Demonstrate querying
        await demonstrate_querying(kg)
        
        # Demonstrate visualization
        demonstrate_visualization(kg)
        
        logger.info("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demo: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())