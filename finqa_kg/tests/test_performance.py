"""
Performance tests for FinQA Knowledge Graph
"""
import asyncio
import sys
import os
from pathlib import Path
import time
import psutil
import logging
import cProfile
import pstats
from memory_profiler import profile

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from finqa_kg.src.builder import ModernFinQAKnowledgeGraph
from finqa_kg.src.query import ModernFinQAGraphQuery

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

@profile
async def test_build_performance():
    """Test graph building performance"""
    logger.info("\nTesting Graph Building Performance...")
    
    # Record starting memory
    start_memory = get_memory_usage()
    logger.info(f"Initial memory usage: {start_memory:.2f} MB")
    
    # Initialize graph
    start_time = time.time()
    kg = ModernFinQAKnowledgeGraph()
    
    # Build from test data
    test_file = Path(__file__).parent / "test_data.json"
    await kg.build_from_json(str(test_file))
    
    # Record metrics
    build_time = time.time() - start_time
    end_memory = get_memory_usage()
    memory_used = end_memory - start_memory
    
    logger.info("\nBuild Performance Metrics:")
    logger.info(f"Build time: {build_time:.2f} seconds")
    logger.info(f"Memory used: {memory_used:.2f} MB")
    logger.info(f"Nodes created: {kg.graph.number_of_nodes()}")
    logger.info(f"Edges created: {kg.graph.number_of_edges()}")
    
    return kg

@profile
async def test_query_performance(kg):
    """Test query performance"""
    logger.info("\nTesting Query Performance...")
    
    # Initialize query engine
    query = ModernFinQAGraphQuery(kg.graph)
    
    # Test semantic search performance
    logger.info("\nTesting semantic search performance:")
    search_times = []
    test_queries = [
        "revenue growth",
        "operating costs",
        "profit margin",
        "financial results"
    ]
    
    for test_query in test_queries:
        start_time = time.time()
        results = query.semantic_search(test_query, k=3)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        logger.info(f"\nQuery: {test_query}")
        logger.info(f"Search time: {search_time:.3f} seconds")
        logger.info(f"Results found: {len(results)}")
    
    avg_search_time = sum(search_times) / len(search_times)
    logger.info(f"\nAverage search time: {avg_search_time:.3f} seconds")
    
    # Test QA performance
    logger.info("\nTesting QA performance:")
    qa_times = []
    test_questions = [
        "What was the revenue?",
        "How much did operating costs decrease?",
        "What is the profit margin?"
    ]
    
    for question in test_questions:
        start_time = time.time()
        answer = await query.answer_question(question)
        qa_time = time.time() - start_time
        qa_times.append(qa_time)
        
        logger.info(f"\nQuestion: {question}")
        logger.info(f"Answer time: {qa_time:.3f} seconds")
        logger.info(f"Answer: {answer}")
    
    avg_qa_time = sum(qa_times) / len(qa_times)
    logger.info(f"\nAverage QA time: {avg_qa_time:.3f} seconds")

async def main():
    """Run all performance tests"""
    try:
        # Run with cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run tests
        kg = await test_build_performance()
        await test_query_performance(kg)
        
        profiler.disable()
        
        # Save profiling results
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.dump_stats(str(output_dir / 'profile_results.prof'))
        
        # Print top 20 time-consuming functions
        logger.info("\nTop 20 time-consuming functions:")
        stats.sort_stats('cumulative').print_stats(20)
        
        logger.info("\nAll performance tests completed!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())