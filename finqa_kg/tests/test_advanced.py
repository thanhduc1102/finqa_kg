"""
Advanced tests for FinQA Knowledge Graph
"""
import asyncio
import sys
import os
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from finqa_kg.src.builder import ModernFinQAKnowledgeGraph
from finqa_kg.src.builder.entity_extractor import EntityExtractor
from finqa_kg.src.builder.relation_extractor import RelationExtractor
from finqa_kg.src.query import ModernFinQAGraphQuery

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_entity_extraction():
    """Test entity extraction capabilities"""
    logger.info("\nTesting Entity Extraction...")
    
    extractor = EntityExtractor()
    
    # Test different types of text
    test_texts = [
        "Revenue grew to $1.2 million in Q3 2023",
        "Operating margin improved from 15% to 18%",
        "The company expects to launch new products by January 2024"
    ]
    
    for text in test_texts:
        logger.info(f"\nProcessing text: {text}")
        entities = await extractor.extract_all_entities(text)
        
        logger.info("Found entities:")
        for entity in entities:
            logger.info(f"- Type: {entity.type}")
            logger.info(f"  Text: {entity.text}")
            logger.info(f"  Confidence: {entity.confidence:.3f}")
            if entity.value is not None:
                logger.info(f"  Value: {entity.value}")

async def test_relation_extraction():
    """Test relation extraction capabilities"""
    logger.info("\nTesting Relation Extraction...")
    
    extractor = RelationExtractor()
    
    # Test text pairs
    text_pairs = [
        (
            "Revenue increased to $1.2M in Q3",
            "This growth represents a 15% increase from Q2"
        ),
        (
            "Operating costs were $800K",
            "This is a 5% reduction from previous quarter"
        )
    ]
    
    for text1, text2 in text_pairs:
        logger.info(f"\nAnalyzing relation between:")
        logger.info(f"Text 1: {text1}")
        logger.info(f"Text 2: {text2}")
        
        # Get semantic similarity
        similarity = extractor.compute_semantic_similarity(text1, text2)
        logger.info(f"Semantic similarity: {similarity:.3f}")
        
        # Extract relations
        relations = await extractor.extract_financial_relations(text1, text2)
        logger.info("\nExtracted relations:")
        for relation in relations:
            logger.info(f"- Type: {relation.relation_type}")
            logger.info(f"  Confidence: {relation.confidence:.3f}")

async def test_numerical_analysis():
    """Test numerical analysis capabilities"""
    logger.info("\nTesting Numerical Analysis...")
    
    # Initialize graph
    kg = ModernFinQAKnowledgeGraph()
    test_file = Path(__file__).parent / "test_data.json"
    await kg.build_from_json(str(test_file))
    
    # Initialize query engine
    query = ModernFinQAGraphQuery(kg.graph)
    
    # Test trend analysis
    logger.info("\nAnalyzing numerical trends:")
    trends = query.analyze_numerical_trends()
    
    for trend in trends:
        logger.info(f"\nContext: {trend['context']}")
        logger.info(f"Count: {trend['count']}")
        logger.info(f"Min: {trend['min']}")
        logger.info(f"Max: {trend['max']}")
        logger.info(f"Mean: {trend['mean']:.2f}")
        if 'trend_direction' in trend:
            logger.info(f"Direction: {trend['trend_direction']}")
        if 'volatility' in trend:
            logger.info(f"Volatility: {trend['volatility']:.3f}")

async def main():
    """Run all advanced tests"""
    try:
        await test_entity_extraction()
        await test_relation_extraction()
        await test_numerical_analysis()
        
        logger.info("\nAll advanced tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())