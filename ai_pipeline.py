"""
AI Pipeline - Main entry point for AI-enhanced dataset research
Orchestrates neural recommendations with intelligent LLM enhancement
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path
import time
from typing import Optional, Dict, Any
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ai.ai_config_manager import AIConfigManager
from src.ai.research_assistant import ResearchAssistant
from src.ai.api_server import app
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIPipeline:
    """
    Main AI Pipeline orchestrator
    Combines 69.99% NDCG@3 neural performance with intelligent AI enhancement
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = AIConfigManager(config_path)
        self.config = self.config_manager.config
        self.research_assistant = ResearchAssistant(self.config)
        
    async def interactive_mode(self):
        """Run interactive command-line interface"""
        print("\n🤖 AI-Powered Dataset Research Assistant")
        print("=" * 60)
        print("Neural Model: Lightweight Cross-Attention Ranker (69.99% NDCG@3)")
        print("Type 'help' for commands, 'exit' to quit\n")
        
        session_id = None
        
        while True:
            try:
                # Get user input
                query = input("\n🔍 Enter your research query: ").strip()
                
                # Handle commands
                if query.lower() == 'exit':
                    print("👋 Thank you for using the research assistant!")
                    break
                    
                elif query.lower() == 'help':
                    self._print_help()
                    continue
                    
                elif query.lower() == 'new':
                    session_id = None
                    print("🆕 Started new session")
                    continue
                    
                elif query.lower() == 'metrics':
                    await self._show_metrics()
                    continue
                    
                elif query.startswith('refine:'):
                    if not session_id:
                        print("❌ No active session. Please make a query first.")
                        continue
                    refinement = query[7:].strip()
                    response = await self.research_assistant.refine_query(session_id, refinement)
                    self._display_response(response)
                    continue
                
                # Process regular query
                if query:
                    print("\n⏳ Processing your query...")
                    start_time = time.time()
                    
                    response = await self.research_assistant.process_query(
                        query=query,
                        session_id=session_id
                    )
                    
                    # Update session ID
                    session_id = response.get('session_id')
                    
                    # Display results
                    self._display_response(response)
                    
                    print(f"\n⏱️  Total processing time: {response['processing_time']:.2f}s")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
                
            except Exception as e:
                logger.error(f"Error in interactive mode: {str(e)}")
                print(f"\n❌ Error: {str(e)}")
    
    def _print_help(self):
        """Print help information"""
        print("\n📚 Available Commands:")
        print("  help          - Show this help message")
        print("  new           - Start a new session")
        print("  metrics       - Show system metrics")
        print("  refine:<text> - Refine previous query")
        print("  exit          - Exit the program")
        print("\n💡 Tips:")
        print("  - Be specific in your queries for better results")
        print("  - Use 'refine:' to improve results based on feedback")
        print("  - The system combines neural ranking with AI explanations")
    
    def _display_response(self, response: Dict[str, Any]):
        """Display formatted response"""
        print("\n" + "=" * 60)
        print("📊 RECOMMENDATIONS")
        print("=" * 60)
        
        # Display recommendations
        for i, rec in enumerate(response.get('recommendations', []), 1):
            dataset = rec.get('dataset', {})
            print(f"\n{i}. {dataset.get('title', 'Unknown Dataset')}")
            print(f"   📍 Source: {dataset.get('source', 'Unknown')}")
            print(f"   📂 Category: {dataset.get('category', 'General')}")
            print(f"   🎯 Confidence: {rec.get('confidence', 0)*100:.0f}%")
            print(f"   📈 Quality Score: {rec.get('quality_score', 0)*100:.0f}%")
            print(f"   💡 Why: {rec.get('explanation', 'Recommended by neural model')}")
        
        # Display relationships
        if response.get('dataset_relationships'):
            print("\n🔗 DATASET RELATIONSHIPS:")
            for rel in response['dataset_relationships'][:3]:
                print(f"   • {rel}")
        
        # Display analytical approaches
        if response.get('analytical_approaches'):
            print("\n🔬 SUGGESTED ANALYSES:")
            for approach in response['analytical_approaches'][:3]:
                print(f"   • {approach}")
        
        # Display Singapore context if relevant
        if response.get('singapore_context'):
            print("\n🇸🇬 SINGAPORE CONTEXT:")
            print(f"   {response['singapore_context'][:200]}...")
        
        # Display refinement suggestions
        if response.get('conversation', {}).get('suggested_refinements'):
            print("\n💭 REFINEMENT SUGGESTIONS:")
            for suggestion in response['conversation']['suggested_refinements']:
                print(f"   • {suggestion}")
        
        # Performance info
        perf = response.get('performance', {})
        print(f"\n🚀 Powered by: {perf.get('neural_model', {}).get('model', 'Neural Model')} + {perf.get('ai_provider', 'AI')}")
    
    async def _show_metrics(self):
        """Show system metrics"""
        try:
            metrics = await self.research_assistant.conversation_manager.evaluation_metrics.get_current_metrics()
            
            print("\n📊 SYSTEM METRICS")
            print("=" * 60)
            
            # User satisfaction
            satisfaction = metrics.get('user_satisfaction', {})
            print(f"\n😊 User Satisfaction:")
            print(f"   Current Rate: {satisfaction.get('current_rate', 0):.1%}")
            print(f"   Target: {satisfaction.get('threshold', 0.85):.1%}")
            print(f"   Status: {'✅ Meeting target' if satisfaction.get('meets_target') else '⚠️  Below target'}")
            
            # Performance
            performance = metrics.get('performance', {})
            print(f"\n⚡ Performance:")
            print(f"   Avg Response Time: {performance.get('avg_response_time', 0):.2f}s")
            print(f"   Total Requests: {performance.get('total_requests', 0)}")
            
            # Neural performance
            neural = metrics.get('neural_performance', {})
            print(f"\n🧠 Neural Model:")
            print(f"   Model: {neural.get('model', 'Unknown')}")
            print(f"   NDCG@3: {neural.get('ndcg_at_3', 0):.1%}")
            
        except Exception as e:
            print(f"❌ Could not retrieve metrics: {str(e)}")
    
    async def process_batch(self, queries_file: str, output_file: str):
        """Process a batch of queries from file"""
        logger.info(f"Processing batch from {queries_file}")
        
        try:
            # Read queries
            with open(queries_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            results = []
            
            # Process each query
            for i, query in enumerate(queries, 1):
                print(f"\nProcessing query {i}/{len(queries)}: {query}")
                
                try:
                    response = await self.research_assistant.process_query(query)
                    
                    # Extract key information
                    result = {
                        "query": query,
                        "recommendations": [
                            {
                                "title": rec['dataset']['title'],
                                "source": rec['dataset']['source'],
                                "confidence": rec['confidence']
                            }
                            for rec in response['recommendations'][:3]
                        ],
                        "processing_time": response['processing_time']
                    }
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {str(e)}")
                    results.append({
                        "query": query,
                        "error": str(e)
                    })
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✅ Batch processing complete. Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            raise
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the API server"""
        logger.info(f"Starting API server on {host}:{port}")
        
        # Run the server
        uvicorn.run(
            "src.ai.api_server:app",
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI-Powered Dataset Research Assistant"
    )
    
    parser.add_argument(
        '--mode',
        choices=['interactive', 'api', 'batch', 'test'],
        default='interactive',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--queries',
        type=str,
        help='Input file for batch mode'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results.json',
        help='Output file for batch mode'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='API server host'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='API server port'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AIPipeline(args.config)
    
    # Execute based on mode
    if args.mode == 'interactive':
        await pipeline.interactive_mode()
        
    elif args.mode == 'api':
        pipeline.start_api_server(args.host, args.port)
        
    elif args.mode == 'batch':
        if not args.queries:
            print("❌ Batch mode requires --queries file")
            return
        await pipeline.process_batch(args.queries, args.output)
        
    elif args.mode == 'test':
        # Run tests
        from test_ai_system import main as test_main
        await test_main()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)