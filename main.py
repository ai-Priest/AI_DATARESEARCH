#!/usr/bin/env python3
"""
Main Application Launcher for AI Dataset Research Assistant
=========================================================

This is the main entry point for the Singapore Dataset Discovery Assistant.
It provides a simple interface to start the full-stack application with
both backend API and frontend web interface.

Usage:
    python main.py              # Start the application
    python main.py --help       # Show help
    python main.py --backend    # Start only backend
    python main.py --frontend   # Start only frontend

Features:
- Dynamic performance monitoring with real-time metrics
- Real-time dataset search and recommendations
- Singapore government data integration
- Intelligent caching and optimization
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import List, Optional

# Set environment variables before any imports to avoid TensorFlow issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TORCH'] = '1'


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_production_metrics():
    """Display production performance metrics with real data"""
    try:
        from src.ai.performance_metrics_collector import PerformanceMetricsCollector
        
        # Create metrics collector
        collector = PerformanceMetricsCollector()
        
        # Get metrics (with timeout to avoid hanging)
        try:
            metrics = asyncio.run(asyncio.wait_for(collector.get_all_metrics(), timeout=5.0))
        except asyncio.TimeoutError:
            metrics = {}
        except Exception:
            metrics = {}
        
        # Format metrics for display
        formatted = collector.format_metrics_for_display(metrics)
        
        # Extract detailed metrics
        neural = metrics.get('neural_performance', {})
        response = metrics.get('response_time', {})
        cache = metrics.get('cache_performance', {})
        
        print(f"{Colors.CYAN}═══════════════ PRODUCTION METRICS ═══════════════{Colors.ENDC}")
        
        # Response time metrics - NO hardcoded values
        if response and 'average_response_time' in response:
            avg_time = response['average_response_time']
            print(f"{Colors.GREEN}📊 Performance: {avg_time:.2f}s average response time{Colors.ENDC}")
        elif response and 'estimated_response_time' in response:
            est_time = response['estimated_response_time']
            improvement = response.get('improvement_percentage', 0)
            print(f"{Colors.GREEN}📊 Performance: {est_time:.2f}s estimated response time ({improvement:.0f}% improvement){Colors.ENDC}")
        else:
            print(f"{Colors.GREEN}📊 Performance: Calculating response time metrics...{Colors.ENDC}")
        
        # Neural performance - NO hardcoded values
        if neural and 'ndcg_at_3' in neural and neural['ndcg_at_3'] > 0:
            ndcg_value = neural['ndcg_at_3']
            status = "(target exceeded!)" if ndcg_value > 70 else "(target: 70%)"
            print(f"{Colors.GREEN}🧠 Neural Performance: {ndcg_value:.1f}% NDCG@3 {status}{Colors.ENDC}")
            
            # Additional neural metrics if available
            if 'singapore_accuracy' in neural and neural['singapore_accuracy'] > 0:
                sg_acc = neural['singapore_accuracy']
                print(f"{Colors.GREEN}🇸🇬 Singapore Accuracy: {sg_acc:.1f}%{Colors.ENDC}")
        else:
            print(f"{Colors.GREEN}🧠 Neural Performance: Calculating NDCG@3 metrics...{Colors.ENDC}")
        
        # Multi-modal search - NO hardcoded values
        if response and 'min_response_time' in response:
            min_time = response['min_response_time']
            print(f"{Colors.GREEN}🔍 Multi-Modal Search: {min_time:.2f}s response time{Colors.ENDC}")
        else:
            print(f"{Colors.GREEN}🔍 Multi-Modal Search: Calculating response metrics...{Colors.ENDC}")
        
        # Cache performance - NO hardcoded values
        if cache and 'overall_hit_rate' in cache:
            hit_rate = cache['overall_hit_rate']
            print(f"{Colors.GREEN}🗄️ Intelligent Caching: {hit_rate:.1f}% hit rate{Colors.ENDC}")
        elif cache and 'documented_hit_rate' in cache:
            hit_rate = cache['documented_hit_rate']
            print(f"{Colors.GREEN}🗄️ Intelligent Caching: {hit_rate:.1f}% hit rate (documented){Colors.ENDC}")
        else:
            print(f"{Colors.GREEN}🗄️ Intelligent Caching: Calculating hit rate metrics...{Colors.ENDC}")
        
        # Timestamp and confidence level
        timestamp = formatted.get('last_updated', 'Just now')
        
        # Determine confidence level based on data quality
        if neural and 'ndcg_at_3' in neural and neural['ndcg_at_3'] > 0:
            confidence = '(High Confidence)'
        elif (response and 'average_response_time' in response) or \
             (cache and 'overall_hit_rate' in cache):
            confidence = '(Live Data)'
        elif ('estimated' in str(response)) or ('documented' in str(cache)):
            confidence = '(Estimated)'
        else:
            confidence = '(Calculating...)'
        
        print(f"{Colors.CYAN}📅 Metrics Updated: {timestamp} {confidence}{Colors.ENDC}")
        
    except Exception as e:
        # Fallback to calculating message if metrics collection fails
        print(f"{Colors.CYAN}═══════════════ PRODUCTION METRICS ═══════════════{Colors.ENDC}")
        print(f"{Colors.GREEN}📊 Performance: Calculating response time metrics...{Colors.ENDC}")
        print(f"{Colors.GREEN}🧠 Neural Performance: Calculating NDCG@3 metrics...{Colors.ENDC}")
        print(f"{Colors.GREEN}🔍 Multi-Modal Search: Calculating response metrics...{Colors.ENDC}")
        print(f"{Colors.GREEN}🗄️ Intelligent Caching: Calculating hit rate metrics...{Colors.ENDC}")
        print(f"{Colors.WARNING}⚠️ Metrics collection error: {str(e)[:50]}...{Colors.ENDC}")
    
    print(f"{Colors.CYAN}═══════════════════════════════════════════════════{Colors.ENDC}")
    print()

def run_background_mode():
    """Run in background daemon mode"""
    print("🚀 Starting server in background mode...")
    
    try:
        import daemon
        import daemon.pidfile
    except ImportError:
        print(f"{Colors.FAIL}❌ Background mode requires 'python-daemon' package{Colors.ENDC}")
        print(f"{Colors.CYAN}Install with: pip install python-daemon{Colors.ENDC}")
        return 1
    
    pid_file = Path(__file__).parent / "server.pid"
    log_file = Path(__file__).parent / "logs" / "background_server.log"
    
    # Ensure log directory exists
    log_file.parent.mkdir(exist_ok=True)
    
    with daemon.DaemonContext(
        pidfile=daemon.pidfile.PIDLockFile(str(pid_file)),
        stdout=open(log_file, 'a'),
        stderr=open(log_file, 'a'),
        working_directory=str(Path(__file__).parent)
    ):
        return run_production_deployment()

def run_production_deployment():
    """Run production deployment using organized structure"""
    project_root = Path(__file__).parent
    deployment_path = project_root / "src" / "deployment" / "start_production.py"
    
    if not deployment_path.exists():
        print(f"{Colors.FAIL}❌ Production deployment not found!{Colors.ENDC}")
        print(f"Expected: {deployment_path}")
        return 1
    
    print(f"📁 Using production deployment: {deployment_path}")
    print()
    
    # Set up environment variables
    env = os.environ.copy()
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'
    env['TRANSFORMERS_NO_TF'] = '1'
    env['USE_TORCH'] = '1'
    
    args = [sys.executable, str(deployment_path)]
    
    try:
        result = subprocess.run(args, cwd=project_root, env=env, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n🛑 Production deployment stopped by user")
        return 0
    except Exception as e:
        print(f"{Colors.FAIL}❌ Production deployment error: {e}{Colors.ENDC}")
        return 1

def print_banner():
    """Display application banner with real performance metrics"""
    # Initialize all metrics as "Calculating..." - NO hardcoded values
    ndcg_display = 'Calculating...'
    ndcg_status = ''
    response_time = 'Calculating...'
    cache_hit_rate = 'Calculating...'
    last_updated = 'Just now'
    confidence_level = '(Calculating...)'
    
    try:
        from src.ai.performance_metrics_collector import PerformanceMetricsCollector
        
        # Create metrics collector
        collector = PerformanceMetricsCollector()
        
        # Get metrics (with timeout to avoid hanging)
        try:
            metrics = asyncio.run(asyncio.wait_for(collector.get_all_metrics(), timeout=3.0))
        except asyncio.TimeoutError:
            # Timeout - keep "Calculating..." values
            metrics = {}
        except Exception:
            # Any other error - keep "Calculating..." values
            metrics = {}
        
        # Format metrics for display only if we have valid data
        if metrics and not metrics.get('error'):
            formatted = collector.format_metrics_for_display(metrics)
            
            # Extract key metrics - only update if we have real data
            if formatted.get('ndcg_at_3') and formatted.get('ndcg_at_3') != 'Calculating...':
                ndcg_display = formatted.get('ndcg_at_3', 'Calculating...')
                ndcg_status = formatted.get('ndcg_status', '')
            
            if formatted.get('response_time') and formatted.get('response_time') != 'Calculating...':
                response_time = formatted.get('response_time', 'Calculating...')
            
            if formatted.get('cache_hit_rate') and formatted.get('cache_hit_rate') != 'Calculating...':
                cache_hit_rate = formatted.get('cache_hit_rate', 'Calculating...')
            
            # Update timestamp only if we have valid metrics
            if formatted.get('last_updated') and formatted.get('last_updated') != 'Just now':
                last_updated = formatted.get('last_updated', 'Just now')
            
            # Determine confidence level based on data quality
            neural_metrics = metrics.get('neural_performance', {})
            response_metrics = metrics.get('response_time', {})
            cache_metrics = metrics.get('cache_performance', {})
            
            # High confidence: Real neural data available
            if neural_metrics and 'ndcg_at_3' in neural_metrics and neural_metrics['ndcg_at_3'] > 0:
                confidence_level = '(High Confidence)'
            # Medium confidence: Some real data, some estimated
            elif (response_metrics and 'average_response_time' in response_metrics) or \
                 (cache_metrics and 'overall_hit_rate' in cache_metrics):
                confidence_level = '(Live Data)'
            # Low confidence: Mostly estimated/documented values
            elif ('estimated' in response_time.lower() if response_time != 'Calculating...' else False) or \
                 ('documented' in cache_hit_rate.lower() if cache_hit_rate != 'Calculating...' else False):
                confidence_level = '(Estimated)'
            # No confidence: Still calculating
            else:
                confidence_level = '(Calculating...)'
        
    except ImportError:
        # Performance metrics collector not available - keep "Calculating..." values
        pass
    except Exception:
        # Any other error - keep "Calculating..." values
        pass
    
    # Format display strings to fit banner width
    perf_line = f"{ndcg_display} {ndcg_status}".strip()
    if len(perf_line) > 45:
        perf_line = perf_line[:42] + "..."
    
    response_line = response_time
    if len(response_line) > 35:
        response_line = response_line[:32] + "..."
    
    cache_line = cache_hit_rate
    if len(cache_line) > 32:
        cache_line = cache_line[:29] + "..."
    
    # Include timestamp and confidence level
    updated_line = f"{last_updated} {confidence_level}".strip()
    if len(updated_line) > 36:
        updated_line = updated_line[:33] + "..."
    
    banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🇸🇬 Singapore Dataset Discovery Assistant 🇸🇬               ║
║                                                              ║
║    🎯 Performance: {perf_line:<45} ║
║    🚀 AI-Powered Dataset Search & Recommendations            ║
║    📊 Real Singapore Government Data Integration             ║
║                                                              ║
║    ⚡ Response Time: {response_line:<35}           ║
║    🗄️ Cache Performance: {cache_line:<32}           ║
║    📅 Last Updated: {updated_line:<36}           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
    print(banner)

def check_requirements():
    """Check if required dependencies are available"""
    print(f"{Colors.BLUE}🔍 Checking system requirements...{Colors.ENDC}")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print(f"{Colors.FAIL}❌ Python 3.8+ required. Current: {sys.version}{Colors.ENDC}")
        return False
    
    # Check required files
    required_files = [
        "src/deployment/production_api_server.py",
        "Frontend/index.html",
        "config/ai_config.yml",
        "data/processed/singapore_datasets.csv"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"{Colors.FAIL}❌ Missing required file: {file_path}{Colors.ENDC}")
            return False
    
    print(f"{Colors.GREEN}✅ All requirements satisfied{Colors.ENDC}")
    return True

def start_backend() -> Optional[subprocess.Popen]:
    """Start the backend API server"""
    print(f"{Colors.BLUE}🚀 Starting backend API server...{Colors.ENDC}")
    
    try:
        # Change to project directory
        os.chdir(Path(__file__).parent)
        
        # Start the backend server
        cmd = [sys.executable, "-m", "src.deployment.production_api_server"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait a moment to check if it started successfully
        time.sleep(2)
        
        if process.poll() is None:  # Still running
            print(f"{Colors.GREEN}✅ Backend server started (PID: {process.pid}){Colors.ENDC}")
            print(f"{Colors.CYAN}   API: http://localhost:8000{Colors.ENDC}")
            print(f"{Colors.CYAN}   Docs: http://localhost:8000/docs{Colors.ENDC}")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"{Colors.FAIL}❌ Backend server failed to start{Colors.ENDC}")
            if stderr:
                print(f"{Colors.WARNING}Error: {stderr}{Colors.ENDC}")
            return None
            
    except Exception as e:
        print(f"{Colors.FAIL}❌ Failed to start backend: {e}{Colors.ENDC}")
        return None

def start_frontend() -> Optional[subprocess.Popen]:
    """Start the frontend web server"""
    print(f"{Colors.BLUE}🌐 Starting frontend web server...{Colors.ENDC}")
    
    try:
        import http.server
        import socketserver
        from threading import Thread

        # Define port and directory
        PORT = 3002
        DIRECTORY = "Frontend"
        
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=DIRECTORY, **kwargs)
            
            def end_headers(self):
                # Add CORS headers
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                super().end_headers()
        
        # Start server in a separate thread
        httpd = socketserver.TCPServer(("", PORT), Handler)
        
        def serve():
            httpd.serve_forever()
        
        thread = Thread(target=serve, daemon=True)
        thread.start()
        
        print(f"{Colors.GREEN}✅ Frontend server started{Colors.ENDC}")
        print(f"{Colors.CYAN}   URL: http://localhost:{PORT}{Colors.ENDC}")
        
        return httpd
        
    except Exception as e:
        print(f"{Colors.FAIL}❌ Failed to start frontend: {e}{Colors.ENDC}")
        return None

def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Wait for server to be ready"""
    import urllib.error
    import urllib.request
    
    print(f"{Colors.BLUE}⏳ Waiting for server to be ready...{Colors.ENDC}")
    
    for i in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except urllib.error.URLError:
            time.sleep(1)
            if i % 5 == 0:
                print(f"   Still waiting... ({i}s)")
    
    return False

def open_browser():
    """Open the application in the default browser"""
    frontend_url = "http://localhost:3002"
    
    print(f"{Colors.BLUE}🌐 Opening application in browser...{Colors.ENDC}")
    
    try:
        webbrowser.open(frontend_url)
        print(f"{Colors.GREEN}✅ Browser opened: {frontend_url}{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.WARNING}⚠️ Could not open browser automatically: {e}{Colors.ENDC}")
        print(f"{Colors.CYAN}Please manually open: {frontend_url}{Colors.ENDC}")

def cleanup_processes(processes: List[subprocess.Popen]):
    """Cleanup running processes"""
    print(f"\n{Colors.BLUE}🛑 Shutting down services...{Colors.ENDC}")
    
    for process in processes:
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"{Colors.GREEN}✅ Process {process.pid} terminated gracefully{Colors.ENDC}")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"{Colors.WARNING}⚠️ Process {process.pid} force killed{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.WARNING}⚠️ Error terminating process: {e}{Colors.ENDC}")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Singapore Dataset Discovery Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Start full application (development mode)
  python main.py --backend          Start only backend API server
  python main.py --frontend         Start only frontend web server
  python main.py --production       Start in production mode with monitoring
  python main.py --production --background  Start as background daemon
  python main.py --help             Show this help message

The application will be available at:
  Frontend: http://localhost:3002
  Backend API: http://localhost:8000
  API Documentation: http://localhost:8000/docs
        """
    )
    
    parser.add_argument('--backend', action='store_true', 
                       help='Start only the backend API server')
    parser.add_argument('--frontend', action='store_true', 
                       help='Start only the frontend web server')
    parser.add_argument('--no-browser', action='store_true', 
                       help='Don\'t open browser automatically')
    parser.add_argument('--production', action='store_true',
                       help='Use production deployment with monitoring and logging')
    parser.add_argument('--background', action='store_true',
                       help='Run in background daemon mode (production only)')
    
    args = parser.parse_args()
    
    # Handle background mode first
    if args.background:
        if not args.production:
            print(f"{Colors.FAIL}❌ Background mode requires --production flag{Colors.ENDC}")
            sys.exit(1)
        return run_background_mode()
    
    # Display banner
    print_banner()
    
    # Show production metrics if in production mode
    if args.production:
        print_production_metrics()
    
    # Check requirements
    if not check_requirements():
        print(f"{Colors.FAIL}❌ Requirements check failed. Please fix the issues above.{Colors.ENDC}")
        sys.exit(1)
    
    processes = []
    
    try:
        # Handle production mode
        if args.production:
            print(f"{Colors.BLUE}🏭 Starting in production mode...{Colors.ENDC}")
            return run_production_deployment()
        
        # Start services based on arguments (development mode)
        if args.backend or not (args.frontend):
            backend_process = start_backend()
            if backend_process:
                processes.append(backend_process)
            else:
                print(f"{Colors.FAIL}❌ Failed to start backend. Exiting.{Colors.ENDC}")
                sys.exit(1)
        
        if args.frontend or not (args.backend):
            frontend_server = start_frontend()
            if not frontend_server:
                print(f"{Colors.FAIL}❌ Failed to start frontend. Exiting.{Colors.ENDC}")
                sys.exit(1)
        
        # Wait for backend to be ready
        if not args.frontend:
            if wait_for_server("http://localhost:8000/api/health"):
                print(f"{Colors.GREEN}✅ Backend server is ready{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}⚠️ Backend server may not be fully ready{Colors.ENDC}")
        
        # Open browser if full app mode
        if not args.backend and not args.frontend and not args.no_browser:
            time.sleep(2)  # Give frontend a moment to start
            open_browser()
        
        # Display final status
        print(f"\n{Colors.GREEN}🎉 Application is running!{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        
        if not args.frontend:
            print(f"{Colors.CYAN}📡 Backend API: http://localhost:8000{Colors.ENDC}")
            print(f"{Colors.CYAN}📚 API Docs: http://localhost:8000/docs{Colors.ENDC}")
        
        if not args.backend:
            print(f"{Colors.CYAN}🌐 Frontend: http://localhost:3002{Colors.ENDC}")
        
        # Get current performance for status display - NO hardcoded values
        try:
            from src.ai.performance_metrics_collector import PerformanceMetricsCollector
            collector = PerformanceMetricsCollector()
            metrics = asyncio.run(asyncio.wait_for(collector.get_all_metrics(), timeout=2.0))
            formatted = collector.format_metrics_for_display(metrics)
            perf_display = formatted.get('ndcg_at_3', 'Calculating...')
            perf_status = formatted.get('ndcg_status', '')
            
            # Add timestamp and confidence info
            last_updated = formatted.get('last_updated', 'Just now')
            neural_metrics = metrics.get('neural_performance', {})
            response_metrics = metrics.get('response_time', {})
            cache_metrics = metrics.get('cache_performance', {})
            
            # Determine confidence level based on data quality
            if neural_metrics and 'ndcg_at_3' in neural_metrics and neural_metrics['ndcg_at_3'] > 0:
                confidence = '(High Confidence)'
            elif (response_metrics and 'average_response_time' in response_metrics) or \
                 (cache_metrics and 'overall_hit_rate' in cache_metrics):
                confidence = '(Live Data)'
            elif ('estimated' in perf_display.lower() if perf_display != 'Calculating...' else False):
                confidence = '(Estimated)'
            else:
                confidence = '(Calculating...)'
            
            print(f"{Colors.CYAN}📊 Performance: {perf_display} {perf_status} {confidence}{Colors.ENDC}")
            print(f"{Colors.CYAN}📅 Metrics Updated: {last_updated}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.CYAN}📊 Performance: Calculating metrics...{Colors.ENDC}")
            print(f"{Colors.WARNING}⚠️ Metrics error: {str(e)[:40]}...{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        print(f"{Colors.CYAN}🔧 Mode: {'Production' if args.production else 'Development'}{Colors.ENDC}")
        print(f"{Colors.WARNING}Press Ctrl+C to stop the application{Colors.ENDC}")
        
        # Keep the application running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    except Exception as e:
        print(f"{Colors.FAIL}❌ Unexpected error: {e}{Colors.ENDC}")
    
    finally:
        cleanup_processes(processes)
        print(f"{Colors.GREEN}✅ Application stopped successfully{Colors.ENDC}")

if __name__ == "__main__":
    main()