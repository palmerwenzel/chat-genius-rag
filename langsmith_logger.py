from langsmith import Client
from functools import wraps
from typing import Any, Callable, Dict, Optional
import os
import inspect
import logging

class LangSmithLogger:
    def __init__(self):
        self.client = Client()
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "chat-genius-rag")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def trace_chain(self, chain_name: str):
        """Decorator to trace LangChain operations."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get function metadata
                metadata = {
                    "function_name": func.__name__,
                    "module": func.__module__,
                    "chain": chain_name,
                    **{k: str(v) for k, v in kwargs.items()}  # Log kwargs as metadata
                }
                
                try:
                    # Start run
                    run = self.client.create_run(
                        name=f"{chain_name}.{func.__name__}",
                        inputs=kwargs,
                        run_type="chain",
                        project_name=self.project_name,
                        metadata=metadata
                    )
                    
                    # Execute function
                    result = await func(*args, **kwargs)
                    
                    # End run successfully
                    self.client.update_run(
                        run.id,
                        outputs={"result": result},
                        status="completed"
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log error if something goes wrong
                    if 'run' in locals():
                        self.client.update_run(
                            run.id,
                            error=str(e),
                            status="failed"
                        )
                    self.logger.error(f"Error in {chain_name}: {str(e)}")
                    raise
                    
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Similar to async but for synchronous functions
                metadata = {
                    "function_name": func.__name__,
                    "module": func.__module__,
                    "chain": chain_name,
                    **{k: str(v) for k, v in kwargs.items()}
                }
                
                try:
                    run = self.client.create_run(
                        name=f"{chain_name}.{func.__name__}",
                        inputs=kwargs,
                        run_type="chain",
                        project_name=self.project_name,
                        metadata=metadata
                    )
                    
                    result = func(*args, **kwargs)
                    
                    self.client.update_run(
                        run.id,
                        outputs={"result": result},
                        status="completed"
                    )
                    
                    return result
                    
                except Exception as e:
                    if 'run' in locals():
                        self.client.update_run(
                            run.id,
                            error=str(e),
                            status="failed"
                        )
                    self.logger.error(f"Error in {chain_name}: {str(e)}")
                    raise
            
            # Return appropriate wrapper based on if function is async
            return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        return decorator

# Initialize global logger
langsmith_logger = LangSmithLogger() 