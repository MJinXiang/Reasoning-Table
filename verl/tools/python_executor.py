import sys
import math
import numpy as np
from io import StringIO
import contextlib
import multiprocessing
from typing import Dict, Any
from .tool_base import Tool

@contextlib.contextmanager
def capture_stdout():
    """Capture standard output"""
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        yield mystdout
    finally:
        sys.stdout = old_stdout

class PythonExecutorTool(Tool):
    """Python code execution tool"""
    
    def __init__(self, timeout=5, memory_limit=100*1024*1024):
        """
        Initialize Python code execution tool
        
        Args:
            timeout: Execution timeout in seconds
            memory_limit: Memory limit in bytes
        """
        super().__init__(
            name="python_executor",
            description="Execute Python code and return the results.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        )
        self.timeout = timeout
        self.memory_limit = memory_limit
        
    def _get_safe_builtins(self):
        """Build a safe execution environment"""
        safe_modules = {
            'math': math,
            'np': np,
        }
        
        safe_builtins = {
            'print': print,
            'sum': sum,
            'min': min,
            'max': max,
            'round': round,
            'abs': abs,
            'range': range,
            'list': list,
            'dict': dict,
            'set': set,
            'int': int,
            'float': float,
            'str': str,
            'len': len,
            'zip': zip,
            'enumerate': enumerate,
        }
        
        # Add all safe functions from safe modules
        for module_name, module in safe_modules.items():
            safe_builtins[module_name] = module
            
        return safe_builtins
    
    def _execute_in_process(self, code):
        """Execute code in a separate process"""
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, self.memory_limit))
        
        safe_builtins = self._get_safe_builtins()
        
        with capture_stdout() as output:
            try:
                compiled_code = compile(code, '<string>', 'exec')
                exec(compiled_code, {"__builtins__": safe_builtins})
                return output.getvalue()
            except Exception as e:
                return f"Execution error: {str(e)}"
    
    def execute(self, args: Dict) -> Dict[str, Any]:
        """Execute Python code"""
        code = args.get("code", "")
        
        # Use multiprocessing to isolate execution
        with multiprocessing.Pool(1) as pool:
            try:
                result = pool.apply_async(self._execute_in_process, (code,))
                output = result.get(timeout=self.timeout)
                return {"content": f"REAL_TOOL_EXECUTION_ID_12345:\n{output}"}
            except multiprocessing.TimeoutError:
                pool.terminate()
                return {"content": "Execution timeout"}
            except Exception as e:
                return {"content": f"Execution error: {str(e)}"}