import math
import numpy as np
from .tool_base import Tool
from typing import Dict, Any

class CalculatorTool(Tool):
    """Calculator tool"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Execute mathematical expression calculations, supporting basic operations and functions.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '2 + 2' or 'sin(0.5)'"
                    }
                },
                "required": ["expression"]
            }
        )
    
    def execute(self, args: Dict) -> Dict[str, Any]:
        """Execute mathematical expression calculation"""
        expression = args.get("expression", "")
        
        # Create safe math environment
        safe_dict = {
            'abs': abs, 'round': round,
            'min': min, 'max': max,
            'sum': sum, 'pow': pow,
            'int': int, 'float': float,
            'math': math, 'np': np
        }
        
        # Add all math module functions
        for name in dir(math):
            if not name.startswith('_'):
                safe_dict[name] = getattr(math, name)
        
        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return {"content": str(result)}
        except Exception as e:
            return {"content": f"Calculation error: {str(e)}"}