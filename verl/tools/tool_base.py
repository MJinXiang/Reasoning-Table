from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import json
import re

class Tool(ABC):
    """Tool base class, defines common interface"""
    
    def __init__(self, name: str, description: str, parameters: Dict = None):
        """
        Initialize tool
        
        Args:
            name: Tool name
            description: Tool description
            parameters: JSON Schema compliant parameter definition
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Ensure parameters contain necessary fields
        if "type" not in self.parameters:
            self.parameters["type"] = "object"
        if "properties" not in self.parameters:
            self.parameters["properties"] = {}
        if "required" not in self.parameters:
            self.parameters["required"] = []
    
    def get_description(self) -> Dict:
        """Get tool description (JSON Schema format)"""
        return {
            "type": "function", 
            "function": {
                "name": self.name, 
                "description": self.description, 
                "parameters": self.parameters
            }
        }
    
    @abstractmethod
    def execute(self, args: Dict) -> Dict[str, Any]:
        """
        Execute tool functionality
        
        Args:
            args: Tool parameters

        Returns:
            Dictionary containing execution results
        """
        pass

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        """Batch execute tool (default implementation)"""
        return [self.execute(args) for args in args_list]
    
    def validate_args(self, args: Dict) -> Tuple[bool, str]:
        """Validate if tool arguments conform to the defined schema"""
        # Check required parameters
        if "required" in self.parameters:
            for param in self.parameters["required"]:
                if param not in args:
                    return False, f"Missing required parameter: {param}"
        
        # Check parameter types
        if "properties" in self.parameters:
            for param_name, param_info in self.parameters["properties"].items():
                if param_name in args:
                    value = args[param_name]
                    expected_type = param_info.get("type")
                    
                    if expected_type and not self._check_type(value, expected_type):
                        return False, f"Parameter {param_name} type error: should be {expected_type}"
        
        return True, ""

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value type matches expected type"""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        return True  # Unknown type passes by default