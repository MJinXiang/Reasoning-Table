import re
import json
from typing import Dict, List, Any, Optional
from .tool_base import Tool

class ToolManager:
    """Tool manager, handles tool registration, calling and result formatting"""
    
    def __init__(self):
        self.tools = {}  # Mapping from tool name to tool instance
    
    def register_tool(self, tool: Tool):
        """Register tool"""
        self.tools[tool.name] = tool
        return self
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def extract_tool_call(self, text: str) -> Dict:
        """
        Extract tool call from text
        
        Args:
            text: Input text
            
        Returns:
            Tool call info dictionary {"tool": "tool_name", "args": {parameters}}
        """
        # Regex to extract tool call <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        
        tool_call_match = re.search(tool_call_pattern, text, re.DOTALL)
        
        if not tool_call_match:
            return {"tool": "invalid", "args": {}}
        
        try:
            tool_call_json = tool_call_match.group(1).strip()
            tool_call_data = json.loads(tool_call_json)
            
            if "name" not in tool_call_data:
                return {"tool": "invalid", "args": {}}
                
            tool_name = tool_call_data["name"]
            tool_args = tool_call_data.get("arguments", {})
            
            return {"tool": tool_name, "args": tool_args}
        except (json.JSONDecodeError, Exception):
            return {"tool": "invalid", "args": {}}
    
    def execute_tool(self, call_info: Dict) -> Dict[str, Any]:
        """
        Execute tool call
        
        Args:
            call_info: Tool call information {"tool": "tool_name", "args": {parameters}}
            
        Returns:
            Execution result dictionary
        """
        tool_name = call_info.get("tool", "")
        args = call_info.get("args", {})
        
        # Tool doesn't exist
        if tool_name not in self.tools:
            return {"content": f"Unknown tool: {tool_name}"}
        
        tool = self.tools[tool_name]
        
        # Validate parameters
        is_valid, error_msg = tool.validate_args(args)
        if not is_valid:
            return {"content": f"Invalid parameters: {error_msg}"}
        
        # Execute tool
        try:
            result = tool.execute(args)
            if not isinstance(result, dict):
                result = {"content": str(result)}
            return result
        except Exception as e:
            return {"content": f"Execution error: {str(e)}"}
    
    def process_text(self, text: str) -> tuple:
        """
        Process text containing tool calls
        
        Args:
            text: Input text
            
        Returns:
            (Processed text, whether tool was executed)
        """
        call_info = self.extract_tool_call(text)
        
        # If no valid tool call
        if call_info["tool"] == "invalid":
            return text, False
        
        # Execute tool
        result = self.execute_tool(call_info)
        result_content = result.get("content", "")
        
        # Build result text, replace original tool call part
        tool_call_match = re.search(r'<tool_call>.*?</tool_call>', text, re.DOTALL)
        if tool_call_match:
            result_text = text[:tool_call_match.start()] + \
                        f"<observation>\n{result_content}\n</observation>\n\n" + \
                        text[tool_call_match.end():]
            return result_text, True
        else:
            return text, False
    
    def get_tool_descriptions(self) -> str:
        """Get descriptions of all tools"""
        result = []
        for tool in self.tools.values():
            result.append(json.dumps(tool.get_description(), ensure_ascii=False))
        return "\n".join(result)