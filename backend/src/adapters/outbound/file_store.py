"""
File Store Adapter

Implements IFileStore interface for local filesystem operations.
"""

import json
import os
from typing import Dict, Any

from src.application.ports import IFileStore


class LocalFileStore(IFileStore):
    """
    Local filesystem implementation of IFileStore.
    
    Provides file I/O operations for JSON and text files.
    """
    
    def read_json(self, path: str) -> Dict[str, Any]:
        """Read JSON file and return parsed content."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def write_json(self, path: str, data: Dict[str, Any]) -> str:
        """Write data as JSON to file. Returns the written path."""
        self.makedirs(os.path.dirname(path))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return path
    
    def read_text(self, path: str) -> str:
        """Read text file and return content."""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def write_text(self, path: str, content: str) -> str:
        """Write text content to file. Returns the written path."""
        self.makedirs(os.path.dirname(path))
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path
    
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        return os.path.exists(path)
    
    def makedirs(self, path: str) -> None:
        """Create directory and parents if they don't exist."""
        if path:
            os.makedirs(path, exist_ok=True)
