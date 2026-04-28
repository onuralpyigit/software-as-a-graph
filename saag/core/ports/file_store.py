"""
File Store Port

Defines the IFileStore interface for filesystem operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class IFileStore(ABC):
    """Abstract interface for file storage operations."""
    
    @abstractmethod
    def read_json(self, path: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def write_json(self, path: str, data: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def read_text(self, path: str) -> str:
        pass

    @abstractmethod
    def write_text(self, path: str, content: str) -> str:
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        pass

    @abstractmethod
    def makedirs(self, path: str) -> None:
        pass
