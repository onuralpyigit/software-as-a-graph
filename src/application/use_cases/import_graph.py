from dataclasses import dataclass
from typing import Dict, Any, Optional

from ..ports import GraphRepository

@dataclass
class ImportGraphUseCase:
    """Use case for importing graph data into the system."""
    
    repository: GraphRepository
    
    def execute(self, data: Dict[str, Any], clear: bool = False) -> Dict[str, int]:
        """
        Import graph data using the injected repository.
        Returns statistics about imported components.
        """
        self.repository.save_graph(data, clear=clear)
        return self.repository.get_statistics()
