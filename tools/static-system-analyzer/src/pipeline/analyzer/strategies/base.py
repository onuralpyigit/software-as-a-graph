"""Base class for analysis strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..models import TopicEntry


class AnalysisStrategy(ABC):
    """Abstract base for topic extraction strategies.

    Each strategy extracts TopicEntry results (pub, sub, uses) from a project
    directory and returns them in a unified format.
    """

    @abstractmethod
    def extract(self, folder_path: Path, folder_name: str) -> List[TopicEntry]:
        """Extract topic entries from a project.

        Args:
            folder_path: Absolute path to the versioned project directory.
            folder_name: Base project name (without version suffix).

        Returns:
            List of TopicEntry objects (pub, sub, uses).
        """
