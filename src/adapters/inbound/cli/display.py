"""
Console Display Adapter

Re-export from legacy location for now.
Future: Move display logic here completely.
"""

# Re-export from legacy location during migration
from src.cli.display import ConsoleDisplay

__all__ = ["ConsoleDisplay"]
