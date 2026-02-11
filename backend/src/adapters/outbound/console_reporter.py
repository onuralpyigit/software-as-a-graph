"""
Console Reporter Adapter

Implements IReporter interface for console output with colors.
"""

from typing import List, Any

from src.application.ports import IReporter


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    HEADER = "\033[95m"


class ConsoleReporter(IReporter):
    """
    Console implementation of IReporter.
    
    Provides formatted terminal output with colors and tables.
    """
    
    def __init__(self, use_color: bool = True):
        self.use_color = use_color
    
    def _color(self, text: str, color: str) -> str:
        """Apply color if enabled."""
        if self.use_color:
            return f"{color}{text}{Colors.RESET}"
        return text
    
    def info(self, message: str) -> None:
        """Display info message."""
        print(f"ℹ️  {message}")
    
    def success(self, message: str) -> None:
        """Display success message."""
        print(self._color(f"✅ {message}", Colors.GREEN))
    
    def warning(self, message: str) -> None:
        """Display warning message."""
        print(self._color(f"⚠️  {message}", Colors.YELLOW))
    
    def error(self, message: str) -> None:
        """Display error message."""
        print(self._color(f"❌ {message}", Colors.RED))
    
    def table(self, headers: List[str], rows: List[List[Any]]) -> None:
        """Display tabular data."""
        if not headers or not rows:
            return
        
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))
        
        # Print header
        header_line = " | ".join(
            str(h).ljust(widths[i]) for i, h in enumerate(headers)
        )
        print(self._color(header_line, Colors.BOLD))
        print("-" * len(header_line))
        
        # Print rows
        for row in rows:
            row_line = " | ".join(
                str(cell).ljust(widths[i]) if i < len(widths) else str(cell)
                for i, cell in enumerate(row)
            )
            print(row_line)
    
    def section(self, title: str) -> None:
        """Display section header."""
        line = "=" * (len(title) + 4)
        print()
        print(self._color(line, Colors.HEADER))
        print(self._color(f"  {title}  ", Colors.HEADER + Colors.BOLD))
        print(self._color(line, Colors.HEADER))
        print()
