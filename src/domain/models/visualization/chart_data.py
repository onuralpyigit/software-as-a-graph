"""
Chart Data Domain Models
"""
from dataclasses import dataclass
from typing import Dict

@dataclass
class ChartOutput:
    """Output from chart generation."""
    title: str
    png_base64: str
    description: str = ""
    alt_text: str = ""
    width: int = 600
    height: int = 400

@dataclass
class ColorTheme:
    """Configurable color theme for charts."""
    # Primary semantic colors
    primary: str = "#3498db"
    secondary: str = "#2c3e50"
    success: str = "#2ecc71"
    warning: str = "#f39c12"
    danger: str = "#e74c3c"
    info: str = "#17a2b8"
    light: str = "#ecf0f1"
    dark: str = "#34495e"
    
    # Criticality level colors
    critical: str = "#e74c3c"
    high: str = "#e67e22"
    medium: str = "#f1c40f"
    low: str = "#2ecc71"
    minimal: str = "#95a5a6"
    
    # Layer-specific colors
    layer_app: str = "#3498db"
    layer_infra: str = "#9b59b6"
    layer_mw_app: str = "#1abc9c"
    layer_mw_infra: str = "#e67e22"
    layer_system: str = "#2c3e50"
    
    def to_colors_dict(self) -> Dict[str, str]:
        """Convert to COLORS dictionary format."""
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "success": self.success,
            "warning": self.warning,
            "danger": self.danger,
            "info": self.info,
            "light": self.light,
            "dark": self.dark,
        }
    
    def to_criticality_dict(self) -> Dict[str, str]:
        """Convert to CRITICALITY_COLORS dictionary format."""
        return {
            "CRITICAL": self.critical,
            "HIGH": self.high,
            "MEDIUM": self.medium,
            "LOW": self.low,
            "MINIMAL": self.minimal,
        }
    
    def to_layer_dict(self) -> Dict[str, str]:
        """Convert to LAYER_COLORS dictionary format."""
        return {
            "app": self.layer_app,
            "infra": self.layer_infra,
            "mw-app": self.layer_mw_app,
            "mw-infra": self.layer_mw_infra,
            "system": self.layer_system,
        }

DEFAULT_THEME = ColorTheme()
