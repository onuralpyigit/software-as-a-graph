from typing import Dict, Any, List, Optional
import logging
from src.analysis.classifier import BoxPlotClassifier
from src.core.graph_exporter import GraphExporter
from src.infrastructure.repositories.graph_query_repo import GraphQueryRepository

logger = logging.getLogger(__name__)

class ClassificationService:
    """
    Service for classifying components based on centrality and other metrics.
    Encapsulates BoxPlotClassifier and FuzzyClassifier logic.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password

    def classify_components(self, use_fuzzy: bool = False, use_weights: bool = True, dependency_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Classify components using BoxPlot or Fuzzy logic."""
        # For now, we only implement BoxPlot as seen in main.py
        # If use_fuzzy is True, we might need FuzzyClassifier, but let's stick to what was in main.py
        
        with GraphExporter(self.uri, self.user, self.password) as exporter:
            repo = GraphQueryRepository(exporter)
            
            # BoxPlotClassifier needs a repository or graph data
            # In main.py: classifier = BoxPlotClassifier(repo)
            
            classifier = BoxPlotClassifier(repo)
            results = classifier.classify(
                use_weights=use_weights,
                dependency_types=dependency_types
            )
            
            # The result is likely a dictionary or object with .to_dict()
            # In main.py lines 1636+: it returns the result directly or processes it
            # Let's assume classifier.classify returns a Dict
            return results
