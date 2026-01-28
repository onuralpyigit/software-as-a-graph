from enum import Enum

class VertexType(str, Enum):
    APPLICATION = "Application"
    BROKER = "Broker"
    TOPIC = "Topic"
    NODE = "Node"
    LIBRARY = "Library"

class EdgeType(str, Enum):
    RUNS_ON = "RUNS_ON"
    ROUTES = "ROUTES"
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    CONNECTS_TO = "CONNECTS_TO"
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"

class DependencyType(str, Enum):
    APP_TO_APP = "app_to_app"
    NODE_TO_NODE = "node_to_node"
    APP_TO_BROKER = "app_to_broker"
    NODE_TO_BROKER = "node_to_broker"

class ApplicationType(str, Enum):
    """Type/category of an application in the pub-sub system."""
    SERVICE = "service"           # General microservice
    DRIVER = "driver"             # Hardware/sensor driver
    CONTROLLER = "controller"     # Control logic component
    GATEWAY = "gateway"           # External interface/API gateway
    PROCESSOR = "processor"       # Data processing component
    MONITOR = "monitor"           # Monitoring/observability component
    AGGREGATOR = "aggregator"     # Data aggregation component
    SCHEDULER = "scheduler"       # Task scheduling component
    LOGGER = "logger"             # Logging/audit component
    UI = "ui"                     # User interface component
