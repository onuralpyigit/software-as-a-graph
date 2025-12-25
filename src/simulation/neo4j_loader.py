#!/usr/bin/env python3
"""
Neo4j Graph Loader for Simulation
==================================

Loads graph data directly from Neo4j for simulation purposes.
Converts Neo4j graph structure to simulation-ready format.

Author: Software-as-a-Graph Research Project
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


class ComponentType(Enum):
    """Types of components in the system"""
    APPLICATION = "Application"
    BROKER = "Broker"
    NODE = "Node"
    TOPIC = "Topic"


class DependencyType(Enum):
    """Types of dependencies"""
    APP_TO_APP = "app_to_app"
    NODE_TO_NODE = "node_to_node"
    APP_TO_BROKER = "app_to_broker"
    NODE_TO_BROKER = "node_to_broker"


@dataclass
class Component:
    """Represents a system component"""
    id: str
    type: ComponentType
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state for simulation
    is_active: bool = True
    is_degraded: bool = False
    capacity: float = 1.0
    queue_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'properties': self.properties,
            'is_active': self.is_active,
            'is_degraded': self.is_degraded,
            'capacity': self.capacity,
            'queue_size': self.queue_size
        }


@dataclass
class Dependency:
    """Represents a dependency between components"""
    source: str
    target: str
    dependency_type: DependencyType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'type': self.dependency_type.value,
            'weight': self.weight,
            'properties': self.properties,
            'is_active': self.is_active
        }


@dataclass
class SimulationGraph:
    """Graph structure optimized for simulation"""
    components: Dict[str, Component] = field(default_factory=dict)
    dependencies: List[Dependency] = field(default_factory=list)
    
    # Indexes for fast lookup
    _outgoing: Dict[str, List[Dependency]] = field(default_factory=dict)
    _incoming: Dict[str, List[Dependency]] = field(default_factory=dict)
    _by_type: Dict[ComponentType, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        self._rebuild_indexes()
    
    def _rebuild_indexes(self):
        """Rebuild lookup indexes"""
        self._outgoing = {c: [] for c in self.components}
        self._incoming = {c: [] for c in self.components}
        self._by_type = {t: [] for t in ComponentType}
        
        for dep in self.dependencies:
            if dep.source in self._outgoing:
                self._outgoing[dep.source].append(dep)
            if dep.target in self._incoming:
                self._incoming[dep.target].append(dep)
        
        for comp_id, comp in self.components.items():
            self._by_type[comp.type].append(comp_id)
    
    def add_component(self, component: Component):
        """Add a component to the graph"""
        self.components[component.id] = component
        self._outgoing[component.id] = []
        self._incoming[component.id] = []
        self._by_type[component.type].append(component.id)
    
    def add_dependency(self, dependency: Dependency):
        """Add a dependency to the graph"""
        self.dependencies.append(dependency)
        if dependency.source in self._outgoing:
            self._outgoing[dependency.source].append(dependency)
        if dependency.target in self._incoming:
            self._incoming[dependency.target].append(dependency)
    
    def get_component(self, comp_id: str) -> Optional[Component]:
        """Get component by ID"""
        return self.components.get(comp_id)
    
    def get_outgoing(self, comp_id: str) -> List[Dependency]:
        """Get outgoing dependencies for a component"""
        return self._outgoing.get(comp_id, [])
    
    def get_incoming(self, comp_id: str) -> List[Dependency]:
        """Get incoming dependencies for a component"""
        return self._incoming.get(comp_id, [])
    
    def get_by_type(self, comp_type: ComponentType) -> List[str]:
        """Get component IDs by type"""
        return self._by_type.get(comp_type, [])
    
    def get_active_components(self) -> List[str]:
        """Get IDs of all active components"""
        return [c for c, comp in self.components.items() if comp.is_active]
    
    def get_dependents(self, comp_id: str) -> Set[str]:
        """Get components that depend on the given component"""
        return {dep.source for dep in self._incoming.get(comp_id, []) if dep.is_active}
    
    def get_dependencies_of(self, comp_id: str) -> Set[str]:
        """Get components that the given component depends on"""
        return {dep.target for dep in self._outgoing.get(comp_id, []) if dep.is_active}
    
    def calculate_reachability(self, from_comp: str) -> Set[str]:
        """Calculate all components reachable from a given component"""
        reachable = set()
        to_visit = [from_comp]
        
        while to_visit:
            current = to_visit.pop()
            if current in reachable:
                continue
            reachable.add(current)
            
            for dep in self._outgoing.get(current, []):
                if dep.is_active and dep.target not in reachable:
                    target_comp = self.components.get(dep.target)
                    if target_comp and target_comp.is_active:
                        to_visit.append(dep.target)
        
        return reachable
    
    def calculate_total_reachability(self) -> int:
        """Calculate sum of reachability from all active components"""
        total = 0
        for comp_id in self.get_active_components():
            total += len(self.calculate_reachability(comp_id))
        return total
    
    def count_connected_components(self) -> int:
        """Count weakly connected components"""
        if not self.components:
            return 0
        
        visited = set()
        count = 0
        
        for comp_id in self.components:
            if comp_id not in visited:
                count += 1
                # BFS to find all connected
                queue = [comp_id]
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    
                    # Add neighbors (both directions for weak connectivity)
                    for dep in self._outgoing.get(current, []):
                        if dep.target not in visited:
                            queue.append(dep.target)
                    for dep in self._incoming.get(current, []):
                        if dep.source not in visited:
                            queue.append(dep.source)
        
        return count
    
    def copy(self) -> 'SimulationGraph':
        """Create a deep copy for simulation"""
        new_graph = SimulationGraph()
        
        for comp_id, comp in self.components.items():
            new_comp = Component(
                id=comp.id,
                type=comp.type,
                properties=comp.properties.copy(),
                is_active=comp.is_active,
                is_degraded=comp.is_degraded,
                capacity=comp.capacity,
                queue_size=comp.queue_size
            )
            new_graph.components[comp_id] = new_comp
        
        for dep in self.dependencies:
            new_dep = Dependency(
                source=dep.source,
                target=dep.target,
                dependency_type=dep.dependency_type,
                weight=dep.weight,
                properties=dep.properties.copy(),
                is_active=dep.is_active
            )
            new_graph.dependencies.append(new_dep)
        
        new_graph._rebuild_indexes()
        return new_graph
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'components': {k: v.to_dict() for k, v in self.components.items()},
            'dependencies': [d.to_dict() for d in self.dependencies],
            'stats': {
                'component_count': len(self.components),
                'dependency_count': len(self.dependencies),
                'active_components': len(self.get_active_components()),
                'components_by_type': {
                    t.value: len(ids) for t, ids in self._by_type.items()
                }
            }
        }


class Neo4jGraphLoader:
    """
    Loads graph data from Neo4j for simulation.
    
    Works with DEPENDS_ON relationships and component types:
    Application, Broker, Node, Topic
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = 'neo4j'):
        """
        Initialize the loader.
        
        Args:
            uri: Neo4j bolt URI
            user: Database username
            password: Database password
            database: Database name
        """
        if not HAS_NEO4J:
            raise ImportError("neo4j driver required: pip install neo4j")
        
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger('Neo4jGraphLoader')
    
    def close(self):
        """Close the database connection"""
        self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def load_graph(self, 
                   dependency_types: Optional[List[str]] = None,
                   component_filter: Optional[Dict[str, Any]] = None) -> SimulationGraph:
        """
        Load the complete graph from Neo4j.
        
        Args:
            dependency_types: Filter by DEPENDS_ON types (default: all)
            component_filter: Additional filters for components
            
        Returns:
            SimulationGraph ready for simulation
        """
        self.logger.info("Loading graph from Neo4j...")
        
        graph = SimulationGraph()
        
        # Load components
        self._load_components(graph, component_filter)
        
        # Load dependencies
        self._load_dependencies(graph, dependency_types)
        
        # Rebuild indexes
        graph._rebuild_indexes()
        
        self.logger.info(f"Loaded {len(graph.components)} components, "
                        f"{len(graph.dependencies)} dependencies")
        
        return graph
    
    def _load_components(self, graph: SimulationGraph, 
                         component_filter: Optional[Dict[str, Any]] = None):
        """Load all components from Neo4j"""
        with self.driver.session(database=self.database) as session:
            # Load each component type
            for comp_type in ComponentType:
                query = f"""
                MATCH (n:{comp_type.value})
                RETURN n.id AS id, properties(n) AS props
                """
                
                result = session.run(query)
                for record in result:
                    comp_id = record['id']
                    props = record['props'] or {}
                    
                    # Apply filter if provided
                    if component_filter:
                        skip = False
                        for key, value in component_filter.items():
                            if props.get(key) != value:
                                skip = True
                                break
                        if skip:
                            continue
                    
                    component = Component(
                        id=comp_id,
                        type=comp_type,
                        properties={k: v for k, v in props.items() if k != 'id'}
                    )
                    graph.components[comp_id] = component
    
    def _load_dependencies(self, graph: SimulationGraph,
                           dependency_types: Optional[List[str]] = None):
        """Load DEPENDS_ON relationships from Neo4j"""
        if dependency_types is None:
            dependency_types = [dt.value for dt in DependencyType]
        
        type_filter = ' OR '.join([f"r.dependency_type = '{t}'" for t in dependency_types])
        
        with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (a)-[r:DEPENDS_ON]->(b)
            WHERE {type_filter}
            RETURN a.id AS source, 
                   b.id AS target,
                   r.dependency_type AS dep_type,
                   coalesce(r.weight, 1.0) AS weight,
                   properties(r) AS props
            """
            
            result = session.run(query)
            for record in result:
                source = record['source']
                target = record['target']
                
                # Only add if both endpoints exist
                if source not in graph.components or target not in graph.components:
                    continue
                
                dep_type_str = record['dep_type']
                try:
                    dep_type = DependencyType(dep_type_str)
                except ValueError:
                    dep_type = DependencyType.APP_TO_APP
                
                props = record['props'] or {}
                
                dependency = Dependency(
                    source=source,
                    target=target,
                    dependency_type=dep_type,
                    weight=record['weight'],
                    properties={k: v for k, v in props.items() 
                               if k not in ['dependency_type', 'weight']}
                )
                graph.dependencies.append(dependency)
    
    def get_component_stats(self) -> Dict[str, Any]:
        """Get statistics about components in the database"""
        stats = {}
        
        with self.driver.session(database=self.database) as session:
            # Component counts
            for comp_type in ComponentType:
                query = f"MATCH (n:{comp_type.value}) RETURN count(n) AS count"
                result = session.run(query).single()
                stats[comp_type.value] = result['count'] if result else 0
            
            # Dependency counts
            query = """
            MATCH ()-[r:DEPENDS_ON]->()
            RETURN r.dependency_type AS type, count(r) AS count
            """
            dep_counts = {}
            for record in session.run(query):
                dep_counts[record['type']] = record['count']
            stats['dependencies'] = dep_counts
            
            # Weight statistics
            query = """
            MATCH ()-[r:DEPENDS_ON]->()
            WHERE r.weight IS NOT NULL
            RETURN avg(r.weight) AS avg, 
                   min(r.weight) AS min, 
                   max(r.weight) AS max,
                   count(r.weight) AS count
            """
            result = session.run(query).single()
            if result and result['count'] > 0:
                stats['weight'] = {
                    'avg': result['avg'],
                    'min': result['min'],
                    'max': result['max'],
                    'count': result['count']
                }
        
        return stats
