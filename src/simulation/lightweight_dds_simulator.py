"""
Lightweight Event-Driven DDS Simulator

High-performance, in-memory pub-sub simulation using Python asyncio.
Designed for large-scale simulations with hundreds of nodes, thousands
of applications/topics, and tens of brokers.

Architecture:
- Event-driven using asyncio
- In-memory message routing
- Configurable QoS policies
- Resource tracking
- Performance metrics
- Scalable to 10,000+ components

Features:
- No container overhead
- Millisecond-level precision
- Real-time metrics
- Memory efficient
- Configurable parallelism
"""

import asyncio
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import json
import logging
from pathlib import Path
import heapq
from enum import Enum


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    URGENT = 3


@dataclass
class Message:
    """Lightweight message structure"""
    id: str
    topic: str
    sender: str
    payload_size: int
    timestamp: float
    priority: MessagePriority = MessagePriority.MEDIUM
    deadline_ms: Optional[float] = None
    ttl_ms: Optional[float] = None
    sequence: int = 0
    
    def is_expired(self, current_time: float) -> bool:
        """Check if message has exceeded TTL"""
        if self.ttl_ms is None:
            return False
        return (current_time - self.timestamp) * 1000 > self.ttl_ms
    
    def missed_deadline(self, current_time: float) -> bool:
        """Check if message missed its deadline"""
        if self.deadline_ms is None:
            return False
        return (current_time - self.timestamp) * 1000 > self.deadline_ms


@dataclass
class SimulationStats:
    """Statistics for simulation run"""
    messages_sent: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    messages_expired: int = 0
    deadline_misses: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    bytes_transferred: int = 0
    
    def avg_latency_ms(self) -> float:
        """Calculate average latency"""
        if self.messages_delivered == 0:
            return 0.0
        return self.total_latency_ms / self.messages_delivered
    
    def delivery_rate(self) -> float:
        """Calculate message delivery rate"""
        if self.messages_sent == 0:
            return 0.0
        return self.messages_delivered / self.messages_sent
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'messages_sent': self.messages_sent,
            'messages_delivered': self.messages_delivered,
            'messages_dropped': self.messages_dropped,
            'messages_expired': self.messages_expired,
            'deadline_misses': self.deadline_misses,
            'avg_latency_ms': self.avg_latency_ms(),
            'max_latency_ms': self.max_latency_ms,
            'min_latency_ms': self.min_latency_ms if self.min_latency_ms != float('inf') else 0,
            'delivery_rate': self.delivery_rate(),
            'bytes_transferred': self.bytes_transferred
        }


class Topic:
    """Lightweight topic with QoS and message queue"""
    
    def __init__(self, topic_id: str, name: str, qos: Dict):
        self.id = topic_id
        self.name = name
        self.qos = qos
        self.subscribers: Set[str] = set()
        self.history: deque = deque(maxlen=qos.get('history_depth', 1))
        self.message_count = 0
        
    def add_subscriber(self, app_id: str):
        """Add subscriber to topic"""
        self.subscribers.add(app_id)
    
    def remove_subscriber(self, app_id: str):
        """Remove subscriber from topic"""
        self.subscribers.discard(app_id)
    
    def store_message(self, message: Message):
        """Store message in history"""
        self.history.append(message)
        self.message_count += 1


class Application:
    """Lightweight application (publisher/subscriber/prosumer)"""
    
    def __init__(self, app_id: str, name: str, app_type: str):
        self.id = app_id
        self.name = name
        self.type = app_type
        self.publish_topics: List[Tuple[str, int, int]] = []  # (topic_id, period_ms, msg_size)
        self.subscribe_topics: Set[str] = set()
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.sequence = 0
        self.running = False
        self.stats = SimulationStats()
        
    def add_publisher(self, topic_id: str, period_ms: int, msg_size: int):
        """Configure as publisher to topic"""
        self.publish_topics.append((topic_id, period_ms, msg_size))
    
    def add_subscriber(self, topic_id: str):
        """Configure as subscriber to topic"""
        self.subscribe_topics.add(topic_id)


class Broker:
    """Lightweight broker for message routing"""
    
    def __init__(self, broker_id: str, name: str):
        self.id = broker_id
        self.name = name
        self.topics: Dict[str, Topic] = {}
        self.stats = SimulationStats()
        self.routing_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        
    def register_topic(self, topic: Topic):
        """Register topic with broker"""
        self.topics[topic.id] = topic
    
    async def route_message(self, message: Message, simulator: 'LightweightDDSSimulator'):
        """Route message to subscribers"""
        topic = self.topics.get(message.topic)
        if not topic:
            self.stats.messages_dropped += 1
            return
        
        # Store in history if configured
        topic.store_message(message)
        
        # Route to subscribers
        delivered = 0
        for sub_id in topic.subscribers:
            app = simulator.applications.get(sub_id)
            if app and app.running:
                try:
                    await app.message_queue.put(message)
                    delivered += 1
                except asyncio.QueueFull:
                    self.stats.messages_dropped += 1
        
        self.stats.messages_delivered += delivered


class LightweightDDSSimulator:
    """
    High-performance event-driven DDS simulator
    
    Scales to:
    - 100+ nodes
    - 1,000+ applications
    - 10,000+ topics
    - 10+ brokers
    
    Uses asyncio for concurrent execution without container overhead
    """
    
    def __init__(self):
        self.applications: Dict[str, Application] = {}
        self.topics: Dict[str, Topic] = {}
        self.brokers: Dict[str, Broker] = {}
        self.nodes: Dict[str, Dict] = {}
        
        self.global_stats = SimulationStats()
        self.start_time: float = 0
        self.current_time: float = 0
        self.duration_seconds: int = 0
        self.running = False
        
        self.logger = logging.getLogger(__name__)
        
    def load_from_json(self, json_path: str):
        """Load topology from graph JSON"""
        
        self.logger.info(f"Loading topology from {json_path}...")
        
        with open(json_path) as f:
            data = json.load(f)
        
        # Load nodes
        for node in data.get('nodes', []):
            self.nodes[node['id']] = node
        
        # Load brokers
        for broker_data in data.get('brokers', []):
            broker = Broker(broker_data['id'], broker_data.get('name', broker_data['id']))
            self.brokers[broker.id] = broker
        
        # Load topics
        for topic_data in data.get('topics', []):
            topic = Topic(
                topic_data['id'],
                topic_data['name'],
                topic_data.get('qos', {})
            )
            self.topics[topic.id] = topic
            
            # Register with brokers (find which brokers route this topic)
            for route in data.get('relationships', {}).get('routes', []):
                if route['to'] == topic.id:
                    broker = self.brokers.get(route['from'])
                    if broker:
                        broker.register_topic(topic)
        
        # Load applications
        for app_data in data.get('applications', []):
            app = Application(
                app_data['id'],
                app_data.get('name', app_data['id']),
                app_data.get('type', 'PROSUMER')
            )
            self.applications[app.id] = app
        
        # Configure publish relationships
        for pub in data.get('relationships', {}).get('publishes_to', []):
            app = self.applications.get(pub['from'])
            if app:
                app.add_publisher(
                    pub['to'],
                    pub.get('period_ms', 1000),
                    pub.get('msg_size', 1024)
                )
        
        # Configure subscribe relationships
        for sub in data.get('relationships', {}).get('subscribes_to', []):
            app = self.applications.get(sub['from'])
            topic = self.topics.get(sub['to'])
            if app and topic:
                app.add_subscriber(sub['to'])
                topic.add_subscriber(app.id)
        
        self.logger.info(f"Loaded: {len(self.nodes)} nodes, "
                        f"{len(self.applications)} applications, "
                        f"{len(self.topics)} topics, "
                        f"{len(self.brokers)} brokers")
    
    async def run_simulation(self, duration_seconds: int) -> Dict[str, Any]:
        """
        Run simulation for specified duration
        
        Args:
            duration_seconds: Simulation duration
            
        Returns:
            Dictionary with simulation results
        """
        self.logger.info(f"Starting simulation for {duration_seconds} seconds...")
        
        self.duration_seconds = duration_seconds
        self.start_time = time.time()
        self.running = True
        
        # Start all applications
        tasks = []
        
        # Start publishers
        for app in self.applications.values():
            if app.publish_topics:
                tasks.append(asyncio.create_task(self._run_publisher(app)))
        
        # Start subscribers
        for app in self.applications.values():
            if app.subscribe_topics:
                tasks.append(asyncio.create_task(self._run_subscriber(app)))
        
        # Start brokers
        for broker in self.brokers.values():
            tasks.append(asyncio.create_task(self._run_broker(broker)))
        
        # Wait for duration
        try:
            await asyncio.sleep(duration_seconds)
        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted")
        
        # Shutdown
        self.running = False
        
        # Wait for tasks to complete
        await asyncio.sleep(0.5)  # Grace period
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        results = self._collect_results()
        
        elapsed = time.time() - self.start_time
        self.logger.info(f"Simulation completed in {elapsed:.2f}s")
        
        return results
    
    async def _run_publisher(self, app: Application):
        """Run publisher application"""
        app.running = True
        
        try:
            while self.running:
                for topic_id, period_ms, msg_size in app.publish_topics:
                    # Create message
                    app.sequence += 1
                    message = Message(
                        id=f"{app.id}_{app.sequence}",
                        topic=topic_id,
                        sender=app.id,
                        payload_size=msg_size,
                        timestamp=time.time(),
                        sequence=app.sequence
                    )
                    
                    # Send to broker
                    await self._send_message(message)
                    
                    # Update stats
                    app.stats.messages_sent += 1
                    self.global_stats.messages_sent += 1
                    
                    # Wait for period
                    await asyncio.sleep(period_ms / 1000.0)
                    
        except asyncio.CancelledError:
            pass
        finally:
            app.running = False
    
    async def _run_subscriber(self, app: Application):
        """Run subscriber application"""
        app.running = True
        
        try:
            while self.running:
                try:
                    # Wait for message
                    message = await asyncio.wait_for(
                        app.message_queue.get(),
                        timeout=1.0
                    )
                    
                    # Process message
                    current_time = time.time()
                    latency_ms = (current_time - message.timestamp) * 1000
                    
                    # Update stats
                    app.stats.messages_delivered += 1
                    app.stats.total_latency_ms += latency_ms
                    app.stats.max_latency_ms = max(app.stats.max_latency_ms, latency_ms)
                    app.stats.min_latency_ms = min(app.stats.min_latency_ms, latency_ms)
                    app.stats.bytes_transferred += message.payload_size
                    
                    # Check deadline
                    if message.missed_deadline(current_time):
                        app.stats.deadline_misses += 1
                    
                    # Check expiration
                    if message.is_expired(current_time):
                        app.stats.messages_expired += 1
                        
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            pass
        finally:
            app.running = False
    
    async def _run_broker(self, broker: Broker):
        """Run broker message routing"""
        
        try:
            while self.running:
                try:
                    message = await asyncio.wait_for(
                        broker.routing_queue.get(),
                        timeout=1.0
                    )
                    
                    await broker.route_message(message, self)
                    
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            pass
    
    async def _send_message(self, message: Message):
        """Send message through broker"""
        
        # Find broker for topic
        broker = None
        for b in self.brokers.values():
            if message.topic in b.topics:
                broker = b
                break
        
        if broker:
            try:
                await broker.routing_queue.put(message)
            except asyncio.QueueFull:
                self.global_stats.messages_dropped += 1
    
    def _collect_results(self) -> Dict[str, Any]:
        """Collect simulation results"""
        
        # Aggregate stats from all components
        for app in self.applications.values():
            self.global_stats.messages_delivered += app.stats.messages_delivered
            self.global_stats.total_latency_ms += app.stats.total_latency_ms
            self.global_stats.max_latency_ms = max(
                self.global_stats.max_latency_ms,
                app.stats.max_latency_ms
            )
            if app.stats.min_latency_ms != float('inf'):
                self.global_stats.min_latency_ms = min(
                    self.global_stats.min_latency_ms,
                    app.stats.min_latency_ms
                )
            self.global_stats.bytes_transferred += app.stats.bytes_transferred
            self.global_stats.deadline_misses += app.stats.deadline_misses
            self.global_stats.messages_expired += app.stats.messages_expired
        
        for broker in self.brokers.values():
            self.global_stats.messages_dropped += broker.stats.messages_dropped
        
        # Calculate per-component stats
        app_stats = {}
        for app_id, app in self.applications.items():
            app_stats[app_id] = {
                'type': app.type,
                'messages_sent': app.stats.messages_sent,
                'messages_received': app.stats.messages_delivered,
                'avg_latency_ms': app.stats.avg_latency_ms(),
                'bytes_transferred': app.stats.bytes_transferred
            }
        
        broker_stats = {}
        for broker_id, broker in self.brokers.items():
            broker_stats[broker_id] = {
                'topics': len(broker.topics),
                'messages_routed': broker.stats.messages_delivered,
                'messages_dropped': broker.stats.messages_dropped
            }
        
        topic_stats = {}
        for topic_id, topic in self.topics.items():
            topic_stats[topic_id] = {
                'subscribers': len(topic.subscribers),
                'messages': topic.message_count,
                'history_size': len(topic.history)
            }
        
        return {
            'duration_seconds': self.duration_seconds,
            'elapsed_seconds': time.time() - self.start_time,
            'global_stats': self.global_stats.to_dict(),
            'component_counts': {
                'nodes': len(self.nodes),
                'applications': len(self.applications),
                'topics': len(self.topics),
                'brokers': len(self.brokers)
            },
            'application_stats': app_stats,
            'broker_stats': broker_stats,
            'topic_stats': topic_stats
        }
    
    def print_summary(self, results: Dict[str, Any]):
        """Print simulation summary"""
        
        print("\n" + "=" * 70)
        print("SIMULATION SUMMARY")
        print("=" * 70)
        
        stats = results['global_stats']
        counts = results['component_counts']
        
        print(f"\nDuration: {results['elapsed_seconds']:.2f}s")
        print(f"\nTopology:")
        print(f"  Nodes: {counts['nodes']}")
        print(f"  Applications: {counts['applications']}")
        print(f"  Topics: {counts['topics']}")
        print(f"  Brokers: {counts['brokers']}")
        
        print(f"\nMessage Statistics:")
        print(f"  Sent: {stats['messages_sent']:,}")
        print(f"  Delivered: {stats['messages_delivered']:,}")
        print(f"  Dropped: {stats['messages_dropped']:,}")
        print(f"  Expired: {stats['messages_expired']:,}")
        print(f"  Delivery Rate: {stats['delivery_rate']:.2%}")
        
        print(f"\nLatency:")
        print(f"  Average: {stats['avg_latency_ms']:.2f}ms")
        print(f"  Min: {stats['min_latency_ms']:.2f}ms")
        print(f"  Max: {stats['max_latency_ms']:.2f}ms")
        
        print(f"\nDeadlines:")
        print(f"  Missed: {stats['deadline_misses']:,}")
        
        print(f"\nData Transfer:")
        print(f"  Total: {stats['bytes_transferred'] / 1024 / 1024:.2f} MB")
        print(f"  Throughput: {stats['bytes_transferred'] / results['elapsed_seconds'] / 1024 / 1024:.2f} MB/s")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")


# Convenience function for running simulation
async def run_lightweight_simulation(json_path: str,
                                     duration_seconds: int,
                                     output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run lightweight DDS simulation
    
    Args:
        json_path: Path to graph JSON file
        duration_seconds: Simulation duration
        output_path: Optional path to save results
    
    Returns:
        Simulation results dictionary
    """
    simulator = LightweightDDSSimulator()
    simulator.load_from_json(json_path)
    results = await simulator.run_simulation(duration_seconds)
    
    simulator.print_summary(results)
    
    if output_path:
        simulator.save_results(results, output_path)
    
    return results
