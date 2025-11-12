#!/usr/bin/env python3
"""
Comprehensive Pub-Sub System Simulation Demo
============================================

This demo showcases the complete simulation capabilities:

1. Message Traffic Simulation - Realistic DDS message passing
2. Failure Injection - Single, multiple, and cascading failures
3. Performance Impact Assessment - Latency, throughput, delivery rate
4. Real-Time Monitoring - Progress tracking and live metrics
5. Multiple Scenarios - Baseline, SPOF, cascading, recovery
6. Comprehensive Reporting - JSON outputs and summaries

Author: Onuralp
Research: Graph-Based Modeling and Analysis of Distributed Publish-Subscribe Systems
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics


# ANSI Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str, color=Colors.HEADER):
    print(f"\n{color}{'='*80}")
    print(f"{text.center(80)}")
    print(f"{'='*80}{Colors.ENDC}\n")


def print_section(text: str):
    print(f"\n{Colors.CYAN}{'-'*80}")
    print(f"{text}")
    print(f"{'-'*80}{Colors.ENDC}")


@dataclass
class Message:
    """Lightweight message"""
    id: str
    topic: str
    sender: str
    payload_size: int
    timestamp: float
    priority: int = 1
    deadline_ms: Optional[float] = None
    ttl_ms: Optional[float] = None
    sequence: int = 0
    
    def is_expired(self, current_time: float) -> bool:
        if self.ttl_ms is None:
            return False
        return (current_time - self.timestamp) * 1000 > self.ttl_ms
    
    def missed_deadline(self, current_time: float) -> bool:
        if self.deadline_ms is None:
            return False
        return (current_time - self.timestamp) * 1000 > self.deadline_ms


@dataclass
class SimulationStats:
    """Statistics tracking"""
    messages_sent: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    messages_expired: int = 0
    deadline_misses: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    bytes_transferred: int = 0
    latencies: List[float] = None
    
    def __post_init__(self):
        if self.latencies is None:
            self.latencies = []
    
    @property
    def avg_latency_ms(self) -> float:
        if self.messages_delivered == 0:
            return 0.0
        return self.total_latency_ms / self.messages_delivered
    
    @property
    def delivery_rate(self) -> float:
        if self.messages_sent == 0:
            return 0.0
        return self.messages_delivered / self.messages_sent
    
    def to_dict(self) -> Dict:
        return {
            'messages_sent': self.messages_sent,
            'messages_delivered': self.messages_delivered,
            'messages_dropped': self.messages_dropped,
            'messages_expired': self.messages_expired,
            'deadline_misses': self.deadline_misses,
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'max_latency_ms': round(self.max_latency_ms, 2),
            'min_latency_ms': round(self.min_latency_ms, 2) if self.min_latency_ms != float('inf') else 0,
            'delivery_rate': round(self.delivery_rate, 4),
            'bytes_transferred': self.bytes_transferred
        }


class Application:
    """Simulated application"""
    def __init__(self, app_id: str, name: str, app_type: str, publish_rate_hz: float = 1.0):
        self.id = app_id
        self.name = name
        self.app_type = app_type
        self.publish_rate_hz = publish_rate_hz
        self.publishes: List[str] = []
        self.subscribes: List[str] = []
        self.running = True
        self.stats = SimulationStats()
        self.message_queue = asyncio.Queue(maxsize=1000)
        self.message_counter = 0
        
    def should_publish(self, elapsed: float) -> bool:
        if self.publish_rate_hz == 0:
            return False
        expected_messages = int(elapsed * self.publish_rate_hz)
        return expected_messages > self.message_counter


class Topic:
    """Simulated topic"""
    def __init__(self, topic_id: str, name: str, qos: Dict = None):
        self.id = topic_id
        self.name = name
        self.qos = qos or {}
        self.publishers: List[str] = []
        self.subscribers: List[str] = []


class Broker:
    """Simulated broker"""
    def __init__(self, broker_id: str, name: str):
        self.id = broker_id
        self.name = name
        self.topics: Dict[str, Topic] = {}
        self.routing_queue = asyncio.Queue(maxsize=10000)
        self.stats = SimulationStats()
        self.running = True
        self.routing_delay_ms = 5.0


class SimpleDDSSimulator:
    """Simplified event-driven DDS simulator"""
    
    def __init__(self):
        self.applications: Dict[str, Application] = {}
        self.topics: Dict[str, Topic] = {}
        self.brokers: Dict[str, Broker] = {}
        self.global_stats = SimulationStats()
        self.start_time: float = 0
        self.running = False
        self.monitor_interval = 5.0
        self.monitor_enabled = False
        self.monitor_history: List[Dict] = []
        
    def load_system(self, system_config: Dict):
        """Load system configuration"""
        for app_data in system_config.get('applications', []):
            app = Application(
                app_data['id'],
                app_data.get('name', app_data['id']),
                app_data.get('type', 'unknown'),
                app_data.get('publish_rate_hz', 1.0)
            )
            app.publishes = app_data.get('publishes', [])
            app.subscribes = app_data.get('subscribes', [])
            self.applications[app.id] = app
        
        for topic_data in system_config.get('topics', []):
            topic = Topic(
                topic_data['id'],
                topic_data.get('name', topic_data['id']),
                topic_data.get('qos', {})
            )
            self.topics[topic.id] = topic
        
        for broker_data in system_config.get('brokers', []):
            broker = Broker(broker_data['id'], broker_data.get('name', broker_data['id']))
            self.brokers[broker.id] = broker
        
        # Map topics to brokers
        for app in self.applications.values():
            for topic_id in app.publishes:
                if topic_id in self.topics:
                    self.topics[topic_id].publishers.append(app.id)
                    if self.brokers:
                        first_broker = list(self.brokers.values())[0]
                        first_broker.topics[topic_id] = self.topics[topic_id]
            
            for topic_id in app.subscribes:
                if topic_id in self.topics:
                    self.topics[topic_id].subscribers.append(app.id)
    
    async def run_simulation(self, duration_seconds: int, monitor: bool = False) -> Dict:
        """Run simulation"""
        self.running = True
        self.monitor_enabled = monitor
        self.start_time = time.time()
        
        tasks = []
        
        for app in self.applications.values():
            if app.publishes:
                tasks.append(asyncio.create_task(self._run_publisher(app, duration_seconds)))
            if app.subscribes:
                tasks.append(asyncio.create_task(self._run_subscriber(app)))
        
        for broker in self.brokers.values():
            tasks.append(asyncio.create_task(self._run_broker(broker)))
        
        if monitor:
            tasks.append(asyncio.create_task(self._monitor_progress(duration_seconds)))
        
        await asyncio.sleep(duration_seconds)
        self.running = False
        
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return self._collect_results(duration_seconds)
    
    async def _run_publisher(self, app: Application, duration: int):
        try:
            start = time.time()
            while self.running and (time.time() - start) < duration:
                elapsed = time.time() - start
                
                if app.should_publish(elapsed):
                    for topic_id in app.publishes:
                        message = Message(
                            id=f"{app.id}_{topic_id}_{app.message_counter}",
                            topic=topic_id,
                            sender=app.id,
                            payload_size=1024,
                            timestamp=time.time(),
                            sequence=app.message_counter
                        )
                        
                        if topic_id in self.topics:
                            qos = self.topics[topic_id].qos
                            message.deadline_ms = qos.get('deadline_ms', None)
                            message.ttl_ms = qos.get('ttl_ms', None)
                        
                        await self._send_message(message)
                        app.stats.messages_sent += 1
                        self.global_stats.messages_sent += 1
                        app.message_counter += 1
                
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
    
    async def _run_subscriber(self, app: Application):
        try:
            while self.running:
                try:
                    message = await asyncio.wait_for(app.message_queue.get(), timeout=0.5)
                    
                    current_time = time.time()
                    latency_ms = (current_time - message.timestamp) * 1000
                    
                    app.stats.messages_delivered += 1
                    app.stats.total_latency_ms += latency_ms
                    app.stats.max_latency_ms = max(app.stats.max_latency_ms, latency_ms)
                    app.stats.min_latency_ms = min(app.stats.min_latency_ms, latency_ms)
                    app.stats.bytes_transferred += message.payload_size
                    app.stats.latencies.append(latency_ms)
                    
                    if message.missed_deadline(current_time):
                        app.stats.deadline_misses += 1
                    if message.is_expired(current_time):
                        app.stats.messages_expired += 1
                    
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass
    
    async def _run_broker(self, broker: Broker):
        try:
            while self.running:
                try:
                    message = await asyncio.wait_for(broker.routing_queue.get(), timeout=0.5)
                    await asyncio.sleep(broker.routing_delay_ms / 1000.0)
                    
                    if message.topic in broker.topics:
                        topic = broker.topics[message.topic]
                        delivered = 0
                        
                        for sub_id in topic.subscribers:
                            if sub_id in self.applications:
                                app = self.applications[sub_id]
                                if app.running:
                                    try:
                                        await app.message_queue.put(message)
                                        delivered += 1
                                    except asyncio.QueueFull:
                                        broker.stats.messages_dropped += 1
                                        self.global_stats.messages_dropped += 1
                        
                        broker.stats.messages_delivered += delivered
                        self.global_stats.messages_delivered += delivered
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass
    
    async def _send_message(self, message: Message):
        for broker in self.brokers.values():
            if message.topic in broker.topics:
                try:
                    await broker.routing_queue.put(message)
                    return
                except asyncio.QueueFull:
                    self.global_stats.messages_dropped += 1
                    return
    
    async def _monitor_progress(self, duration: int):
        try:
            start = time.time()
            last_print = start
            
            while self.running and (time.time() - start) < duration:
                await asyncio.sleep(1.0)
                current = time.time()
                
                if current - last_print >= self.monitor_interval:
                    elapsed = current - start
                    progress = (elapsed / duration) * 100
                    throughput = self.global_stats.messages_delivered / elapsed if elapsed > 0 else 0
                    
                    print(f"{Colors.CYAN}[{elapsed:.0f}s/{duration}s] "
                          f"Progress: {progress:.1f}% | "
                          f"Sent: {self.global_stats.messages_sent} | "
                          f"Delivered: {self.global_stats.messages_delivered} | "
                          f"Throughput: {throughput:.1f} msg/s{Colors.ENDC}")
                    
                    self.monitor_history.append({
                        'timestamp': elapsed,
                        'messages_sent': self.global_stats.messages_sent,
                        'messages_delivered': self.global_stats.messages_delivered,
                        'throughput': throughput
                    })
                    
                    last_print = current
        except asyncio.CancelledError:
            pass
    
    def _collect_results(self, duration: float) -> Dict:
        all_latencies = []
        for app in self.applications.values():
            self.global_stats.total_latency_ms += app.stats.total_latency_ms
            self.global_stats.deadline_misses += app.stats.deadline_misses
            self.global_stats.messages_expired += app.stats.messages_expired
            self.global_stats.bytes_transferred += app.stats.bytes_transferred
            all_latencies.extend(app.stats.latencies)
        
        latency_p50 = statistics.median(all_latencies) if all_latencies else 0
        latency_p95 = statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) > 20 else 0
        latency_p99 = statistics.quantiles(all_latencies, n=100)[98] if len(all_latencies) > 100 else 0
        
        return {
            'duration_seconds': duration,
            'global_stats': self.global_stats.to_dict(),
            'throughput_msg_s': self.global_stats.messages_delivered / duration if duration > 0 else 0,
            'latency_percentiles': {
                'p50': round(latency_p50, 2),
                'p95': round(latency_p95, 2),
                'p99': round(latency_p99, 2)
            },
            'application_stats': {app_id: app.stats.to_dict() for app_id, app in self.applications.items()},
            'broker_stats': {broker_id: broker.stats.to_dict() for broker_id, broker in self.brokers.items()},
            'monitor_history': self.monitor_history if self.monitor_enabled else []
        }
    
    def inject_failure(self, component_id: str, failure_type: str = 'complete'):
        print(f"\n{Colors.RED}💥 INJECTING FAILURE: {component_id} ({failure_type}){Colors.ENDC}")
        
        if component_id in self.applications:
            app = self.applications[component_id]
            if failure_type == 'complete':
                app.running = False
                print(f"  Application {component_id} stopped")
            elif failure_type == 'partial':
                app.publish_rate_hz *= 0.5
                print(f"  Application {component_id} degraded (50% rate)")
        elif component_id in self.brokers:
            broker = self.brokers[component_id]
            if failure_type == 'complete':
                broker.running = False
                print(f"  Broker {component_id} stopped")
            elif failure_type == 'partial':
                broker.routing_delay_ms *= 2.0
                print(f"  Broker {component_id} degraded (2x delay)")
    
    def recover_component(self, component_id: str):
        print(f"\n{Colors.GREEN}🔄 RECOVERING: {component_id}{Colors.ENDC}")
        if component_id in self.applications:
            self.applications[component_id].running = True
        elif component_id in self.brokers:
            self.brokers[component_id].running = True


def create_smart_city_system() -> Dict:
    """Create Smart City IoT system"""
    return {
        "applications": [
            {"id": "TempSensor1", "name": "Temperature Sensor", "type": "sensor", 
             "publish_rate_hz": 2.0, "publishes": ["temperature_data"], "subscribes": []},
            {"id": "TrafficCamera1", "name": "Traffic Camera", "type": "sensor", 
             "publish_rate_hz": 10.0, "publishes": ["video_stream"], "subscribes": []},
            {"id": "AirQualitySensor1", "name": "Air Quality Monitor", "type": "sensor", 
             "publish_rate_hz": 0.5, "publishes": ["air_quality_data"], "subscribes": []},
            {"id": "AnalyticsEngine", "name": "Analytics", "type": "processing", 
             "publish_rate_hz": 1.0, "publishes": ["alerts", "analytics_results"],
             "subscribes": ["temperature_data", "air_quality_data", "video_stream"]},
            {"id": "AlertService", "name": "Alert System", "type": "service", 
             "publish_rate_hz": 0.1, "publishes": ["emergency_alerts"], "subscribes": ["alerts"]},
            {"id": "TrafficController", "name": "Traffic Control", "type": "actuator", 
             "publish_rate_hz": 0.0, "publishes": [],
             "subscribes": ["analytics_results", "emergency_alerts"]},
            {"id": "DataArchiver", "name": "Data Storage", "type": "storage", 
             "publish_rate_hz": 0.0, "publishes": [],
             "subscribes": ["temperature_data", "air_quality_data", "analytics_results"]},
            {"id": "Dashboard", "name": "Dashboard", "type": "visualization", 
             "publish_rate_hz": 0.0, "publishes": [],
             "subscribes": ["temperature_data", "air_quality_data", "video_stream", "alerts"]}
        ],
        "topics": [
            {"id": "temperature_data", "name": "Temperature", "qos": {"deadline_ms": 1000, "ttl_ms": 5000}},
            {"id": "air_quality_data", "name": "Air Quality", "qos": {"deadline_ms": 2000, "ttl_ms": 10000}},
            {"id": "video_stream", "name": "Video", "qos": {"deadline_ms": 100, "ttl_ms": 1000}},
            {"id": "alerts", "name": "Alerts", "qos": {"deadline_ms": 500, "ttl_ms": 2000}},
            {"id": "analytics_results", "name": "Analytics", "qos": {"deadline_ms": 1000, "ttl_ms": 5000}},
            {"id": "emergency_alerts", "name": "Emergency", "qos": {"deadline_ms": 100, "ttl_ms": 60000}}
        ],
        "brokers": [
            {"id": "Broker1", "name": "Edge Broker"},
            {"id": "Broker2", "name": "Cloud Broker"}
        ]
    }


def print_results(results: Dict, baseline: Optional[Dict] = None):
    """Print formatted results"""
    print_header("Simulation Results", Colors.GREEN)
    
    duration = results['duration_seconds']
    stats = results['global_stats']
    throughput = results['throughput_msg_s']
    
    print(f"{Colors.BOLD}⏱️  Execution:{Colors.ENDC}")
    print(f"   Duration: {duration}s")
    print(f"   Throughput: {throughput:.1f} msg/s")
    
    print(f"\n{Colors.BOLD}📊 Message Traffic:{Colors.ENDC}")
    print(f"   Sent:         {stats['messages_sent']:,}")
    print(f"   Delivered:    {stats['messages_delivered']:,}")
    print(f"   Dropped:      {stats['messages_dropped']:,}")
    print(f"   Delivery Rate: {stats['delivery_rate']*100:.2f}%")
    
    print(f"\n{Colors.BOLD}📈 Performance:{Colors.ENDC}")
    print(f"   Avg Latency:  {stats['avg_latency_ms']:.2f} ms")
    print(f"   Min Latency:  {stats['min_latency_ms']:.2f} ms")
    print(f"   Max Latency:  {stats['max_latency_ms']:.2f} ms")
    
    if 'latency_percentiles' in results:
        perc = results['latency_percentiles']
        print(f"   P50 Latency:  {perc['p50']:.2f} ms")
        print(f"   P95 Latency:  {perc['p95']:.2f} ms")
        print(f"   P99 Latency:  {perc['p99']:.2f} ms")
    
    print(f"   Deadline Misses: {stats['deadline_misses']}")
    print(f"   Data Transfer: {stats['bytes_transferred'] / (1024*1024):.2f} MB")
    
    if baseline:
        print(f"\n{Colors.BOLD}📊 Baseline Comparison:{Colors.ENDC}")
        baseline_stats = baseline['global_stats']
        
        latency_change = stats['avg_latency_ms'] - baseline_stats['avg_latency_ms']
        latency_pct = (latency_change / baseline_stats['avg_latency_ms'] * 100) if baseline_stats['avg_latency_ms'] > 0 else 0
        
        throughput_change = results['throughput_msg_s'] - baseline['throughput_msg_s']
        throughput_pct = (throughput_change / baseline['throughput_msg_s'] * 100) if baseline['throughput_msg_s'] > 0 else 0
        
        delivery_change = stats['delivery_rate'] - baseline_stats['delivery_rate']
        
        color = Colors.RED if latency_change > 0 else Colors.GREEN
        print(f"   {color}Latency Change:    {latency_change:+.2f} ms ({latency_pct:+.1f}%){Colors.ENDC}")
        
        color = Colors.GREEN if throughput_change > 0 else Colors.RED
        print(f"   {color}Throughput Change: {throughput_change:+.1f} msg/s ({throughput_pct:+.1f}%){Colors.ENDC}")
        
        color = Colors.GREEN if delivery_change > 0 else Colors.RED
        print(f"   {color}Delivery Change:   {delivery_change:+.4f} ({delivery_change*100:+.2f}%){Colors.ENDC}")


async def scenario_baseline(simulator, duration: int) -> Dict:
    """Baseline - No failures"""
    print_section("Scenario 1: Baseline (No Failures)")
    results = await simulator.run_simulation(duration, monitor=True)
    print(f"\n{Colors.GREEN}✓ Baseline completed{Colors.ENDC}")
    return results


async def scenario_single_failure(simulator, duration: int, component: str, failure_time: int) -> Dict:
    """Single component failure"""
    print_section(f"Scenario 2: Single Failure ({component})")
    sim_task = asyncio.create_task(simulator.run_simulation(duration, monitor=True))
    await asyncio.sleep(failure_time)
    simulator.inject_failure(component, 'complete')
    results = await sim_task
    print(f"\n{Colors.GREEN}✓ Single failure completed{Colors.ENDC}")
    return results


async def scenario_cascading(simulator, duration: int) -> Dict:
    """Cascading failures"""
    print_section("Scenario 3: Cascading Failures")
    sim_task = asyncio.create_task(simulator.run_simulation(duration, monitor=True))
    await asyncio.sleep(20)
    simulator.inject_failure('Broker1', 'complete')
    await asyncio.sleep(20)
    simulator.inject_failure('AnalyticsEngine', 'partial')
    results = await sim_task
    print(f"\n{Colors.GREEN}✓ Cascading completed{Colors.ENDC}")
    return results


async def scenario_recovery(simulator, duration: int) -> Dict:
    """Failure with recovery"""
    print_section("Scenario 4: Failure and Recovery")
    sim_task = asyncio.create_task(simulator.run_simulation(duration, monitor=True))
    await asyncio.sleep(20)
    simulator.inject_failure('TempSensor1', 'complete')
    await asyncio.sleep(30)
    simulator.recover_component('TempSensor1')
    results = await sim_task
    print(f"\n{Colors.GREEN}✓ Recovery completed{Colors.ENDC}")
    return results


def save_results(results: Dict, scenario_name: str, output_dir: Path):
    """Save results"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / f"{scenario_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"{Colors.GREEN}  ✓ Saved: {json_path}{Colors.ENDC}")
    
    summary_path = output_dir / f"{scenario_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Simulation: {scenario_name}\n{'='*60}\n\n")
        f.write(f"Duration: {results['duration_seconds']}s\n")
        f.write(f"Messages Sent: {results['global_stats']['messages_sent']:,}\n")
        f.write(f"Messages Delivered: {results['global_stats']['messages_delivered']:,}\n")
        f.write(f"Delivery Rate: {results['global_stats']['delivery_rate']*100:.2f}%\n")
        f.write(f"Avg Latency: {results['global_stats']['avg_latency_ms']:.2f} ms\n")
        f.write(f"Throughput: {results['throughput_msg_s']:.1f} msg/s\n")
    print(f"{Colors.GREEN}  ✓ Saved: {summary_path}{Colors.ENDC}")


async def main():
    """Main demo workflow"""
    print_header("Comprehensive Pub-Sub Simulation Demo", Colors.HEADER)
    
    DURATION = 60
    OUTPUT_DIR = Path("/tmp/simulation_results")
    
    try:
        print_header("Setup: Creating Smart City IoT System")
        system_config = create_smart_city_system()
        print(f"{Colors.GREEN}✓ Created system with:")
        print(f"  • {len(system_config['applications'])} applications")
        print(f"  • {len(system_config['topics'])} topics")
        print(f"  • {len(system_config['brokers'])} brokers{Colors.ENDC}")
        
        # Scenario 1: Baseline
        print_header("Phase 1: Baseline Simulation")
        sim1 = SimpleDDSSimulator()
        sim1.load_system(system_config)
        baseline_results = await scenario_baseline(sim1, DURATION)
        print_results(baseline_results)
        save_results(baseline_results, "baseline", OUTPUT_DIR)
        
        # Scenario 2: Single failure
        print_header("Phase 2: Single Component Failure")
        sim2 = SimpleDDSSimulator()
        sim2.load_system(system_config)
        failure_results = await scenario_single_failure(sim2, DURATION, 'AnalyticsEngine', 30)
        print_results(failure_results, baseline=baseline_results)
        save_results(failure_results, "single_failure", OUTPUT_DIR)
        
        # Scenario 3: Cascading
        print_header("Phase 3: Cascading Failures")
        sim3 = SimpleDDSSimulator()
        sim3.load_system(system_config)
        cascade_results = await scenario_cascading(sim3, DURATION)
        print_results(cascade_results, baseline=baseline_results)
        save_results(cascade_results, "cascading", OUTPUT_DIR)
        
        # Scenario 4: Recovery
        print_header("Phase 4: Failure and Recovery")
        sim4 = SimpleDDSSimulator()
        sim4.load_system(system_config)
        recovery_results = await scenario_recovery(sim4, DURATION)
        print_results(recovery_results, baseline=baseline_results)
        save_results(recovery_results, "recovery", OUTPUT_DIR)
        
        # Summary
        print_header("Demo Completed Successfully!", Colors.GREEN)
        
        scenarios = [
            ("Baseline", baseline_results),
            ("Single Failure", failure_results),
            ("Cascading", cascade_results),
            ("Recovery", recovery_results)
        ]
        
        print(f"\n{Colors.BOLD}Scenario Comparison:{Colors.ENDC}\n")
        print(f"{'Scenario':<20} {'Throughput':<15} {'Avg Latency':<15} {'Delivery Rate':<15}")
        print("-" * 65)
        for name, results in scenarios:
            throughput = results['throughput_msg_s']
            latency = results['global_stats']['avg_latency_ms']
            delivery = results['global_stats']['delivery_rate'] * 100
            print(f"{name:<20} {throughput:>10.1f} msg/s {latency:>10.2f} ms   {delivery:>10.2f}%")
        
        print(f"\n{Colors.CYAN}Results saved to: {OUTPUT_DIR}/{Colors.ENDC}")
        
        print(f"\n{Colors.GREEN}{'='*80}")
        print(f"{'✓ All scenarios completed successfully!'.center(80)}")
        print(f"{'='*80}{Colors.ENDC}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {str(e)}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
