"""
Docker DDS Simulation Orchestrator

Creates and manages a containerized pub-sub system simulation using FastDDS.
Converts graph JSON to Docker Compose configuration with FastDDS applications.

Architecture:
- Each Application → FastDDS container (publisher/subscriber/prosumer)
- Each Broker → Discovery server container
- Each Node → Docker network with resource constraints
- Topics → DDS topics with configured QoS policies

Capabilities:
- Build simulation from graph JSON
- Generate FastDDS application code
- Create Docker Compose configuration
- Configure QoS policies
- Set resource limits
- Run timed simulations
- Collect metrics
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import subprocess
import time
import shutil


@dataclass
class SimulationConfig:
    """Configuration for Docker DDS simulation"""
    duration_seconds: int = 60
    fastdds_version: str = "v3.4.0"
    base_image: str = "ubuntu:22.04"
    network_mode: str = "bridge"
    enable_discovery: bool = True
    discovery_port: int = 11811
    enable_monitoring: bool = True
    output_dir: str = "simulation_output"
    cleanup_on_exit: bool = False
    max_containers: int = 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'duration_seconds': self.duration_seconds,
            'fastdds_version': self.fastdds_version,
            'base_image': self.base_image,
            'network_mode': self.network_mode,
            'enable_discovery': bool(self.enable_discovery),
            'discovery_port': self.discovery_port,
            'enable_monitoring': bool(self.enable_monitoring),
            'output_dir': self.output_dir,
            'cleanup_on_exit': bool(self.cleanup_on_exit),
            'max_containers': self.max_containers
        }


class DockerDDSSimulator:
    """
    Orchestrates Docker-based FastDDS simulations from graph JSON
    
    Features:
    - Parse graph JSON to extract topology
    - Generate FastDDS C++ code for each application
    - Create Dockerfiles for each component
    - Generate Docker Compose configuration
    - Configure resource limits based on nodes
    - Set up DDS discovery servers
    - Run timed simulations
    - Collect logs and metrics
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize simulator"""
        self.config = config or SimulationConfig()
        self.logger = logging.getLogger(__name__)
        self.simulation_dir = Path(self.config.output_dir)
        
    def build_from_json(self,
                       json_path: str,
                       simulation_name: Optional[str] = None) -> str:
        """
        Build complete simulation from graph JSON
        
        Args:
            json_path: Path to graph JSON file
            simulation_name: Optional name for simulation
        
        Returns:
            Path to simulation directory
        """
        self.logger.info(f"Building simulation from {json_path}...")
        
        # Load graph
        with open(json_path) as f:
            graph_data = json.load(f)
        
        # Create simulation directory
        sim_name = simulation_name or f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sim_dir = self.simulation_dir / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating simulation in {sim_dir}")
        
        # Parse graph
        topology = self._parse_graph(graph_data)
        
        # Generate components
        self._generate_base_dockerfile(sim_dir)
        self._generate_applications(sim_dir, topology)
        self._generate_docker_compose(sim_dir, topology)
        self._generate_run_script(sim_dir)
        self._generate_monitor_script(sim_dir, topology)
        
        self.logger.info(f"✓ Simulation ready at {sim_dir}")
        
        return str(sim_dir)
    
    def run_simulation(self,
                      simulation_dir: str,
                      duration: Optional[int] = None) -> Dict[str, Any]:
        """
        Run Docker-based simulation
        
        Args:
            simulation_dir: Path to simulation directory
            duration: Override duration in seconds
        
        Returns:
            Dictionary with simulation results
        """
        sim_dir = Path(simulation_dir)
        duration = duration or self.config.duration_seconds
        
        self.logger.info(f"Starting simulation for {duration} seconds...")
        
        # Build containers
        self._docker_compose(sim_dir, "build", quiet=False)
        
        # Start simulation
        start_time = time.time()
        self._docker_compose(sim_dir, "up -d")
        
        self.logger.info("Simulation running...")
        
        # Wait for duration
        try:
            for i in range(duration):
                time.sleep(1)
                if (i + 1) % 10 == 0:
                    self.logger.info(f"  {i + 1}/{duration} seconds elapsed...")
        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")
        
        # Stop simulation
        self._docker_compose(sim_dir, "down")
        
        elapsed = time.time() - start_time
        
        # Collect results
        results = {
            'duration_seconds': elapsed,
            'simulation_dir': str(sim_dir),
            'containers': self._get_container_stats(sim_dir),
            'logs_collected': self._collect_logs(sim_dir)
        }
        
        self.logger.info(f"✓ Simulation completed in {elapsed:.1f}s")
        
        return results
    
    def _parse_graph(self, graph_data: Dict) -> Dict[str, Any]:
        """Parse graph JSON into simulation topology"""
        
        topology = {
            'applications': [],
            'topics': [],
            'brokers': [],
            'nodes': [],
            'publishes': [],
            'subscribes': [],
            'routes': [],
            'runs_on': []
        }
        
        # Extract components
        if 'applications' in graph_data:
            topology['applications'] = graph_data['applications']
        
        if 'topics' in graph_data:
            topology['topics'] = graph_data['topics']
        
        if 'brokers' in graph_data:
            topology['brokers'] = graph_data['brokers']
        
        if 'nodes' in graph_data:
            topology['nodes'] = graph_data['nodes']
        
        # Extract relationships
        if 'relationships' in graph_data:
            rels = graph_data['relationships']
            topology['publishes'] = rels.get('publishes_to', [])
            topology['subscribes'] = rels.get('subscribes_to', [])
            topology['routes'] = rels.get('routes', [])
            topology['runs_on'] = rels.get('runs_on', [])
        
        self.logger.info(f"Parsed topology: {len(topology['applications'])} apps, "
                        f"{len(topology['topics'])} topics, "
                        f"{len(topology['brokers'])} brokers, "
                        f"{len(topology['nodes'])} nodes")
        
        return topology
    
    def _generate_base_dockerfile(self, sim_dir: Path):
        """Generate base Dockerfile for FastDDS"""
        
        dockerfile = f"""FROM {self.config.base_image}

# Install FastDDS
RUN apt-get update && apt-get install -y \\
    software-properties-common \\
    wget \\
    curl \\
    gnupg2 \\
    lsb-release \\
    build-essential \\
    cmake \\
    git \\
    python3 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Install FastDDS from source
WORKDIR /opt
RUN git clone https://github.com/eProsima/Fast-DDS.git -b {self.config.fastdds_version} && \\
    cd Fast-DDS && \\
    mkdir build && cd build && \\
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \\
    cmake --build . --target install && \\
    ldconfig

# Install monitoring tools
RUN pip3 install psutil pandas

WORKDIR /app
"""
        
        (sim_dir / "Dockerfile.base").write_text(dockerfile)
        self.logger.info("  Generated base Dockerfile")
    
    def _generate_applications(self, sim_dir: Path, topology: Dict):
        """Generate FastDDS applications"""
        
        apps_dir = sim_dir / "apps"
        apps_dir.mkdir(exist_ok=True)
        
        # Generate code for each application
        for app in topology['applications']:
            app_id = app['id']
            app_dir = apps_dir / app_id
            app_dir.mkdir(exist_ok=True)
            
            # Determine pub/sub behavior
            publishes = [p for p in topology['publishes'] if p['from'] == app_id]
            subscribes = [s for s in topology['subscribes'] if s['from'] == app_id]
            
            # Generate application code
            self._generate_app_cpp(app_dir, app, publishes, subscribes, topology)
            self._generate_app_cmake(app_dir, app_id)
            self._generate_app_dockerfile(app_dir, app_id)
            
        self.logger.info(f"  Generated {len(topology['applications'])} application containers")
    
    def _generate_app_cpp(self, app_dir: Path, app: Dict,
                         publishes: List, subscribes: List, topology: Dict):
        """Generate C++ FastDDS application code"""
        
        app_id = app['id']
        app_name = app.get('name', app_id)
        
        # Get topics with QoS
        pub_topics = self._get_topics_for_app(publishes, topology, 'to')
        sub_topics = self._get_topics_for_app(subscribes, topology, 'to')
        
        cpp_code = f"""/**
 * {app_name} - FastDDS Application
 * Generated from graph model
 */

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>

#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>

using namespace eprosima::fastdds::dds;

// Simple message type
struct Message {{
    std::string topic_name;
    std::string sender;
    uint64_t sequence;
    uint64_t timestamp;
    std::vector<uint8_t> payload;
}};

std::atomic<bool> running(true);

void signal_handler(int signal) {{
    running = false;
}}

class MessageListener : public DataReaderListener {{
public:
    MessageListener(const std::string& topic) : topic_name_(topic), received_(0) {{}}
    
    void on_data_available(DataReader* reader) override {{
        Message msg;
        SampleInfo info;
        
        if (reader->take_next_sample(&msg, &info) == ReturnCode_t::RETCODE_OK) {{
            if (info.valid_data) {{
                received_++;
                std::cout << "[" << topic_name_ << "] Received message #" 
                         << msg.sequence << " from " << msg.sender << std::endl;
            }}
        }}
    }}
    
    uint64_t get_received_count() const {{ return received_; }}
    
private:
    std::string topic_name_;
    std::atomic<uint64_t> received_;
}};

int main(int argc, char** argv) {{
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "Starting {app_name}..." << std::endl;
    
    // Create participant
    DomainParticipantQos pqos;
    pqos.name("{app_name}");
    
    DomainParticipant* participant = 
        DomainParticipantFactory::get_instance()->create_participant(0, pqos);
    
    if (participant == nullptr) {{
        std::cerr << "Failed to create participant" << std::endl;
        return 1;
    }}
    
    // Register type
    TypeSupport type(new MessagePubSubType());
    type.register_type(participant);
    
"""
        
        # Add publishers
        if pub_topics:
            cpp_code += "    // Create publishers\n"
            for i, topic_info in enumerate(pub_topics):
                topic_name = topic_info['name']
                qos = topic_info.get('qos', {})
                
                cpp_code += f"""
    Topic* pub_topic_{i} = participant->create_topic(
        "{topic_name}", type.get_type_name(), TOPIC_QOS_DEFAULT);
    Publisher* publisher_{i} = participant->create_publisher(PUBLISHER_QOS_DEFAULT);
    DataWriter* writer_{i} = publisher_{i}->create_datawriter(
        pub_topic_{i}, DATAWRITER_QOS_DEFAULT);
"""
        
        # Add subscribers
        if sub_topics:
            cpp_code += "\n    // Create subscribers\n"
            for i, topic_info in enumerate(sub_topics):
                topic_name = topic_info['name']
                
                cpp_code += f"""
    Topic* sub_topic_{i} = participant->create_topic(
        "{topic_name}", type.get_type_name(), TOPIC_QOS_DEFAULT);
    Subscriber* subscriber_{i} = participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
    MessageListener listener_{i}("{topic_name}");
    DataReader* reader_{i} = subscriber_{i}->create_datareader(
        sub_topic_{i}, DATAREADER_QOS_DEFAULT, &listener_{i});
"""
        
        # Main loop
        cpp_code += f"""
    
    std::cout << "{app_name} running..." << std::endl;
    
    uint64_t sequence = 0;
    while (running) {{
"""
        
        # Publish messages
        if pub_topics:
            cpp_code += """
        // Publish messages
        Message msg;
        msg.sender = "{app_name}";
        msg.sequence = ++sequence;
        msg.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        msg.payload.resize(1024); // 1KB payload
        
"""
            for i, topic_info in enumerate(pub_topics):
                period_ms = topic_info.get('publish_period_ms', 1000)
                cpp_code += f"""
        msg.topic_name = "{topic_info['name']}";
        writer_{i}->write(&msg);
"""
            
            cpp_code += f"""
        std::this_thread::sleep_for(std::chrono::milliseconds({pub_topics[0].get('publish_period_ms', 1000)}));
"""
        else:
            cpp_code += """
        std::this_thread::sleep_for(std::chrono::seconds(1));
"""
        
        cpp_code += """
    }
    
    std::cout << "{app_name} shutting down..." << std::endl;
    
    // Cleanup
    DomainParticipantFactory::get_instance()->delete_participant(participant);
    
    return 0;
}
"""
        
        (app_dir / f"{app['id']}.cpp").write_text(cpp_code)
    
    def _generate_app_cmake(self, app_dir: Path, app_id: str):
        """Generate CMakeLists.txt for application"""
        
        cmake = f"""cmake_minimum_required(VERSION 3.16)
project({app_id})

set(CMAKE_CXX_STANDARD 17)

find_package(fastrtps REQUIRED)
find_package(fastcdr REQUIRED)

add_executable({app_id} {app_id}.cpp)
target_link_libraries({app_id} fastrtps fastcdr)
"""
        
        (app_dir / "CMakeLists.txt").write_text(cmake)
    
    def _generate_app_dockerfile(self, app_dir: Path, app_id: str):
        """Generate Dockerfile for specific application"""
        
        dockerfile = f"""FROM simulation_base:latest

COPY . /app
WORKDIR /app

RUN mkdir build && cd build && \\
    cmake .. && \\
    make

CMD ["./build/{app_id}"]
"""
        
        (app_dir / "Dockerfile").write_text(dockerfile)
    
    def _generate_docker_compose(self, sim_dir: Path, topology: Dict):
        """Generate Docker Compose configuration"""
        
        compose = {
            'version': '3.8',
            'services': {},
            'networks': {
                'dds_network': {
                    'driver': self.config.network_mode
                }
            }
        }
        
        # Add discovery server if enabled
        if self.config.enable_discovery:
            compose['services']['discovery'] = {
                'build': {
                    'context': '.',
                    'dockerfile': 'Dockerfile.base'
                },
                'container_name': 'dds_discovery',
                'command': 'fast-discovery-server -i 0 -l 0.0.0.0 -p 11811',
                'networks': ['dds_network'],
                'ports': [f'{self.config.discovery_port}:11811/udp']
            }
        
        # Add application containers
        for app in topology['applications']:
            app_id = app['id']
            
            # Find node assignment for resource limits
            node_assignment = next(
                (r for r in topology['runs_on'] if r['from'] == app_id),
                None
            )
            
            node = None
            if node_assignment:
                node_id = node_assignment['to']
                node = next((n for n in topology['nodes'] if n['id'] == node_id), None)
            
            service_config = {
                'build': {
                    'context': f'./apps/{app_id}',
                    'dockerfile': 'Dockerfile'
                },
                'container_name': f'app_{app_id}',
                'depends_on': ['discovery'] if self.config.enable_discovery else [],
                'networks': ['dds_network'],
                'environment': [
                    f'FASTDDS_DISCOVERY_SERVER=discovery:{self.config.discovery_port}',
                    f'APP_ID={app_id}',
                    f'APP_NAME={app.get("name", app_id)}'
                ]
            }
            
            # Add resource limits if node info available
            if node:
                cpu_limit = node.get('cpu_capacity', 4) / 100.0  # As fraction
                mem_limit = f"{int(node.get('memory_gb', 4))}g"
                
                service_config['deploy'] = {
                    'resources': {
                        'limits': {
                            'cpus': str(cpu_limit),
                            'memory': mem_limit
                        }
                    }
                }
            
            compose['services'][app_id] = service_config
        
        # Write compose file
        with open(sim_dir / "docker-compose.yml", 'w') as f:
            yaml.dump(compose, f, default_flow_style=False)
        
        self.logger.info(f"  Generated Docker Compose with {len(compose['services'])} services")
    
    def _generate_run_script(self, sim_dir: Path):
        """Generate simulation run script"""
        
        script = f"""#!/bin/bash

echo "Building simulation..."
docker-compose build

echo "Starting simulation for {self.config.duration_seconds} seconds..."
docker-compose up -d

echo "Simulation running..."
sleep {self.config.duration_seconds}

echo "Stopping simulation..."
docker-compose down

echo "Collecting logs..."
mkdir -p logs
docker-compose logs > logs/simulation.log

echo "Simulation complete!"
"""
        
        script_path = sim_dir / "run_simulation.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)
        
        self.logger.info("  Generated run script")
    
    def _generate_monitor_script(self, sim_dir: Path, topology: Dict):
        """Generate monitoring script"""
        
        script = """#!/usr/bin/env python3
import docker
import time
import json
from datetime import datetime

client = docker.from_env()

print("Monitoring containers...")
print(f"Time: {datetime.now()}")

while True:
    try:
        containers = client.containers.list()
        
        stats = {}
        for container in containers:
            if container.name.startswith('app_'):
                stat = container.stats(stream=False)
                stats[container.name] = {
                    'cpu_percent': stat['cpu_stats']['cpu_usage']['total_usage'],
                    'memory_mb': stat['memory_stats']['usage'] / 1024 / 1024,
                    'network_rx': stat['networks']['eth0']['rx_bytes'],
                    'network_tx': stat['networks']['eth0']['tx_bytes']
                }
        
        print(f"\\n{datetime.now()}")
        for name, stat in stats.items():
            print(f"  {name}: MEM={stat['memory_mb']:.1f}MB RX={stat['network_rx']} TX={stat['network_tx']}")
        
        time.sleep(5)
        
    except KeyboardInterrupt:
        break

print("Monitoring stopped")
"""
        
        script_path = sim_dir / "monitor.py"
        script_path.write_text(script)
        script_path.chmod(0o755)
    
    def _get_topics_for_app(self, relationships: List, topology: Dict, key: str) -> List[Dict]:
        """Get topic information for application"""
        
        topic_ids = [r[key] for r in relationships]
        topics = []
        
        for topic_id in topic_ids:
            topic = next((t for t in topology['topics'] if t['id'] == topic_id), None)
            if topic:
                # Add publishing info
                pub_rel = next((r for r in relationships if r[key] == topic_id), {})
                topic_copy = topic.copy()
                topic_copy['publish_period_ms'] = pub_rel.get('period_ms', 1000)
                topic_copy['message_size'] = pub_rel.get('msg_size', 1024)
                topics.append(topic_copy)
        
        return topics
    
    def _docker_compose(self, sim_dir: Path, command: str, quiet: bool = True):
        """Execute docker-compose command"""
        
        cmd = f"docker-compose {command}"
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=sim_dir,
                capture_output=quiet,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"Docker compose failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error running docker-compose: {e}")
    
    def _get_container_stats(self, sim_dir: Path) -> Dict:
        """Get statistics about containers"""
        
        try:
            result = subprocess.run(
                "docker-compose ps",
                shell=True,
                cwd=sim_dir,
                capture_output=True,
                text=True
            )
            
            lines = result.stdout.strip().split('\n')
            return {'container_count': len(lines) - 1 if len(lines) > 1 else 0}
            
        except:
            return {'container_count': 0}
    
    def _collect_logs(self, sim_dir: Path) -> bool:
        """Collect container logs"""
        
        logs_dir = sim_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        try:
            subprocess.run(
                f"docker-compose logs > {logs_dir}/simulation.log",
                shell=True,
                cwd=sim_dir
            )
            return True
        except:
            return False
