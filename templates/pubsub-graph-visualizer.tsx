import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import * as d3 from 'd3';
import { Search, Filter, Layers, ZoomIn, ZoomOut, Maximize2, Settings, AlertCircle, Activity, GitBranch, Circle, Box, RefreshCw } from 'lucide-react';

// Main Application Component
const GraphVisualizer = () => {
  const [data, setData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [layoutMode, setLayoutMode] = useState('force');
  const [filterLevel, setFilterLevel] = useState('all');
  const [selectedNode, setSelectedNode] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [stats, setStats] = useState({
    nodes: 0,
    applications: 0,
    topics: 0,
    brokers: 0,
    criticalComponents: 0
  });

  // Load initial data
  useEffect(() => {
    loadGraphData();
  }, [filterLevel]);

  const loadGraphData = async () => {
    setLoading(true);
    // Simulate API call - replace with actual API endpoint
    setTimeout(() => {
      const mockData = generateMockData();
      setData(mockData);
      updateStats(mockData);
      setLoading(false);
    }, 1000);
  };

  const updateStats = (graphData) => {
    const nodeTypes = graphData.nodes.reduce((acc, node) => {
      acc[node.type] = (acc[node.type] || 0) + 1;
      return acc;
    }, {});
    
    const criticalCount = graphData.nodes.filter(n => n.criticality > 0.7).length;
    
    setStats({
      nodes: nodeTypes.Node || 0,
      applications: nodeTypes.Application || 0,
      topics: nodeTypes.Topic || 0,
      brokers: nodeTypes.Broker || 0,
      criticalComponents: criticalCount
    });
  };

  return (
    <div className="w-full h-screen bg-gray-900 text-gray-100 flex flex-col">
      {/* Header */}
      <Header 
        searchTerm={searchTerm}
        setSearchTerm={setSearchTerm}
        onRefresh={loadGraphData}
      />
      
      <div className="flex-1 flex">
        {/* Sidebar */}
        <Sidebar
          stats={stats}
          filterLevel={filterLevel}
          setFilterLevel={setFilterLevel}
          layoutMode={layoutMode}
          setLayoutMode={setLayoutMode}
          selectedNode={selectedNode}
        />
        
        {/* Main Visualization Area */}
        <div className="flex-1 relative">
          {loading ? (
            <LoadingOverlay />
          ) : (
            <>
              <GraphCanvas
                data={data}
                layoutMode={layoutMode}
                searchTerm={searchTerm}
                onNodeSelect={setSelectedNode}
                filterLevel={filterLevel}
              />
              <ControlPanel layoutMode={layoutMode} setLayoutMode={setLayoutMode} />
            </>
          )}
        </div>
      </div>
    </div>
  );
};

// Header Component
const Header = ({ searchTerm, setSearchTerm, onRefresh }) => (
  <header className="bg-gray-800 border-b border-gray-700 p-4">
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-4">
        <GitBranch className="w-8 h-8 text-blue-400" />
        <h1 className="text-2xl font-bold">Publish-Subscribe System Visualizer</h1>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search nodes..."
            className="pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-blue-400 text-gray-100"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <button
          onClick={onRefresh}
          className="p-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>
    </div>
  </header>
);

// Sidebar Component
const Sidebar = ({ stats, filterLevel, setFilterLevel, layoutMode, setLayoutMode, selectedNode }) => (
  <aside className="w-80 bg-gray-800 border-r border-gray-700 p-4 overflow-y-auto">
    {/* Statistics */}
    <div className="mb-6">
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <Activity className="w-5 h-5 mr-2 text-blue-400" />
        System Statistics
      </h2>
      <div className="space-y-2">
        <StatCard label="Nodes" value={stats.nodes} color="blue" />
        <StatCard label="Applications" value={stats.applications} color="green" />
        <StatCard label="Topics" value={stats.topics} color="purple" />
        <StatCard label="Brokers" value={stats.brokers} color="orange" />
        <StatCard label="Critical Components" value={stats.criticalComponents} color="red" />
      </div>
    </div>

    {/* Filters */}
    <div className="mb-6">
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <Filter className="w-5 h-5 mr-2 text-blue-400" />
        Filters
      </h2>
      <div className="space-y-2">
        <label className="block text-sm text-gray-400 mb-1">Criticality Level</label>
        <select
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-blue-400"
          value={filterLevel}
          onChange={(e) => setFilterLevel(e.target.value)}
        >
          <option value="all">All Components</option>
          <option value="critical">Critical Only (>0.7)</option>
          <option value="high">High & Critical (>0.5)</option>
          <option value="medium">Medium+ (>0.3)</option>
        </select>
      </div>
    </div>

    {/* Layout Options */}
    <div className="mb-6">
      <h2 className="text-lg font-semibold mb-4 flex items-center">
        <Layers className="w-5 h-5 mr-2 text-blue-400" />
        Layout Mode
      </h2>
      <div className="space-y-2">
        <LayoutOption
          mode="force"
          currentMode={layoutMode}
          setMode={setLayoutMode}
          label="Force-Directed"
          description="Physics-based layout"
        />
        <LayoutOption
          mode="hierarchical"
          currentMode={layoutMode}
          setMode={setLayoutMode}
          label="Hierarchical"
          description="Layered tree structure"
        />
        <LayoutOption
          mode="circular"
          currentMode={layoutMode}
          setMode={setLayoutMode}
          label="Circular"
          description="Radial arrangement"
        />
      </div>
    </div>

    {/* Selected Node Details */}
    {selectedNode && (
      <div className="mb-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center">
          <Box className="w-5 h-5 mr-2 text-blue-400" />
          Selected Component
        </h2>
        <NodeDetails node={selectedNode} />
      </div>
    )}
  </aside>
);

// Statistics Card Component
const StatCard = ({ label, value, color }) => {
  const colorClasses = {
    blue: 'bg-blue-500/20 text-blue-400',
    green: 'bg-green-500/20 text-green-400',
    purple: 'bg-purple-500/20 text-purple-400',
    orange: 'bg-orange-500/20 text-orange-400',
    red: 'bg-red-500/20 text-red-400'
  };

  return (
    <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
      <div className="flex justify-between items-center">
        <span className="text-sm">{label}</span>
        <span className="text-2xl font-bold">{value}</span>
      </div>
    </div>
  );
};

// Layout Option Component
const LayoutOption = ({ mode, currentMode, setMode, label, description }) => (
  <button
    onClick={() => setMode(mode)}
    className={`w-full p-3 rounded-lg border transition-all ${
      currentMode === mode
        ? 'bg-blue-500/20 border-blue-400 text-blue-400'
        : 'bg-gray-700 border-gray-600 hover:border-gray-500'
    }`}
  >
    <div className="text-left">
      <div className="font-semibold">{label}</div>
      <div className="text-xs text-gray-400">{description}</div>
    </div>
  </button>
);

// Node Details Component
const NodeDetails = ({ node }) => (
  <div className="bg-gray-700 rounded-lg p-4 space-y-3">
    <div>
      <div className="text-xs text-gray-400">ID</div>
      <div className="font-mono text-sm">{node.id}</div>
    </div>
    <div>
      <div className="text-xs text-gray-400">Type</div>
      <div className="text-sm">{node.type}</div>
    </div>
    <div>
      <div className="text-xs text-gray-400">Criticality Score</div>
      <div className="flex items-center">
        <div className="flex-1 bg-gray-600 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${
              node.criticality > 0.7 ? 'bg-red-500' :
              node.criticality > 0.5 ? 'bg-orange-500' :
              node.criticality > 0.3 ? 'bg-yellow-500' : 'bg-green-500'
            }`}
            style={{ width: `${node.criticality * 100}%` }}
          />
        </div>
        <span className="ml-2 text-sm">{(node.criticality * 100).toFixed(0)}%</span>
      </div>
    </div>
    {node.qos && (
      <div>
        <div className="text-xs text-gray-400">QoS Score</div>
        <div className="text-sm">{node.qos.toFixed(3)}</div>
      </div>
    )}
  </div>
);

// Control Panel Component
const ControlPanel = ({ layoutMode, setLayoutMode }) => {
  const svgRef = useRef(null);
  
  const handleZoomIn = () => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(300).call(
        d3.zoom().scaleBy, 1.3
      );
    }
  };

  const handleZoomOut = () => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(300).call(
        d3.zoom().scaleBy, 0.7
      );
    }
  };

  const handleZoomReset = () => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(300).call(
        d3.zoom().transform, d3.zoomIdentity
      );
    }
  };

  return (
    <div className="absolute top-4 right-4 bg-gray-800 rounded-lg shadow-lg p-2 flex flex-col space-y-2">
      <button
        onClick={handleZoomIn}
        className="p-2 hover:bg-gray-700 rounded transition-colors"
        title="Zoom In"
      >
        <ZoomIn className="w-5 h-5" />
      </button>
      <button
        onClick={handleZoomOut}
        className="p-2 hover:bg-gray-700 rounded transition-colors"
        title="Zoom Out"
      >
        <ZoomOut className="w-5 h-5" />
      </button>
      <button
        onClick={handleZoomReset}
        className="p-2 hover:bg-gray-700 rounded transition-colors"
        title="Reset View"
      >
        <Maximize2 className="w-5 h-5" />
      </button>
    </div>
  );
};

// Main Graph Canvas Component with D3.js
const GraphCanvas = ({ data, layoutMode, searchTerm, onNodeSelect, filterLevel }) => {
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const simulationRef = useRef(null);

  // Filter data based on criticality level
  const filteredData = useMemo(() => {
    let filteredNodes = [...data.nodes];
    let filteredLinks = [...data.links];

    // Apply criticality filter
    if (filterLevel !== 'all') {
      const thresholds = {
        critical: 0.7,
        high: 0.5,
        medium: 0.3
      };
      const threshold = thresholds[filterLevel];
      filteredNodes = filteredNodes.filter(n => n.criticality >= threshold);
      const nodeIds = new Set(filteredNodes.map(n => n.id));
      filteredLinks = filteredLinks.filter(l => 
        nodeIds.has(l.source.id || l.source) && 
        nodeIds.has(l.target.id || l.target)
      );
    }

    // Apply search filter
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      filteredNodes = filteredNodes.filter(n => 
        n.id.toLowerCase().includes(searchLower) ||
        n.type.toLowerCase().includes(searchLower)
      );
      const nodeIds = new Set(filteredNodes.map(n => n.id));
      filteredLinks = filteredLinks.filter(l => 
        nodeIds.has(l.source.id || l.source) && 
        nodeIds.has(l.target.id || l.target)
      );
    }

    return { nodes: filteredNodes, links: filteredLinks };
  }, [data, filterLevel, searchTerm]);

  // Main D3 visualization effect
  useEffect(() => {
    if (!containerRef.current || filteredData.nodes.length === 0) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Clear previous visualization
    d3.select(svgRef.current).selectAll("*").remove();

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    // Create zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
      });

    svg.call(zoom);

    // Create main group
    const g = svg.append("g");

    // Define arrow markers for directed edges
    svg.append("defs").selectAll("marker")
      .data(["end"])
      .enter().append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 15)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#666");

    // Create layout based on mode
    let simulation;
    const nodes = filteredData.nodes.map(d => ({ ...d }));
    const links = filteredData.links.map(d => ({ ...d }));

    if (layoutMode === 'force') {
      // Force-directed layout
      simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(30));
    } else if (layoutMode === 'hierarchical') {
      // Hierarchical layout
      const stratify = d3.stratify()
        .id(d => d.id)
        .parentId(d => {
          // Determine parent based on node type hierarchy
          if (d.type === 'Application') return d.node || null;
          if (d.type === 'Topic') return d.broker || null;
          if (d.type === 'Broker') return d.node || null;
          return null;
        });

      try {
        const root = stratify(nodes);
        const treeLayout = d3.tree().size([width - 100, height - 100]);
        const treeData = treeLayout(root);
        
        treeData.descendants().forEach(d => {
          const node = nodes.find(n => n.id === d.data.id);
          if (node) {
            node.x = d.x + 50;
            node.y = d.y + 50;
          }
        });
      } catch (e) {
        // Fallback to force layout if hierarchy fails
        console.log("Hierarchical layout failed, using force layout");
        simulation = d3.forceSimulation(nodes)
          .force("link", d3.forceLink(links).id(d => d.id))
          .force("charge", d3.forceManyBody().strength(-300))
          .force("center", d3.forceCenter(width / 2, height / 2));
      }
    } else if (layoutMode === 'circular') {
      // Circular layout
      const radius = Math.min(width, height) / 3;
      const angleStep = (2 * Math.PI) / nodes.length;
      
      nodes.forEach((node, i) => {
        const angle = i * angleStep;
        node.x = width / 2 + radius * Math.cos(angle);
        node.y = height / 2 + radius * Math.sin(angle);
      });
    }

    // Create links
    const link = g.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke", d => {
        // Color based on edge type
        const edgeColors = {
          'PUBLISHES_TO': '#10b981',
          'SUBSCRIBES_TO': '#3b82f6',
          'ROUTES': '#f59e0b',
          'RUNS_ON': '#8b5cf6',
          'DEPENDS_ON': '#ef4444'
        };
        return edgeColors[d.type] || '#666';
      })
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", d => Math.sqrt(d.weight || 1))
      .attr("marker-end", "url(#arrow)");

    // Create node groups
    const node = g.append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(nodes)
      .enter().append("g")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Add circles for nodes
    node.append("circle")
      .attr("r", d => {
        // Size based on criticality
        return 5 + d.criticality * 20;
      })
      .attr("fill", d => {
        // Color based on node type
        const nodeColors = {
          'Node': '#3b82f6',
          'Application': '#10b981',
          'Topic': '#f59e0b',
          'Broker': '#8b5cf6'
        };
        return nodeColors[d.type] || '#666';
      })
      .attr("stroke", d => {
        // Highlight critical nodes
        return d.criticality > 0.7 ? '#ef4444' : '#fff';
      })
      .attr("stroke-width", d => d.criticality > 0.7 ? 3 : 1)
      .attr("opacity", 0.9);

    // Add labels
    node.append("text")
      .text(d => d.id)
      .attr("x", 0)
      .attr("y", -10)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("fill", "#fff")
      .style("pointer-events", "none");

    // Add hover effects
    node
      .on("mouseenter", function(event, d) {
        // Highlight connected edges
        link
          .attr("stroke-opacity", l => 
            l.source.id === d.id || l.target.id === d.id ? 1 : 0.1
          )
          .attr("stroke-width", l => 
            l.source.id === d.id || l.target.id === d.id ? 3 : 1
          );
        
        // Highlight connected nodes
        node.attr("opacity", n => {
          const connected = links.some(l => 
            (l.source.id === d.id && l.target.id === n.id) ||
            (l.target.id === d.id && l.source.id === n.id) ||
            n.id === d.id
          );
          return connected ? 1 : 0.3;
        });
      })
      .on("mouseleave", function() {
        link
          .attr("stroke-opacity", 0.6)
          .attr("stroke-width", d => Math.sqrt(d.weight || 1));
        node.attr("opacity", 0.9);
      })
      .on("click", (event, d) => {
        onNodeSelect(d);
      });

    // Tooltip
    const tooltip = d3.select("body").append("div")
      .attr("class", "tooltip")
      .style("position", "absolute")
      .style("padding", "10px")
      .style("background", "rgba(0, 0, 0, 0.8)")
      .style("color", "#fff")
      .style("border-radius", "5px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("opacity", 0);

    node
      .on("mouseover", (event, d) => {
        tooltip
          .style("opacity", 1)
          .html(`
            <strong>${d.id}</strong><br/>
            Type: ${d.type}<br/>
            Criticality: ${(d.criticality * 100).toFixed(0)}%<br/>
            ${d.qos ? `QoS: ${d.qos.toFixed(3)}` : ''}
          `)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 10) + "px");
      })
      .on("mouseout", () => {
        tooltip.style("opacity", 0);
      });

    // Update positions on simulation tick
    if (simulation) {
      simulation.on("tick", () => {
        link
          .attr("x1", d => d.source.x)
          .attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x)
          .attr("y2", d => d.target.y);

        node.attr("transform", d => `translate(${d.x},${d.y})`);
      });
      
      simulationRef.current = simulation;
    } else {
      // For non-force layouts, set positions directly
      link
        .attr("x1", d => {
          const source = nodes.find(n => n.id === (d.source.id || d.source));
          return source ? source.x : 0;
        })
        .attr("y1", d => {
          const source = nodes.find(n => n.id === (d.source.id || d.source));
          return source ? source.y : 0;
        })
        .attr("x2", d => {
          const target = nodes.find(n => n.id === (d.target.id || d.target));
          return target ? target.x : 0;
        })
        .attr("y2", d => {
          const target = nodes.find(n => n.id === (d.target.id || d.target));
          return target ? target.y : 0;
        });

      node.attr("transform", d => `translate(${d.x},${d.y})`);
    }

    // Drag functions
    function dragstarted(event, d) {
      if (simulation) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
      if (simulation) {
        d.x = event.x;
        d.y = event.y;
      }
    }

    function dragended(event, d) {
      if (simulation) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }
    }

    // Cleanup
    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
      tooltip.remove();
    };
  }, [filteredData, layoutMode, onNodeSelect]);

  return (
    <div ref={containerRef} className="w-full h-full bg-gray-900">
      <svg ref={svgRef}></svg>
    </div>
  );
};

// Loading Overlay Component
const LoadingOverlay = () => (
  <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50">
    <div className="text-center">
      <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-400 mx-auto mb-4"></div>
      <p className="text-gray-300">Loading graph data...</p>
    </div>
  </div>
);

// Mock data generator
const generateMockData = () => {
  const nodes = [];
  const links = [];
  
  // Create nodes
  const nodeTypes = ['Node', 'Application', 'Topic', 'Broker'];
  const nodeCounts = { Node: 5, Application: 20, Topic: 30, Broker: 3 };
  
  for (const [type, count] of Object.entries(nodeCounts)) {
    for (let i = 0; i < count; i++) {
      nodes.push({
        id: `${type.toLowerCase()}${i + 1}`,
        type: type,
        criticality: Math.random(),
        qos: Math.random(),
        node: type === 'Application' ? `node${Math.floor(Math.random() * 5) + 1}` : null,
        broker: type === 'Topic' ? `broker${Math.floor(Math.random() * 3) + 1}` : null
      });
    }
  }
  
  // Create links
  const linkTypes = ['PUBLISHES_TO', 'SUBSCRIBES_TO', 'ROUTES', 'RUNS_ON', 'DEPENDS_ON'];
  
  // Applications publish to topics
  for (let i = 0; i < 40; i++) {
    const source = `application${Math.floor(Math.random() * 20) + 1}`;
    const target = `topic${Math.floor(Math.random() * 30) + 1}`;
    links.push({
      source,
      target,
      type: Math.random() > 0.5 ? 'PUBLISHES_TO' : 'SUBSCRIBES_TO',
      weight: Math.random() * 5
    });
  }
  
  // Brokers route topics
  for (let i = 0; i < 30; i++) {
    const source = `broker${Math.floor(Math.random() * 3) + 1}`;
    const target = `topic${i + 1}`;
    links.push({
      source,
      target,
      type: 'ROUTES',
      weight: Math.random() * 3
    });
  }
  
  // Applications run on nodes
  for (let i = 0; i < 20; i++) {
    const source = `application${i + 1}`;
    const target = `node${Math.floor(Math.random() * 5) + 1}`;
    links.push({
      source,
      target,
      type: 'RUNS_ON',
      weight: 1
    });
  }
  
  return { nodes, links };
};

export default GraphVisualizer;