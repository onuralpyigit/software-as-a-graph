import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import * as d3 from 'd3';
import { Play, Pause, BarChart3, Network, TrendingUp, AlertTriangle, Shield, Cpu, Database, Globe, ChevronRight, ChevronDown, Eye, EyeOff, Zap, Layers3, GitMerge } from 'lucide-react';

// Advanced Graph Visualizer with Progressive Rendering and Clustering
const AdvancedGraphVisualizer = () => {
  const [data, setData] = useState({ nodes: [], links: [] });
  const [renderMode, setRenderMode] = useState('progressive');
  const [clustering, setClustering] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);
  const [selectedPath, setSelectedPath] = useState(null);
  const [failureNodes, setFailureNodes] = useState(new Set());
  const [performanceMode, setPerformanceMode] = useState('balanced');
  const [levelOfDetail, setLevelOfDetail] = useState('standard');
  
  useEffect(() => {
    // Load large dataset
    loadLargeDataset();
  }, []);
  
  const loadLargeDataset = () => {
    const largeData = generateLargeDataset();
    setData(largeData);
  };
  
  return (
    <div className="w-full h-screen bg-gray-900 flex flex-col">
      <AdvancedHeader 
        renderMode={renderMode}
        setRenderMode={setRenderMode}
        performanceMode={performanceMode}
        setPerformanceMode={setPerformanceMode}
      />
      
      <div className="flex-1 flex">
        <AdvancedSidebar
          clustering={clustering}
          setClustering={setClustering}
          showMetrics={showMetrics}
          setShowMetrics={setShowMetrics}
          failureNodes={failureNodes}
          setFailureNodes={setFailureNodes}
          levelOfDetail={levelOfDetail}
          setLevelOfDetail={setLevelOfDetail}
        />
        
        <div className="flex-1 relative">
          <ProgressiveGraphCanvas
            data={data}
            renderMode={renderMode}
            clustering={clustering}
            showMetrics={showMetrics}
            performanceMode={performanceMode}
            levelOfDetail={levelOfDetail}
            onPathSelect={setSelectedPath}
            failureNodes={failureNodes}
          />
          
          {selectedPath && (
            <PathAnalysisOverlay path={selectedPath} onClose={() => setSelectedPath(null)} />
          )}
          
          <PerformanceIndicator data={data} performanceMode={performanceMode} />
        </div>
        
        {showMetrics && <MetricsPanel data={data} />}
      </div>
    </div>
  );
};

// Advanced Header with Performance Controls
const AdvancedHeader = ({ renderMode, setRenderMode, performanceMode, setPerformanceMode }) => (
  <header className="bg-gray-800 border-b border-gray-700 p-4">
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-4">
        <Network className="w-8 h-8 text-blue-400" />
        <h1 className="text-2xl font-bold">Advanced Graph Visualizer</h1>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <Layers3 className="w-5 h-5 text-gray-400" />
          <select
            className="px-3 py-1 bg-gray-700 border border-gray-600 rounded text-sm"
            value={renderMode}
            onChange={(e) => setRenderMode(e.target.value)}
          >
            <option value="progressive">Progressive Rendering</option>
            <option value="virtualized">Virtualized</option>
            <option value="webgl">WebGL (GPU)</option>
            <option value="standard">Standard</option>
          </select>
        </div>
        
        <div className="flex items-center space-x-2">
          <Zap className="w-5 h-5 text-gray-400" />
          <select
            className="px-3 py-1 bg-gray-700 border border-gray-600 rounded text-sm"
            value={performanceMode}
            onChange={(e) => setPerformanceMode(e.target.value)}
          >
            <option value="quality">Quality</option>
            <option value="balanced">Balanced</option>
            <option value="performance">Performance</option>
          </select>
        </div>
      </div>
    </div>
  </header>
);

// Advanced Sidebar with Failure Simulation
const AdvancedSidebar = ({ 
  clustering, setClustering, 
  showMetrics, setShowMetrics,
  failureNodes, setFailureNodes,
  levelOfDetail, setLevelOfDetail
}) => {
  const [expandedSections, setExpandedSections] = useState({
    visualization: true,
    analysis: true,
    simulation: false
  });
  
  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };
  
  return (
    <aside className="w-80 bg-gray-800 border-r border-gray-700 overflow-y-auto">
      {/* Visualization Controls */}
      <div className="border-b border-gray-700">
        <button
          onClick={() => toggleSection('visualization')}
          className="w-full p-4 flex items-center justify-between hover:bg-gray-700/50 transition-colors"
        >
          <div className="flex items-center space-x-2">
            <Eye className="w-5 h-5 text-blue-400" />
            <span className="font-semibold">Visualization</span>
          </div>
          {expandedSections.visualization ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </button>
        
        {expandedSections.visualization && (
          <div className="p-4 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm">Enable Clustering</span>
              <button
                onClick={() => setClustering(!clustering)}
                className={`w-12 h-6 rounded-full transition-colors ${
                  clustering ? 'bg-blue-500' : 'bg-gray-600'
                }`}
              >
                <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
                  clustering ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm">Show Metrics</span>
              <button
                onClick={() => setShowMetrics(!showMetrics)}
                className={`w-12 h-6 rounded-full transition-colors ${
                  showMetrics ? 'bg-blue-500' : 'bg-gray-600'
                }`}
              >
                <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
                  showMetrics ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </div>
            
            <div>
              <label className="text-sm text-gray-400 block mb-1">Level of Detail</label>
              <select
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded"
                value={levelOfDetail}
                onChange={(e) => setLevelOfDetail(e.target.value)}
              >
                <option value="overview">Overview</option>
                <option value="standard">Standard</option>
                <option value="detailed">Detailed</option>
              </select>
            </div>
          </div>
        )}
      </div>
      
      {/* Analysis Tools */}
      <div className="border-b border-gray-700">
        <button
          onClick={() => toggleSection('analysis')}
          className="w-full p-4 flex items-center justify-between hover:bg-gray-700/50 transition-colors"
        >
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-blue-400" />
            <span className="font-semibold">Analysis</span>
          </div>
          {expandedSections.analysis ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </button>
        
        {expandedSections.analysis && (
          <div className="p-4 space-y-3">
            <button className="w-full px-4 py-2 bg-blue-500/20 text-blue-400 rounded hover:bg-blue-500/30 transition-colors">
              Find Critical Paths
            </button>
            <button className="w-full px-4 py-2 bg-green-500/20 text-green-400 rounded hover:bg-green-500/30 transition-colors">
              Identify Bottlenecks
            </button>
            <button className="w-full px-4 py-2 bg-purple-500/20 text-purple-400 rounded hover:bg-purple-500/30 transition-colors">
              Analyze Dependencies
            </button>
          </div>
        )}
      </div>
      
      {/* Failure Simulation */}
      <div>
        <button
          onClick={() => toggleSection('simulation')}
          className="w-full p-4 flex items-center justify-between hover:bg-gray-700/50 transition-colors"
        >
          <div className="flex items-center space-x-2">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            <span className="font-semibold">Failure Simulation</span>
          </div>
          {expandedSections.simulation ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </button>
        
        {expandedSections.simulation && (
          <div className="p-4">
            <FailureSimulationPanel 
              failureNodes={failureNodes}
              setFailureNodes={setFailureNodes}
            />
          </div>
        )}
      </div>
    </aside>
  );
};

// Failure Simulation Panel
const FailureSimulationPanel = ({ failureNodes, setFailureNodes }) => {
  const [selectedNodes, setSelectedNodes] = useState([]);
  
  const simulateFailure = () => {
    setFailureNodes(new Set(selectedNodes));
  };
  
  const clearSimulation = () => {
    setFailureNodes(new Set());
    setSelectedNodes([]);
  };
  
  return (
    <div className="space-y-3">
      <div>
        <label className="text-sm text-gray-400 block mb-2">Select Components to Fail</label>
        <div className="space-y-2 max-h-40 overflow-y-auto">
          {['node1', 'broker1', 'application5', 'topic10'].map(nodeId => (
            <label key={nodeId} className="flex items-center space-x-2">
              <input
                type="checkbox"
                className="rounded border-gray-600"
                checked={selectedNodes.includes(nodeId)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSelectedNodes([...selectedNodes, nodeId]);
                  } else {
                    setSelectedNodes(selectedNodes.filter(id => id !== nodeId));
                  }
                }}
              />
              <span className="text-sm">{nodeId}</span>
            </label>
          ))}
        </div>
      </div>
      
      <div className="flex space-x-2">
        <button
          onClick={simulateFailure}
          className="flex-1 px-3 py-2 bg-red-500/20 text-red-400 rounded hover:bg-red-500/30 transition-colors"
        >
          Simulate
        </button>
        <button
          onClick={clearSimulation}
          className="flex-1 px-3 py-2 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
        >
          Clear
        </button>
      </div>
      
      {failureNodes.size > 0 && (
        <div className="mt-3 p-3 bg-red-900/20 rounded">
          <div className="text-sm text-red-400 font-semibold mb-1">Impact Analysis</div>
          <div className="text-xs text-gray-300 space-y-1">
            <div>Failed Components: {failureNodes.size}</div>
            <div>Affected Paths: 12</div>
            <div>Isolated Components: 3</div>
            <div>QoS Degradation: 35%</div>
          </div>
        </div>
      )}
    </div>
  );
};

// Progressive Graph Canvas with Advanced Features
const ProgressiveGraphCanvas = ({ 
  data, renderMode, clustering, showMetrics, 
  performanceMode, levelOfDetail, onPathSelect, failureNodes 
}) => {
  const canvasRef = useRef(null);
  const svgRef = useRef(null);
  const [renderProgress, setRenderProgress] = useState(0);
  const [clusters, setClusters] = useState([]);
  const renderBatchSize = useMemo(() => {
    const sizes = { quality: 50, balanced: 100, performance: 200 };
    return sizes[performanceMode] || 100;
  }, [performanceMode]);
  
  useEffect(() => {
    if (!data || data.nodes.length === 0) return;
    
    if (renderMode === 'webgl') {
      renderWebGL();
    } else if (renderMode === 'progressive') {
      renderProgressive();
    } else if (renderMode === 'virtualized') {
      renderVirtualized();
    } else {
      renderStandard();
    }
  }, [data, renderMode, clustering, levelOfDetail, failureNodes]);
  
  const renderProgressive = () => {
    const container = canvasRef.current;
    if (!container) return;
    
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Clear previous
    d3.select(svgRef.current).selectAll("*").remove();
    
    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);
    
    const g = svg.append("g");
    
    // Setup zoom
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
        updateLevelOfDetail(event.transform.k);
      });
    
    svg.call(zoom);
    
    // Process data in batches
    let processedNodes = [];
    let processedLinks = [];
    let currentBatch = 0;
    
    const processBatch = () => {
      const startIdx = currentBatch * renderBatchSize;
      const endIdx = Math.min(startIdx + renderBatchSize, data.nodes.length);
      
      // Add batch of nodes
      const batchNodes = data.nodes.slice(startIdx, endIdx);
      processedNodes = [...processedNodes, ...batchNodes];
      
      // Add corresponding links
      const nodeIds = new Set(processedNodes.map(n => n.id));
      processedLinks = data.links.filter(l => 
        nodeIds.has(l.source.id || l.source) && 
        nodeIds.has(l.target.id || l.target)
      );
      
      // Update visualization
      updateVisualization(g, processedNodes, processedLinks, width, height);
      
      // Update progress
      const progress = Math.min(100, (endIdx / data.nodes.length) * 100);
      setRenderProgress(progress);
      
      currentBatch++;
      
      // Continue if more batches
      if (endIdx < data.nodes.length) {
        requestAnimationFrame(processBatch);
      } else {
        setRenderProgress(100);
        if (clustering) {
          applyClustering(g, processedNodes, processedLinks);
        }
      }
    };
    
    processBatch();
  };
  
  const updateVisualization = (g, nodes, links, width, height) => {
    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(50))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(20));
    
    // Links
    const link = g.selectAll(".link")
      .data(links, d => `${d.source.id || d.source}-${d.target.id || d.target}`);
    
    link.exit().remove();
    
    const linkEnter = link.enter()
      .append("line")
      .attr("class", "link")
      .attr("stroke", d => failureNodes.has(d.source.id) || failureNodes.has(d.target.id) ? "#ef4444" : "#4b5563")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", 1);
    
    // Nodes
    const node = g.selectAll(".node")
      .data(nodes, d => d.id);
    
    node.exit().remove();
    
    const nodeEnter = node.enter()
      .append("g")
      .attr("class", "node")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));
    
    nodeEnter.append("circle")
      .attr("r", d => {
        if (levelOfDetail === 'overview') return 3;
        if (levelOfDetail === 'standard') return 5 + d.criticality * 10;
        return 5 + d.criticality * 15;
      })
      .attr("fill", d => {
        if (failureNodes.has(d.id)) return "#ef4444";
        const colors = {
          'Node': '#3b82f6',
          'Application': '#10b981',
          'Topic': '#f59e0b',
          'Broker': '#8b5cf6'
        };
        return colors[d.type] || '#6b7280';
      })
      .attr("opacity", d => failureNodes.has(d.id) ? 0.5 : 0.9);
    
    if (levelOfDetail !== 'overview') {
      nodeEnter.append("text")
        .text(d => levelOfDetail === 'detailed' ? d.id : '')
        .attr("x", 0)
        .attr("y", -10)
        .attr("text-anchor", "middle")
        .attr("font-size", "8px")
        .attr("fill", "#e5e7eb");
    }
    
    // Update simulation
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      
      node.attr("transform", d => `translate(${d.x},${d.y})`);
    });
    
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    
    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  };
  
  const applyClustering = (g, nodes, links) => {
    // Group nodes by type or broker
    const groups = d3.group(nodes, d => d.type);
    const clusterCenters = new Map();
    
    let i = 0;
    for (const [key, group] of groups) {
      const angle = (i / groups.size) * 2 * Math.PI;
      const radius = 200;
      clusterCenters.set(key, {
        x: 400 + radius * Math.cos(angle),
        y: 300 + radius * Math.sin(angle)
      });
      i++;
    }
    
    // Apply cluster forces
    nodes.forEach(node => {
      const center = clusterCenters.get(node.type);
      if (center) {
        node.x = center.x + (Math.random() - 0.5) * 100;
        node.y = center.y + (Math.random() - 0.5) * 100;
      }
    });
  };
  
  const updateLevelOfDetail = (zoomLevel) => {
    // Automatically adjust detail based on zoom
    if (zoomLevel < 0.5) {
      setLevelOfDetail('overview');
    } else if (zoomLevel < 2) {
      setLevelOfDetail('standard');
    } else {
      setLevelOfDetail('detailed');
    }
  };
  
  const renderWebGL = () => {
    // WebGL rendering would go here - simplified for demo
    console.log("WebGL rendering mode - would use Three.js or similar");
  };
  
  const renderVirtualized = () => {
    // Virtualized rendering - only render visible nodes
    console.log("Virtualized rendering mode");
  };
  
  const renderStandard = () => {
    // Standard D3 rendering without progressive loading
    renderProgressive();
  };
  
  return (
    <div ref={canvasRef} className="w-full h-full bg-gray-900 relative">
      <svg ref={svgRef}></svg>
      
      {renderProgress < 100 && renderProgress > 0 && (
        <div className="absolute top-4 left-4 bg-gray-800 rounded-lg p-3">
          <div className="text-sm text-gray-400 mb-1">Rendering Progress</div>
          <div className="w-48 h-2 bg-gray-700 rounded-full">
            <div 
              className="h-2 bg-blue-500 rounded-full transition-all duration-300"
              style={{ width: `${renderProgress}%` }}
            />
          </div>
          <div className="text-xs text-gray-400 mt-1">{renderProgress.toFixed(0)}% Complete</div>
        </div>
      )}
    </div>
  );
};

// Metrics Panel
const MetricsPanel = ({ data }) => (
  <aside className="w-64 bg-gray-800 border-l border-gray-700 p-4">
    <h3 className="text-lg font-semibold mb-4 flex items-center">
      <TrendingUp className="w-5 h-5 mr-2 text-blue-400" />
      Real-time Metrics
    </h3>
    
    <div className="space-y-4">
      <MetricCard 
        label="Avg. Latency"
        value="12.5ms"
        trend="+2.3%"
        color="green"
      />
      <MetricCard 
        label="Message Rate"
        value="1.2K/s"
        trend="+15%"
        color="blue"
      />
      <MetricCard 
        label="Error Rate"
        value="0.02%"
        trend="-5%"
        color="green"
      />
      <MetricCard 
        label="CPU Usage"
        value="68%"
        trend="+8%"
        color="yellow"
      />
      
      <div className="pt-4 border-t border-gray-700">
        <h4 className="text-sm font-semibold text-gray-400 mb-2">Critical Paths</h4>
        <div className="space-y-2">
          <PathIndicator from="app1" to="topic5" status="healthy" />
          <PathIndicator from="app3" to="topic8" status="warning" />
          <PathIndicator from="app7" to="topic2" status="critical" />
        </div>
      </div>
    </div>
  </aside>
);

// Metric Card Component
const MetricCard = ({ label, value, trend, color }) => {
  const colorClasses = {
    green: 'text-green-400',
    blue: 'text-blue-400',
    yellow: 'text-yellow-400',
    red: 'text-red-400'
  };
  
  return (
    <div className="bg-gray-700 rounded-lg p-3">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className={`text-xl font-bold ${colorClasses[color]}`}>{value}</div>
      <div className="text-xs text-gray-400">{trend}</div>
    </div>
  );
};

// Path Indicator Component
const PathIndicator = ({ from, to, status }) => {
  const statusColors = {
    healthy: 'bg-green-500',
    warning: 'bg-yellow-500',
    critical: 'bg-red-500'
  };
  
  return (
    <div className="flex items-center space-x-2">
      <div className={`w-2 h-2 rounded-full ${statusColors[status]}`} />
      <span className="text-xs text-gray-400">{from} → {to}</span>
    </div>
  );
};

// Path Analysis Overlay
const PathAnalysisOverlay = ({ path, onClose }) => (
  <div className="absolute top-4 right-4 bg-gray-800 rounded-lg shadow-lg p-4 w-80">
    <div className="flex items-center justify-between mb-3">
      <h3 className="font-semibold">Path Analysis</h3>
      <button onClick={onClose} className="text-gray-400 hover:text-gray-200">×</button>
    </div>
    
    <div className="space-y-3">
      <div>
        <div className="text-xs text-gray-400">Path Components</div>
        <div className="text-sm">{path?.length || 0} nodes</div>
      </div>
      <div>
        <div className="text-xs text-gray-400">Total Latency</div>
        <div className="text-sm">24.5ms</div>
      </div>
      <div>
        <div className="text-xs text-gray-400">Reliability Score</div>
        <div className="text-sm">98.5%</div>
      </div>
    </div>
  </div>
);

// Performance Indicator
const PerformanceIndicator = ({ data, performanceMode }) => (
  <div className="absolute bottom-4 left-4 bg-gray-800 rounded-lg p-2 text-xs">
    <div className="flex items-center space-x-4">
      <div className="flex items-center space-x-1">
        <Cpu className="w-3 h-3 text-blue-400" />
        <span>{data.nodes.length} nodes</span>
      </div>
      <div className="flex items-center space-x-1">
        <GitMerge className="w-3 h-3 text-green-400" />
        <span>{data.links.length} edges</span>
      </div>
      <div className="flex items-center space-x-1">
        <Zap className="w-3 h-3 text-yellow-400" />
        <span>{performanceMode}</span>
      </div>
    </div>
  </div>
);

// Generate large dataset for testing
const generateLargeDataset = () => {
  const nodes = [];
  const links = [];
  
  // Create a large number of nodes
  const nodeConfig = {
    Node: 10,
    Broker: 5,
    Application: 100,
    Topic: 200
  };
  
  // Generate nodes with properties
  for (const [type, count] of Object.entries(nodeConfig)) {
    for (let i = 0; i < count; i++) {
      nodes.push({
        id: `${type.toLowerCase()}${i + 1}`,
        type: type,
        criticality: Math.random(),
        qos: Math.random(),
        metrics: {
          cpu: Math.random() * 100,
          memory: Math.random() * 100,
          latency: Math.random() * 50,
          throughput: Math.random() * 1000
        },
        cluster: Math.floor(Math.random() * 5),
        status: Math.random() > 0.9 ? 'warning' : 'healthy'
      });
    }
  }
  
  // Generate relationships
  // Applications to Topics (PUBLISHES_TO and SUBSCRIBES_TO)
  for (let i = 0; i < 300; i++) {
    const appId = `application${Math.floor(Math.random() * 100) + 1}`;
    const topicId = `topic${Math.floor(Math.random() * 200) + 1}`;
    links.push({
      source: appId,
      target: topicId,
      type: Math.random() > 0.5 ? 'PUBLISHES_TO' : 'SUBSCRIBES_TO',
      weight: Math.random() * 5,
      metrics: {
        messageRate: Math.random() * 1000,
        avgSize: Math.random() * 1024,
        errorRate: Math.random() * 0.01
      }
    });
  }
  
  // Brokers to Topics (ROUTES)
  for (let i = 1; i <= 200; i++) {
    const brokerId = `broker${Math.floor(Math.random() * 5) + 1}`;
    links.push({
      source: brokerId,
      target: `topic${i}`,
      type: 'ROUTES',
      weight: Math.random() * 3
    });
  }
  
  // Applications to Nodes (RUNS_ON)
  for (let i = 1; i <= 100; i++) {
    const nodeId = `node${Math.floor(Math.random() * 10) + 1}`;
    links.push({
      source: `application${i}`,
      target: nodeId,
      type: 'RUNS_ON',
      weight: 1
    });
  }
  
  // Some DEPENDS_ON relationships
  for (let i = 0; i < 50; i++) {
    const app1 = `application${Math.floor(Math.random() * 100) + 1}`;
    const app2 = `application${Math.floor(Math.random() * 100) + 1}`;
    if (app1 !== app2) {
      links.push({
        source: app1,
        target: app2,
        type: 'DEPENDS_ON',
        weight: Math.random() * 2
      });
    }
  }
  
  return { nodes, links };
};

export default AdvancedGraphVisualizer;