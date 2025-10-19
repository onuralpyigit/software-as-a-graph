"""
Report Generator

Generates comprehensive reports from analysis results.
Supports multiple output formats (JSON, HTML, PDF, Markdown) and report types.

Capabilities:
- Executive summaries
- Detailed technical reports
- Comparison reports
- Trend analysis reports
- Custom report templates
- Multi-format export
"""

import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import logging


class ReportFormat(Enum):
    """Report output formats"""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    TEXT = "text"


class ReportType(Enum):
    """Types of reports"""
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    COMPARISON = "comparison"
    TREND = "trend"
    INCIDENT = "incident"
    HEALTH = "health"
    COMPREHENSIVE = "comprehensive"


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    report_type: ReportType
    format: ReportFormat
    include_charts: bool = True
    include_recommendations: bool = True
    include_raw_data: bool = False
    max_detail_level: int = 3
    title: Optional[str] = None
    author: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'report_type': self.report_type.value,
            'format': self.format.value,
            'include_charts': self.include_charts,
            'include_recommendations': self.include_recommendations,
            'include_raw_data': self.include_raw_data,
            'max_detail_level': self.max_detail_level,
            'title': self.title,
            'author': self.author
        }


class ReportGenerator:
    """
    Generates comprehensive reports from analysis results
    
    Features:
    - Multiple report types
    - Multiple output formats
    - Customizable templates
    - Charts and visualizations
    - Recommendations
    - Executive summaries
    """
    
    def __init__(self):
        """Initialize report generator"""
        self.logger = logging.getLogger(__name__)
    
    def generate_report(self,
                       analysis_results: Dict[str, Any],
                       config: Optional[ReportConfig] = None,
                       output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive report
        
        Args:
            analysis_results: Dictionary with analysis results
            config: Report configuration
            output_path: Path to save report
        
        Returns:
            Report content as string
        """
        if config is None:
            config = ReportConfig(
                report_type=ReportType.COMPREHENSIVE,
                format=ReportFormat.HTML
            )
        
        self.logger.info(f"Generating {config.report_type.value} report in {config.format.value} format...")
        
        # Generate based on type
        if config.report_type == ReportType.EXECUTIVE:
            content = self._generate_executive_report(analysis_results, config)
        elif config.report_type == ReportType.TECHNICAL:
            content = self._generate_technical_report(analysis_results, config)
        elif config.report_type == ReportType.COMPARISON:
            content = self._generate_comparison_report(analysis_results, config)
        elif config.report_type == ReportType.HEALTH:
            content = self._generate_health_report(analysis_results, config)
        elif config.report_type == ReportType.INCIDENT:
            content = self._generate_incident_report(analysis_results, config)
        else:
            content = self._generate_comprehensive_report(analysis_results, config)
        
        # Save if path provided
        if output_path:
            Path(output_path).write_text(content)
            self.logger.info(f"Saved report to {output_path}")
        
        return content
    
    def generate_executive_summary(self,
                                  analysis_results: Dict[str, Any]) -> str:
        """
        Generate executive summary (brief overview)
        
        Args:
            analysis_results: Dictionary with analysis results
        
        Returns:
            Executive summary as string
        """
        summary_parts = []
        
        # System overview
        if 'topology' in analysis_results:
            topo = analysis_results['topology']
            summary_parts.append(
                f"System consists of {topo.get('node_count', 0)} components "
                f"with {topo.get('edge_count', 0)} connections."
            )
        
        # Critical findings
        if 'criticality' in analysis_results:
            critical_count = sum(
                1 for score in analysis_results['criticality'].values()
                if score > 0.8
            )
            if critical_count > 0:
                summary_parts.append(
                    f"{critical_count} components identified as critical."
                )
        
        # Health status
        if 'health' in analysis_results:
            health = analysis_results['health']
            summary_parts.append(
                f"Overall health: {health.get('overall_health', 0):.1f}% "
                f"({health.get('status', 'UNKNOWN')})"
            )
        
        # Key recommendations
        if 'recommendations' in analysis_results:
            recs = analysis_results['recommendations']
            if recs:
                summary_parts.append(
                    f"{len(recs)} recommendations for improvement."
                )
        
        return " ".join(summary_parts)
    
    def generate_json_report(self,
                            analysis_results: Dict[str, Any],
                            include_metadata: bool = True) -> str:
        """
        Generate JSON report
        
        Args:
            analysis_results: Dictionary with analysis results
            include_metadata: Include metadata in output
        
        Returns:
            JSON string
        """
        report = {
            'analysis_results': analysis_results
        }
        
        if include_metadata:
            report['metadata'] = {
                'generated_at': datetime.now().isoformat(),
                'generator': 'ReportGenerator',
                'version': '1.0'
            }
        
        return json.dumps(report, indent=2)
    
    def compare_analyses(self,
                        baseline_results: Dict[str, Any],
                        current_results: Dict[str, Any],
                        output_path: Optional[str] = None) -> str:
        """
        Generate comparison report between two analyses
        
        Args:
            baseline_results: Baseline analysis results
            current_results: Current analysis results
            output_path: Path to save report
        
        Returns:
            HTML comparison report
        """
        self.logger.info("Generating comparison report...")
        
        # Calculate differences
        differences = self._calculate_differences(baseline_results, current_results)
        
        # Generate HTML
        html = self._create_comparison_html(differences, baseline_results, current_results)
        
        if output_path:
            Path(output_path).write_text(html)
            self.logger.info(f"Saved comparison report to {output_path}")
        
        return html
    
    def generate_incident_report(self,
                                incident_data: Dict[str, Any],
                                simulation_results: Optional[Dict] = None,
                                output_path: Optional[str] = None) -> str:
        """
        Generate incident analysis report
        
        Args:
            incident_data: Incident information
            simulation_results: Optional simulation results
            output_path: Path to save report
        
        Returns:
            HTML incident report
        """
        config = ReportConfig(
            report_type=ReportType.INCIDENT,
            format=ReportFormat.HTML,
            title=f"Incident Report: {incident_data.get('incident_id', 'Unknown')}"
        )
        
        # Combine data
        combined = {
            'incident': incident_data,
            'simulation': simulation_results or {}
        }
        
        return self.generate_report(combined, config, output_path)
    
    def _generate_executive_report(self,
                                   results: Dict[str, Any],
                                   config: ReportConfig) -> str:
        """Generate executive report (high-level overview)"""
        
        if config.format == ReportFormat.HTML:
            return self._create_executive_html(results, config)
        elif config.format == ReportFormat.JSON:
            return self.generate_json_report(results)
        elif config.format == ReportFormat.MARKDOWN:
            return self._create_executive_markdown(results, config)
        else:
            return self._create_executive_text(results, config)
    
    def _generate_technical_report(self,
                                   results: Dict[str, Any],
                                   config: ReportConfig) -> str:
        """Generate detailed technical report"""
        
        if config.format == ReportFormat.HTML:
            return self._create_technical_html(results, config)
        elif config.format == ReportFormat.JSON:
            return self.generate_json_report(results, include_metadata=True)
        else:
            return self._create_technical_markdown(results, config)
    
    def _generate_comprehensive_report(self,
                                      results: Dict[str, Any],
                                      config: ReportConfig) -> str:
        """Generate comprehensive report with all details"""
        
        if config.format == ReportFormat.HTML:
            return self._create_comprehensive_html(results, config)
        elif config.format == ReportFormat.JSON:
            return self.generate_json_report(results, include_metadata=True)
        else:
            return self._create_comprehensive_markdown(results, config)
    
    def _generate_comparison_report(self,
                                   results: Dict[str, Any],
                                   config: ReportConfig) -> str:
        """Generate comparison report"""
        
        if 'baseline' not in results or 'current' not in results:
            raise ValueError("Comparison report requires 'baseline' and 'current' results")
        
        return self.compare_analyses(
            results['baseline'],
            results['current']
        )
    
    def _generate_health_report(self,
                               results: Dict[str, Any],
                               config: ReportConfig) -> str:
        """Generate health status report"""
        
        if config.format == ReportFormat.HTML:
            return self._create_health_html(results, config)
        else:
            return self._create_health_markdown(results, config)
    
    def _generate_incident_report(self,
                                 results: Dict[str, Any],
                                 config: ReportConfig) -> str:
        """Generate incident analysis report"""
        
        if config.format == ReportFormat.HTML:
            return self._create_incident_html(results, config)
        else:
            return self._create_incident_markdown(results, config)
    
    def _create_executive_html(self,
                              results: Dict[str, Any],
                              config: ReportConfig) -> str:
        """Create executive HTML report"""
        
        title = config.title or "Executive Summary Report"
        summary = self.generate_executive_summary(results)
        
        # Extract key metrics
        key_metrics = self._extract_key_metrics(results)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 40px;
                    background: #f5f7fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 15px;
                    margin-bottom: 30px;
                }}
                .summary {{
                    background: #ecf0f1;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                    font-size: 16px;
                    line-height: 1.6;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 36px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .metric-label {{
                    font-size: 14px;
                    opacity: 0.9;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .footer {{
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ecf0f1;
                    color: #7f8c8d;
                    font-size: 12px;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 {title}</h1>
                
                <div class="summary">
                    <strong>Executive Summary:</strong><br><br>
                    {summary}
                </div>
                
                <h2>Key Metrics</h2>
                <div class="metrics">
                    {self._create_metric_cards_html(key_metrics)}
                </div>
                
                {self._create_recommendations_html(results) if config.include_recommendations else ''}
                
                <div class="footer">
                    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    {f' | Author: {config.author}' if config.author else ''}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_technical_html(self,
                              results: Dict[str, Any],
                              config: ReportConfig) -> str:
        """Create detailed technical HTML report"""
        
        title = config.title or "Technical Analysis Report"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #f5f7fa;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; }}
                h2 {{ color: #34495e; margin-top: 40px; border-left: 4px solid #3498db; padding-left: 15px; }}
                .section {{
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th {{
                    background: #3498db;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                td {{
                    padding: 12px;
                    border-bottom: 1px solid #ecf0f1;
                }}
                tr:hover {{ background: #f8f9fa; }}
                .badge {{
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 3px;
                    font-size: 12px;
                    font-weight: 600;
                }}
                .badge-critical {{ background: #e74c3c; color: white; }}
                .badge-high {{ background: #e67e22; color: white; }}
                .badge-medium {{ background: #f39c12; color: white; }}
                .badge-low {{ background: #3498db; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔧 {title}</h1>
                
                {self._create_topology_section_html(results)}
                {self._create_criticality_section_html(results)}
                {self._create_centrality_section_html(results)}
                {self._create_simulation_section_html(results)}
                
                <div class="footer" style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px;">
                    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_comprehensive_html(self,
                                  results: Dict[str, Any],
                                  config: ReportConfig) -> str:
        """Create comprehensive HTML report"""
        
        title = config.title or "Comprehensive System Analysis Report"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #f5f7fa;
                }}
                .container {{
                    max-width: 1600px;
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                }}
                h1 {{ color: #2c3e50; }}
                .toc {{
                    background: #ecf0f1;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .toc a {{
                    display: block;
                    padding: 5px 0;
                    color: #3498db;
                    text-decoration: none;
                }}
                .toc a:hover {{ text-decoration: underline; }}
                .section {{
                    margin: 40px 0;
                    padding: 30px;
                    background: #f8f9fa;
                    border-radius: 8px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📋 {title}</h1>
                
                <div class="toc">
                    <h2>Table of Contents</h2>
                    <a href="#executive">1. Executive Summary</a>
                    <a href="#topology">2. System Topology</a>
                    <a href="#criticality">3. Criticality Analysis</a>
                    <a href="#centrality">4. Centrality Metrics</a>
                    <a href="#simulation">5. Failure Simulation</a>
                    <a href="#recommendations">6. Recommendations</a>
                </div>
                
                <div id="executive" class="section">
                    <h2>Executive Summary</h2>
                    {self.generate_executive_summary(results)}
                </div>
                
                <div id="topology" class="section">
                    {self._create_topology_section_html(results)}
                </div>
                
                <div id="criticality" class="section">
                    {self._create_criticality_section_html(results)}
                </div>
                
                <div id="centrality" class="section">
                    {self._create_centrality_section_html(results)}
                </div>
                
                <div id="simulation" class="section">
                    {self._create_simulation_section_html(results)}
                </div>
                
                <div id="recommendations" class="section">
                    {self._create_recommendations_html(results)}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_health_html(self,
                           results: Dict[str, Any],
                           config: ReportConfig) -> str:
        """Create health status HTML report"""
        
        health = results.get('health', {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>System Health Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 40px;
                    background: #f5f7fa;
                }}
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                }}
                .health-score {{
                    text-align: center;
                    padding: 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 10px;
                    margin: 30px 0;
                }}
                .score-value {{
                    font-size: 72px;
                    font-weight: bold;
                }}
                .status-badge {{
                    display: inline-block;
                    padding: 10px 30px;
                    border-radius: 25px;
                    font-size: 18px;
                    font-weight: 600;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏥 System Health Report</h1>
                
                <div class="health-score">
                    <div class="score-value">{health.get('overall_health', 0):.1f}%</div>
                    <div class="status-badge">{health.get('status', 'UNKNOWN')}</div>
                </div>
                
                <h2>Health Components</h2>
                <ul>
                    <li>Connectivity: {health.get('connectivity', 0):.1f}%</li>
                    <li>Component Health: {health.get('components', 0):.1f}%</li>
                    <li>Redundancy: {health.get('redundancy', 0):.1f}%</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_incident_html(self,
                             results: Dict[str, Any],
                             config: ReportConfig) -> str:
        """Create incident analysis HTML report"""
        
        incident = results.get('incident', {})
        simulation = results.get('simulation', {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Incident Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 40px;
                    background: #f5f7fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                }}
                .alert {{
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 20px;
                    margin: 20px 0;
                }}
                .timeline {{
                    border-left: 3px solid #3498db;
                    padding-left: 30px;
                    margin: 30px 0;
                }}
                .timeline-item {{
                    margin: 20px 0;
                    position: relative;
                }}
                .timeline-item::before {{
                    content: '';
                    position: absolute;
                    left: -36px;
                    width: 12px;
                    height: 12px;
                    background: #3498db;
                    border-radius: 50%;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚨 Incident Report: {incident.get('incident_id', 'Unknown')}</h1>
                
                <div class="alert">
                    <strong>Status:</strong> {incident.get('status', 'Unknown')}<br>
                    <strong>Severity:</strong> {incident.get('severity', 'Unknown')}<br>
                    <strong>Occurred:</strong> {incident.get('timestamp', 'Unknown')}
                </div>
                
                <h2>Impact Analysis</h2>
                <p>Failed components: {len(incident.get('failed_components', []))}</p>
                <p>Affected components: {len(incident.get('affected_components', []))}</p>
                
                {self._create_simulation_section_html(simulation) if simulation else ''}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_executive_markdown(self,
                                   results: Dict[str, Any],
                                   config: ReportConfig) -> str:
        """Create executive markdown report"""
        
        summary = self.generate_executive_summary(results)
        key_metrics = self._extract_key_metrics(results)
        
        md = f"""# Executive Summary Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

{summary}

## Key Metrics

"""
        
        for metric_name, metric_value in key_metrics.items():
            md += f"- **{metric_name}**: {metric_value}\n"
        
        if config.include_recommendations and 'recommendations' in results:
            md += "\n## Recommendations\n\n"
            for i, rec in enumerate(results['recommendations'], 1):
                md += f"{i}. {rec}\n"
        
        return md
    
    def _create_technical_markdown(self,
                                   results: Dict[str, Any],
                                   config: ReportConfig) -> str:
        """Create technical markdown report"""
        
        md = f"""# Technical Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Topology

"""
        
        if 'topology' in results:
            topo = results['topology']
            md += f"- Nodes: {topo.get('node_count', 0)}\n"
            md += f"- Edges: {topo.get('edge_count', 0)}\n"
            md += f"- Density: {topo.get('density', 0):.3f}\n"
        
        if 'criticality' in results:
            md += "\n## Criticality Analysis\n\n"
            crit_items = sorted(
                results['criticality'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for component, score in crit_items:
                md += f"- {component}: {score:.3f}\n"
        
        return md
    
    def _create_comprehensive_markdown(self,
                                      results: Dict[str, Any],
                                      config: ReportConfig) -> str:
        """Create comprehensive markdown report"""
        
        md = f"""# Comprehensive System Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Topology](#system-topology)
3. [Criticality Analysis](#criticality-analysis)
4. [Recommendations](#recommendations)

## Executive Summary

{self.generate_executive_summary(results)}

"""
        
        md += self._create_technical_markdown(results, config)
        
        return md
    
    def _create_health_markdown(self,
                               results: Dict[str, Any],
                               config: ReportConfig) -> str:
        """Create health markdown report"""
        
        health = results.get('health', {})
        
        return f"""# System Health Report

## Overall Health: {health.get('overall_health', 0):.1f}%

Status: **{health.get('status', 'UNKNOWN')}**

### Health Components

- Connectivity: {health.get('connectivity', 0):.1f}%
- Component Health: {health.get('components', 0):.1f}%
- Redundancy: {health.get('redundancy', 0):.1f}%
"""
    
    def _create_incident_markdown(self,
                                 results: Dict[str, Any],
                                 config: ReportConfig) -> str:
        """Create incident markdown report"""
        
        incident = results.get('incident', {})
        
        return f"""# Incident Report: {incident.get('incident_id', 'Unknown')}

**Status**: {incident.get('status', 'Unknown')}  
**Severity**: {incident.get('severity', 'Unknown')}  
**Occurred**: {incident.get('timestamp', 'Unknown')}

## Impact

- Failed components: {len(incident.get('failed_components', []))}
- Affected components: {len(incident.get('affected_components', []))}

## Failed Components

{self._list_items(incident.get('failed_components', []))}
"""
    
    def _create_executive_text(self,
                              results: Dict[str, Any],
                              config: ReportConfig) -> str:
        """Create plain text executive report"""
        
        summary = self.generate_executive_summary(results)
        
        text = f"""EXECUTIVE SUMMARY REPORT
{'=' * 50}

{summary}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return text
    
    def _create_comparison_html(self,
                               differences: Dict[str, Any],
                               baseline: Dict[str, Any],
                               current: Dict[str, Any]) -> str:
        """Create comparison HTML"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comparison Report</title>
            <style>
                body {{ font-family: Arial; margin: 40px; background: #f5f7fa; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }}
                .comparison {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 30px 0; }}
                .column {{ padding: 20px; background: #f8f9fa; border-radius: 5px; }}
                .improved {{ color: #27ae60; font-weight: 600; }}
                .degraded {{ color: #e74c3c; font-weight: 600; }}
                .unchanged {{ color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Comparison Report</h1>
                
                <div class="comparison">
                    <div class="column">
                        <h2>Baseline</h2>
                        <p>Analysis from previous period</p>
                    </div>
                    <div class="column">
                        <h2>Current</h2>
                        <p>Current analysis results</p>
                    </div>
                </div>
                
                <h2>Key Differences</h2>
                {self._create_differences_html(differences)}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _calculate_differences(self,
                              baseline: Dict[str, Any],
                              current: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate differences between two analyses"""
        
        differences = {}
        
        # Compare topology
        if 'topology' in baseline and 'topology' in current:
            base_topo = baseline['topology']
            curr_topo = current['topology']
            
            differences['topology'] = {
                'nodes': curr_topo.get('node_count', 0) - base_topo.get('node_count', 0),
                'edges': curr_topo.get('edge_count', 0) - base_topo.get('edge_count', 0),
                'density': curr_topo.get('density', 0) - base_topo.get('density', 0)
            }
        
        return differences
    
    def _extract_key_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from results"""
        
        metrics = {}
        
        if 'topology' in results:
            topo = results['topology']
            metrics['Components'] = topo.get('node_count', 0)
            metrics['Connections'] = topo.get('edge_count', 0)
        
        if 'criticality' in results:
            critical = sum(1 for s in results['criticality'].values() if s > 0.8)
            metrics['Critical Components'] = critical
        
        if 'health' in results:
            metrics['Overall Health'] = f"{results['health'].get('overall_health', 0):.1f}%"
        
        return metrics
    
    def _create_metric_cards_html(self, metrics: Dict[str, Any]) -> str:
        """Create HTML for metric cards"""
        
        html = ""
        for label, value in metrics.items():
            html += f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """
        
        return html
    
    def _create_recommendations_html(self, results: Dict[str, Any]) -> str:
        """Create HTML for recommendations section"""
        
        if 'recommendations' not in results or not results['recommendations']:
            return ""
        
        recs_html = "<h2>Recommendations</h2><ol>"
        for rec in results['recommendations']:
            recs_html += f"<li>{rec}</li>"
        recs_html += "</ol>"
        
        return recs_html
    
    def _create_topology_section_html(self, results: Dict[str, Any]) -> str:
        """Create topology section HTML"""
        
        if 'topology' not in results:
            return ""
        
        topo = results['topology']
        
        return f"""
        <h2>System Topology</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Nodes</td><td>{topo.get('node_count', 0)}</td></tr>
            <tr><td>Edges</td><td>{topo.get('edge_count', 0)}</td></tr>
            <tr><td>Density</td><td>{topo.get('density', 0):.3f}</td></tr>
            <tr><td>Avg Degree</td><td>{topo.get('avg_degree', 0):.2f}</td></tr>
        </table>
        """
    
    def _create_criticality_section_html(self, results: Dict[str, Any]) -> str:
        """Create criticality section HTML"""
        
        if 'criticality' not in results:
            return ""
        
        items = sorted(
            results['criticality'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        html = "<h2>Top Critical Components</h2><table><tr><th>Component</th><th>Score</th></tr>"
        for component, score in items:
            badge = self._get_criticality_badge(score)
            html += f"<tr><td>{component}</td><td>{score:.3f} {badge}</td></tr>"
        html += "</table>"
        
        return html
    
    def _create_centrality_section_html(self, results: Dict[str, Any]) -> str:
        """Create centrality section HTML"""
        
        if 'centrality' not in results:
            return ""
        
        return "<h2>Centrality Analysis</h2><p>Centrality metrics available in detailed data.</p>"
    
    def _create_simulation_section_html(self, results: Dict[str, Any]) -> str:
        """Create simulation section HTML"""
        
        if 'simulation' not in results:
            return ""
        
        sim = results['simulation']
        
        return f"""
        <h2>Failure Simulation Results</h2>
        <p>Impact Score: {sim.get('impact_score', 0):.3f}</p>
        <p>Affected Components: {len(sim.get('affected_components', []))}</p>
        """
    
    def _create_differences_html(self, differences: Dict[str, Any]) -> str:
        """Create differences HTML"""
        
        html = "<ul>"
        for category, diffs in differences.items():
            html += f"<li><strong>{category.title()}</strong><ul>"
            for key, value in diffs.items():
                change_class = "improved" if value > 0 else "degraded" if value < 0 else "unchanged"
                sign = "+" if value > 0 else ""
                html += f"<li class='{change_class}'>{key}: {sign}{value}</li>"
            html += "</ul></li>"
        html += "</ul>"
        
        return html
    
    def _get_criticality_badge(self, score: float) -> str:
        """Get HTML badge for criticality score"""
        
        if score >= 0.8:
            return '<span class="badge badge-critical">CRITICAL</span>'
        elif score >= 0.6:
            return '<span class="badge badge-high">HIGH</span>'
        elif score >= 0.4:
            return '<span class="badge badge-medium">MEDIUM</span>'
        else:
            return '<span class="badge badge-low">LOW</span>'
    
    def _list_items(self, items: List[str]) -> str:
        """Convert list to markdown list"""
        return "\n".join(f"- {item}" for item in items)
