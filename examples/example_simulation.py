"""
Example: Simulation Modules Usage

Demonstrates:
1. Failure simulation (single, multiple, cascading)
2. Impact calculation (business, technical, financial)
3. Validation against historical data
4. Complete failure analysis workflow
"""

import sys
from pathlib import Path
import networkx as nx
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

from src.simulation.failure_simulator import (
    FailureSimulator, FailureType, FailureMode
)
from src.simulation.impact_calculator import ImpactCalculator
from src.simulation.validation_engine import (
    ValidationEngine, HistoricalIncident
)


def create_example_system():
    """Create example system for simulation"""
    
    G = nx.DiGraph()
    
    # Add components with metadata
    components = {
        # Frontend
        'WebApp': {'type': 'Application', 'criticality_score': 0.7},
        'MobileApp': {'type': 'Application', 'criticality_score': 0.6},
        
        # API Layer
        'APIGateway': {'type': 'Application', 'criticality_score': 0.9},
        
        # Services
        'AuthService': {'type': 'Application', 'criticality_score': 0.95},
        'UserService': {'type': 'Application', 'criticality_score': 0.85},
        'OrderService': {'type': 'Application', 'criticality_score': 0.9},
        'PaymentService': {'type': 'Application', 'criticality_score': 0.95},
        
        # Message Broker
        'MainBroker': {'type': 'Broker', 'criticality_score': 0.85},
        
        # Databases
        'UserDB': {'type': 'Node', 'criticality_score': 0.9},
        'OrderDB': {'type': 'Node', 'criticality_score': 0.95},
    }
    
    for name, attrs in components.items():
        G.add_node(name, **attrs)
    
    # Add dependencies
    edges = [
        ('WebApp', 'APIGateway'),
        ('MobileApp', 'APIGateway'),
        ('APIGateway', 'AuthService'),
        ('APIGateway', 'UserService'),
        ('APIGateway', 'OrderService'),
        ('OrderService', 'PaymentService'),
        ('UserService', 'UserDB'),
        ('OrderService', 'OrderDB'),
        ('OrderService', 'MainBroker'),
        ('PaymentService', 'MainBroker'),
    ]
    
    G.add_edges_from(edges, weight=1.0)
    
    return G


def example_1_single_failure():
    """Example 1: Simulate single component failure"""
    
    print("\n" + "=" * 70)
    print("Example 1: Single Component Failure Simulation")
    print("=" * 70)
    
    # Create system
    print("\n[Step 1] Creating system...")
    G = create_example_system()
    print(f"  ✓ System with {len(G)} components")
    
    # Initialize simulator
    simulator = FailureSimulator()
    
    # Simulate APIGateway failure
    print("\n[Step 2] Simulating APIGateway failure...")
    result = simulator.simulate_single_failure(
        G,
        'APIGateway',
        failure_type=FailureType.COMPLETE,
        enable_cascade=True
    )
    
    print(f"\n  Results:")
    print(f"    Failed components:   {len(result.failed_components)}")
    print(f"    Affected components: {len(result.affected_components)}")
    print(f"    Isolated components: {len(result.isolated_components)}")
    print(f"    Impact score:        {result.impact_score:.3f}")
    print(f"    Resilience score:    {result.resilience_score:.3f}")
    print(f"    Service continuity:  {result.service_continuity:.1%}")
    
    if result.affected_components:
        print(f"\n  Affected components:")
        for comp in result.affected_components[:5]:
            print(f"    - {comp}")


def example_2_impact_calculation():
    """Example 2: Calculate business and technical impact"""
    
    print("\n" + "=" * 70)
    print("Example 2: Impact Calculation")
    print("=" * 70)
    
    G = create_example_system()
    
    # Initialize impact calculator
    print("\n[Step 1] Initializing impact calculator...")
    calculator = ImpactCalculator(
        hourly_downtime_cost_usd=10000.0,
        user_base_size=100000,
        avg_transaction_value_usd=50.0
    )
    
    # Calculate impact for critical service
    print("\n[Step 2] Calculating impact for OrderService...")
    impact = calculator.calculate_component_impact(G, 'OrderService')
    
    print(f"\n  Component Impact Analysis:")
    print(f"    Component:         {impact.component}")
    print(f"    Type:              {impact.component_type}")
    print(f"    Direct impact:     {impact.direct_impact:.3f}")
    print(f"    Indirect impact:   {impact.indirect_impact:.3f}")
    print(f"    Total impact:      {impact.total_impact:.3f}")
    print(f"    Criticality:       {impact.criticality_level}")
    
    print(f"\n  Impact Metrics:")
    metrics = impact.metrics
    print(f"    Business impact:   {metrics.business_impact:.3f}")
    print(f"    Technical impact:  {metrics.technical_impact:.3f}")
    print(f"    Financial impact:  ${metrics.financial_impact_usd:,.2f}")
    print(f"    Affected users:    {metrics.affected_users:,}")
    print(f"    Recovery estimate: {metrics.recovery_time_estimate_hours:.1f} hours")
    
    # Calculate business impact
    print("\n[Step 3] Calculating business impact (2-hour outage)...")
    business_impact = calculator.calculate_business_impact(
        G,
        'OrderService',
        downtime_hours=2.0
    )
    
    biz_metrics = business_impact['business_metrics']
    print(f"\n  Business Metrics:")
    print(f"    Affected users:      {biz_metrics['affected_users']:,}")
    print(f"    Lost transactions:   {biz_metrics['lost_transactions']:,}")
    print(f"    Revenue loss:        ${biz_metrics['revenue_loss_usd']:,.2f}")
    print(f"    Downtime cost:       ${biz_metrics['downtime_cost_usd']:,.2f}")
    print(f"    Total financial:     ${biz_metrics['total_financial_impact_usd']:,.2f}")


def example_3_cascading_impact():
    """Example 3: Analyze cascading impact"""
    
    print("\n" + "=" * 70)
    print("Example 3: Cascading Impact Analysis")
    print("=" * 70)
    
    G = create_example_system()
    calculator = ImpactCalculator()
    
    # Simulate cascading failure
    print("\n[Step 1] Analyzing cascading impact from AuthService...")
    cascade = calculator.calculate_cascading_impact(
        G,
        'AuthService',
        propagation_factor=0.6
    )
    
    print(f"\n  Cascading Impact Results:")
    print(f"    Initial component:    {cascade['initial_component']}")
    print(f"    Initial impact:       {cascade['initial_impact']:.3f}")
    print(f"    Affected components:  {cascade['affected_components']}")
    print(f"    Total cascade impact: {cascade['total_cascade_impact']:.3f}")
    print(f"    Cascade factor:       {cascade['cascade_factor']:.2f}x")
    
    print(f"\n  Impact Wave:")
    for comp, impact_val in list(cascade['impact_wave'].items())[:5]:
        print(f"    {comp:20s}: {impact_val:.3f}")


def example_4_sla_impact():
    """Example 4: Calculate SLA impact"""
    
    print("\n" + "=" * 70)
    print("Example 4: SLA Impact Assessment")
    print("=" * 70)
    
    G = create_example_system()
    calculator = ImpactCalculator(sla_availability_target=0.999)  # 99.9% SLA
    
    # Test different downtime scenarios
    components = ['APIGateway', 'OrderService', 'PaymentService']
    downtime_minutes = 30
    
    print(f"\n[Analyzing SLA impact of {downtime_minutes} minute outage]\n")
    
    for component in components:
        sla_impact = calculator.calculate_sla_impact(
            G,
            component,
            downtime_minutes
        )
        
        sla = sla_impact['sla_metrics']
        print(f"  {component}:")
        print(f"    Current availability:  {sla['current_availability']}%")
        print(f"    SLA target:            {sla['sla_target']}%")
        print(f"    SLA violated:          {sla['sla_violated']}")
        if sla['sla_violated']:
            print(f"    Breach severity:       {sla['breach_severity']:.3f}")
        print(f"    Availability nines:    {sla['nines']}")
        print()


def example_5_rank_components():
    """Example 5: Rank components by impact"""
    
    print("\n" + "=" * 70)
    print("Example 5: Rank Components by Impact")
    print("=" * 70)
    
    G = create_example_system()
    calculator = ImpactCalculator()
    
    print("\n[Ranking all components by potential impact...]")
    
    rankings = calculator.rank_components_by_impact(G, top_n=10)
    
    print(f"\n  Top 10 Components by Impact:")
    for rank, (component, impact_score) in enumerate(rankings, 1):
        print(f"    {rank:2d}. {component:20s}: {impact_score:.3f}")


def example_6_historical_validation():
    """Example 6: Validate against historical incidents"""
    
    print("\n" + "=" * 70)
    print("Example 6: Historical Data Validation")
    print("=" * 70)
    
    G = create_example_system()
    
    # Initialize validation engine
    print("\n[Step 1] Initializing validation engine...")
    validator = ValidationEngine()
    
    # Add historical incidents
    print("\n[Step 2] Loading historical incidents...")
    
    incidents = [
        {
            'incident_id': 'INC-001',
            'timestamp': '2024-01-15T14:30:00',
            'failed_components': ['APIGateway'],
            'affected_components': ['WebApp', 'MobileApp'],
            'actual_impact_score': 0.6,
            'actual_downtime_minutes': 45,
            'actual_users_affected': 60000,
            'actual_financial_impact_usd': 25000,
            'root_cause': 'Memory leak',
            'resolution_time_hours': 0.75
        },
        {
            'incident_id': 'INC-002',
            'timestamp': '2024-02-20T09:15:00',
            'failed_components': ['OrderDB'],
            'affected_components': ['OrderService', 'PaymentService'],
            'actual_impact_score': 0.8,
            'actual_downtime_minutes': 120,
            'actual_users_affected': 80000,
            'actual_financial_impact_usd': 50000,
            'root_cause': 'Disk failure',
            'resolution_time_hours': 2.0
        },
        {
            'incident_id': 'INC-003',
            'timestamp': '2024-03-10T16:45:00',
            'failed_components': ['MainBroker'],
            'affected_components': ['OrderService', 'PaymentService'],
            'actual_impact_score': 0.5,
            'actual_downtime_minutes': 30,
            'actual_users_affected': 50000,
            'actual_financial_impact_usd': 15000,
            'root_cause': 'Network issue',
            'resolution_time_hours': 0.5
        }
    ]
    
    validator.load_incidents_from_dict(incidents)
    print(f"  ✓ Loaded {len(incidents)} historical incidents")
    
    # Simulate and validate
    print("\n[Step 3] Simulating and validating incidents...")
    
    simulator = FailureSimulator()
    validation_results = []
    
    for incident in validator.historical_incidents:
        # Simulate the failure
        failed_component = incident.failed_components[0]
        sim_result = simulator.simulate_single_failure(
            G,
            failed_component,
            enable_cascade=True
        )
        
        # Validate
        val_result = validator.validate_simulation(G, incident, sim_result)
        validation_results.append(val_result)
        
        print(f"\n  {incident.incident_id}:")
        print(f"    Predicted impact:  {val_result.predicted_impact:.3f}")
        print(f"    Actual impact:     {val_result.actual_impact:.3f}")
        print(f"    Error:             {val_result.impact_error:.3f}")
        print(f"    Accuracy:          {val_result.accuracy_score:.3f}")
        print(f"    Confidence:        {val_result.confidence_level}")
    
    # Generate calibration report
    print("\n[Step 4] Generating calibration report...")
    report = validator.generate_calibration_report(validation_results)
    
    print(f"\n  Calibration Summary:")
    print(f"    Overall accuracy:  {report.overall_accuracy:.3f}")
    print(f"    MAE:               {report.mean_absolute_error:.3f}")
    print(f"    RMSE:              {report.root_mean_square_error:.3f}")
    print(f"    Precision:         {report.precision:.3f}")
    print(f"    Recall:            {report.recall:.3f}")
    print(f"    F1 Score:          {report.f1_score:.3f}")
    print(f"    Quality:           {report.calibration_quality}")


def example_7_complete_workflow():
    """Example 7: Complete failure analysis workflow"""
    
    print("\n" + "=" * 70)
    print("Example 7: Complete Failure Analysis Workflow")
    print("=" * 70)
    
    G = create_example_system()
    
    # Step 1: Identify critical components
    print("\n[Step 1] Identifying critical components...")
    calculator = ImpactCalculator()
    rankings = calculator.rank_components_by_impact(G, top_n=3)
    
    print(f"  Top 3 Critical Components:")
    for rank, (comp, score) in enumerate(rankings, 1):
        print(f"    {rank}. {comp}: {score:.3f}")
    
    critical_component = rankings[0][0]
    
    # Step 2: Simulate failure
    print(f"\n[Step 2] Simulating failure of {critical_component}...")
    simulator = FailureSimulator()
    sim_result = simulator.simulate_single_failure(
        G,
        critical_component,
        enable_cascade=True
    )
    
    print(f"  Simulation Results:")
    print(f"    Impact score:      {sim_result.impact_score:.3f}")
    print(f"    Affected:          {len(sim_result.affected_components)} components")
    print(f"    Service continuity: {sim_result.service_continuity:.1%}")
    
    # Step 3: Calculate business impact
    print(f"\n[Step 3] Calculating business impact...")
    impact = calculator.calculate_component_impact(G, critical_component)
    business = calculator.calculate_business_impact(
        G,
        critical_component,
        downtime_hours=1.0
    )
    
    print(f"  Business Impact (1-hour outage):")
    biz = business['business_metrics']
    print(f"    Affected users:    {biz['affected_users']:,}")
    print(f"    Lost transactions: {biz['lost_transactions']:,}")
    print(f"    Financial impact:  ${biz['total_financial_impact_usd']:,.2f}")
    
    # Step 4: Generate report
    print(f"\n[Step 4] Generating failure report...")
    report = simulator.generate_failure_report(G, sim_result)
    
    print(f"\n  Report Summary:")
    print(f"    Severity:          {report['summary']['severity']}")
    print(f"    Recommendations:   {len(report['recommendations'])}")
    
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"\n    {i}. {rec}")


def example_8_recovery_analysis():
    """Example 8: Recovery scenario analysis"""
    
    print("\n" + "=" * 70)
    print("Example 8: Recovery Scenario Analysis")
    print("=" * 70)
    
    G = create_example_system()
    calculator = ImpactCalculator()
    
    component = 'PaymentService'
    recovery_hours = 2.0
    
    print(f"\n[Analyzing recovery of {component}]")
    print(f"Recovery time: {recovery_hours} hours\n")
    
    recovery = calculator.estimate_recovery_impact(
        G,
        component,
        recovery_hours
    )
    
    print(f"  Recovery Analysis:")
    print(f"    Total impact:      {recovery['total_impact']:.3f}")
    print(f"    Financial impact:  ${recovery['financial_impact']:,.2f}")
    
    print(f"\n  Recovery Timeline:")
    for milestone in recovery['recovery_timeline']:
        print(f"    {milestone['elapsed_hours']:.1f}h "
              f"({milestone['recovery_percentage']:.0f}% recovered) "
              f"- Service level: {milestone['service_level']:.0f}%")


def main():
    """Run all examples"""
    
    print("\n" + "=" * 70)
    print("SIMULATION MODULES - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    try:
        example_1_single_failure()
        example_2_impact_calculation()
        example_3_cascading_impact()
        example_4_sla_impact()
        example_5_rank_components()
        example_6_historical_validation()
        example_7_complete_workflow()
        example_8_recovery_analysis()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\n📚 Summary of Capabilities:")
        print("  ✓ Single & multiple component failure simulation")
        print("  ✓ Cascading failure analysis")
        print("  ✓ Business, technical, and financial impact calculation")
        print("  ✓ SLA impact assessment")
        print("  ✓ Component ranking by impact")
        print("  ✓ Historical data validation")
        print("  ✓ Recovery scenario analysis")
        print("  ✓ Comprehensive failure reports")
        
        print("\n📖 Usage in Your Code:")
        print("""
from refactored.simulation.failure_simulator import FailureSimulator
from refactored.simulation.impact_calculator import ImpactCalculator
from refactored.simulation.validation_engine import ValidationEngine

# Simulate failure
simulator = FailureSimulator()
result = simulator.simulate_single_failure(graph, 'MyComponent')

# Calculate impact
calculator = ImpactCalculator(hourly_downtime_cost_usd=10000)
impact = calculator.calculate_component_impact(graph, 'MyComponent')

# Validate against history
validator = ValidationEngine()
validator.load_incidents_from_dict(historical_incidents)
report = validator.generate_calibration_report(validation_results)
        """)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
