import pytest
from datetime import datetime, timedelta
from src.ecs_sehi_analyzer.core.finance.quantum_finance import (
    QuantumFinanceEngine,
    BudgetAllocation,
    ResourceUsage,
    ResourceType
)

@pytest.fixture
def finance_engine():
    return QuantumFinanceEngine()

@pytest.fixture
def sample_allocations():
    now = datetime.now()
    return [
        BudgetAllocation(
            resource_type=ResourceType.COMPUTE,
            amount=10000.0,
            start_date=now,
            end_date=now + timedelta(days=30),
            project_id="PROJ-001"
        ),
        BudgetAllocation(
            resource_type=ResourceType.MATERIALS,
            amount=5000.0,
            start_date=now,
            end_date=now + timedelta(days=30),
            project_id="PROJ-001"
        )
    ]

def test_budget_allocation(finance_engine, sample_allocations):
    totals = finance_engine.allocate_budget("PROJ-001", sample_allocations)
    assert totals[ResourceType.COMPUTE.value] == 10000.0
    assert totals[ResourceType.MATERIALS.value] == 5000.0
    assert "PROJ-001" in finance_engine.budgets

def test_resource_usage(finance_engine, sample_allocations):
    finance_engine.allocate_budget("PROJ-001", sample_allocations)
    
    usage = ResourceUsage(
        resource_type=ResourceType.COMPUTE,
        amount_used=1000.0,
        timestamp=datetime.now(),
        user_id="USER-001",
        project_id="PROJ-001",
        efficiency_score=0.85
    )
    
    finance_engine.record_usage(usage)
    assert len(finance_engine.usage_history) == 1
    assert finance_engine.efficiency_metrics["PROJ-001"] == 0.85

def test_efficiency_analysis(finance_engine, sample_allocations):
    finance_engine.allocate_budget("PROJ-001", sample_allocations)
    
    # Record multiple usages
    for i in range(3):
        usage = ResourceUsage(
            resource_type=ResourceType.COMPUTE,
            amount_used=1000.0,
            timestamp=datetime.now(),
            user_id="USER-001",
            project_id="PROJ-001",
            efficiency_score=0.8 + i*0.1
        )
        finance_engine.record_usage(usage)
    
    analysis = finance_engine.analyze_efficiency("PROJ-001")
    assert "efficiency_score" in analysis
    assert "resource_distribution" in analysis
    assert "recommendations" in analysis 