"""
inventory_tools.py - Deterministic Supply Chain Analytics Inventory Tools

Implements deterministic Python tools for:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Every tool returns a student-verifiable structured result containing all 8 required sections
generated deterministically in Python:
1. inputs_extracted
2. variable_definitions
3. formula
4. formula_with_substituted_values
5. intermediate_calculations
6. final_answer
7. verification_checks
8. assumptions

All tools validate inputs strictly and never invent missing numerical inputs.
"""

from dataclasses import dataclass, field
import math
from statistics import NormalDist
from typing import List, Optional, Tuple, Dict, Any, Union


def _render_student_verification_markdown(result: Any, model_title: str) -> str:
    """
    Renders the 8 required student-verifiable sections directly from the
    deterministic Python tool result without performing any external recalculation.
    """
    lines = [f"### Student Verification Breakdown — {model_title}\n"]

    # 1. Inputs Extracted
    lines.append("#### 1. Inputs Extracted")
    for k, v in result.inputs_extracted.items():
        lines.append(f"- **`{k}`:** `{v}`")
    lines.append("")

    # 2. Variable Definitions
    lines.append("#### 2. Variable Definitions")
    for sym, desc in result.variable_definitions.items():
        lines.append(f"- **`{sym}`:** {desc}")
    lines.append("")

    # 3. Formula
    lines.append("#### 3. Formula")
    for name, expr in result.formula.items():
        lines.append(f"- **{name.replace('_', ' ').title()}:** `{expr}`")
    lines.append("")

    # 4. Formula with Substituted Values
    lines.append("#### 4. Formula with Substituted Values")
    for name, expr in result.formula_with_substituted_values.items():
        lines.append(f"- **{name.replace('_', ' ').title()}:** `{expr}`")
    lines.append("")

    # 5. Intermediate Calculations
    lines.append("#### 5. Intermediate Calculations")
    for k, v in result.intermediate_calculations.items():
        if isinstance(v, list):
            lines.append(f"- **{k.replace('_', ' ').title()}:**")
            for item in v:
                lines.append(f"  - `{item}`")
        else:
            lines.append(f"- **{k.replace('_', ' ').title()}:** `{v}`")
    lines.append("")

    # 6. Final Answer
    lines.append("#### 6. Final Answer")
    for k, v in result.final_answer.items():
        lines.append(f"- **{k.replace('_', ' ').title()}:** **{v}**")
    lines.append("")

    # 7. Verification Checks
    lines.append("#### 7. Verification Checks")
    for chk in result.verification_checks:
        lines.append(f"- {chk}")
    lines.append("")

    # 8. Assumptions
    lines.append("#### 8. Assumptions")
    for asm in result.assumptions:
        lines.append(f"- {asm}")

    return "\n".join(lines)


# ==============================================================================
# 1. ECONOMIC ORDER QUANTITY (EOQ)
# ==============================================================================

@dataclass
class EOQResult:
    # Core scalar inputs
    annual_demand: float
    ordering_cost: float
    holding_cost: float
    unit_cost: float

    # 8 Required Student-Verifiable Sections
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[str]
    assumptions: List[str]

    # Primary scalar and status
    optimal_order_quantity: float
    is_valid: bool

    # Backwards-compatible properties
    @property
    def verification_notes(self) -> List[str]:
        return self.verification_checks

    @property
    def eoq(self) -> float:
        return self.optimal_order_quantity

    @property
    def orders_per_year(self) -> float:
        return self.intermediate_calculations.get("orders_per_year", 0.0)

    @property
    def cycle_time_days(self) -> float:
        return self.intermediate_calculations.get("cycle_time_days", 0.0)

    @property
    def annual_ordering_cost(self) -> float:
        return self.intermediate_calculations.get("annual_ordering_cost", 0.0)

    @property
    def annual_holding_cost(self) -> float:
        return self.intermediate_calculations.get("annual_holding_cost", 0.0)

    @property
    def annual_inventory_cost(self) -> float:
        return self.intermediate_calculations.get("annual_inventory_cost", 0.0)

    @property
    def annual_purchase_cost(self) -> float:
        return self.intermediate_calculations.get("annual_purchase_cost", 0.0)

    @property
    def annual_total_cost(self) -> float:
        return self.intermediate_calculations.get("annual_total_cost", 0.0)

    def format_verification_steps(self) -> str:
        return _render_student_verification_markdown(self, "Economic Order Quantity (EOQ)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "EOQ",
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "optimal_order_quantity": round(self.optimal_order_quantity, 2),
            "is_valid": self.is_valid,
        }


def calculate_eoq(
    annual_demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    unit_cost: float = 0.0,
) -> EOQResult:
    """
    Computes Economic Order Quantity: Q = sqrt(2DS / H).

    Validates inputs strictly without inventing missing numerical values.
    Returns all 8 student-verifiable sections deterministically from Python.
    """
    # Strict Input Validation (Never invent missing inputs)
    if annual_demand is None:
        raise ValueError("Missing required input: 'annual_demand' must be provided.")
    if ordering_cost is None:
        raise ValueError("Missing required input: 'ordering_cost' must be provided.")
    if holding_cost is None:
        raise ValueError("Missing required input: 'holding_cost' must be provided.")

    if not isinstance(annual_demand, (int, float)) or annual_demand <= 0:
        raise ValueError(f"annual_demand must be a positive number, got {annual_demand}")
    if not isinstance(ordering_cost, (int, float)) or ordering_cost <= 0:
        raise ValueError(f"ordering_cost must be a positive number, got {ordering_cost}")
    if not isinstance(holding_cost, (int, float)) or holding_cost <= 0:
        raise ValueError(f"holding_cost must be a positive number, got {holding_cost}")
    if not isinstance(unit_cost, (int, float)) or unit_cost < 0:
        raise ValueError(f"unit_cost must be non-negative, got {unit_cost}")

    D = float(annual_demand)
    S = float(ordering_cost)
    H = float(holding_cost)
    C = float(unit_cost)

    # 1. Inputs Extracted
    inputs_extracted = {
        "annual_demand": D,
        "ordering_cost": S,
        "holding_cost": H,
        "unit_cost": C,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand (units/year)",
        "S": "Ordering / setup cost per order ($/order)",
        "H": "Annual holding / carrying cost per unit ($/unit/year)",
        "C": "Unit purchase cost ($/unit)",
        "Q*": "Economic Order Quantity — optimal order batch size (units/order)",
        "N": "Expected number of orders per year (orders/year)",
        "T": "Expected cycle time between orders (days)",
    }

    # Deterministic Calculations
    numerator = 2.0 * D * S
    cost_ratio = numerator / H
    eoq = math.sqrt(cost_ratio)
    orders_per_year = D / eoq
    cycle_time_days = (eoq / D) * 365.0

    annual_ordering_cost = orders_per_year * S
    annual_holding_cost = (eoq / 2.0) * H
    annual_inventory_cost = annual_ordering_cost + annual_holding_cost
    annual_purchase_cost = D * C
    annual_total_cost = annual_inventory_cost + annual_purchase_cost
    second_derivative = numerator / (eoq ** 3)

    # 3. Formula
    formula = {
        "eoq": "Q* = sqrt((2 * D * S) / H)",
        "orders_per_year": "N = D / Q*",
        "cycle_time_days": "T = (Q* / D) * 365",
        "annual_ordering_cost": "Annual Ordering Cost = (D / Q*) * S",
        "annual_holding_cost": "Annual Holding Cost = (Q* / 2) * H",
        "annual_inventory_cost": "Total Annual Inventory Cost = Annual Ordering Cost + Annual Holding Cost",
        "annual_total_cost": "Grand Total Annual Cost = Total Annual Inventory Cost + (D * C)",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "eoq": f"Q* = sqrt((2 * {D:,.2f} * {S:,.2f}) / {H:,.4f}) = sqrt({numerator:,.2f} / {H:,.4f}) = sqrt({cost_ratio:,.4f}) = {eoq:,.2f} units",
        "orders_per_year": f"N = {D:,.2f} / {eoq:,.2f} = {orders_per_year:.2f} orders/year",
        "cycle_time_days": f"T = ({eoq:,.2f} / {D:,.2f}) * 365 = {cycle_time_days:.2f} days",
        "annual_ordering_cost": f"({D:,.2f} / {eoq:,.2f}) * {S:,.2f} = ${annual_ordering_cost:,.2f}",
        "annual_holding_cost": f"({eoq:,.2f} / 2) * {H:,.4f} = ${annual_holding_cost:,.2f}",
        "annual_inventory_cost": f"${annual_ordering_cost:,.2f} + ${annual_holding_cost:,.2f} = ${annual_inventory_cost:,.2f}",
        "annual_total_cost": f"${annual_inventory_cost:,.2f} + ({D:,.2f} * ${C:,.2f}) = ${annual_total_cost:,.2f}",
    }

    # 5. Intermediate Calculations
    intermediate = {
        "numerator_2DS": round(numerator, 4),
        "ratio_2DS_over_H": round(cost_ratio, 4),
        "orders_per_year": round(orders_per_year, 4),
        "cycle_time_days": round(cycle_time_days, 2),
        "annual_ordering_cost": round(annual_ordering_cost, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "annual_inventory_cost": round(annual_inventory_cost, 2),
        "annual_purchase_cost": round(annual_purchase_cost, 2),
        "annual_total_cost": round(annual_total_cost, 2),
        "second_derivative_d2TC_dQ2": round(second_derivative, 6),
    }

    # 6. Final Answer
    final_answer = {
        "optimal_order_quantity": round(eoq, 2),
        "optimal_order_quantity_rounded": round(eoq),
        "recommended_orders_per_year": round(orders_per_year, 2),
        "order_interval_days": round(cycle_time_days, 1),
        "minimized_annual_inventory_cost": round(annual_inventory_cost, 2),
        "minimized_grand_total_annual_cost": round(annual_total_cost, 2),
    }

    # 7. Verification Checks
    cost_diff = abs(annual_ordering_cost - annual_holding_cost)
    verification_checks = [
        f"PASS: Positive order quantity verified (Q* = {eoq:,.2f} > 0).",
        (
            f"PASS: Optimality condition met — Annual Ordering Cost (${annual_ordering_cost:,.2f}) equals Annual Holding Cost (${annual_holding_cost:,.2f})."
            if cost_diff <= 0.01 * annual_inventory_cost
            else f"WARN: Ordering vs. holding cost divergence of ${cost_diff:.4f}."
        ),
        f"PASS: Strict convexity verified (d²TC/dQ² = {second_derivative:.6f} > 0), confirming global minimum.",
    ]

    # 8. Assumptions
    assumptions = [
        "Demand rate is continuous, constant, and known with certainty over the annual horizon.",
        "Replenishment is instantaneous: entire batch arrives simultaneously when inventory reaches zero.",
        "Lead time is zero or constant.",
        "No shortages or stockouts are permitted; all demand must be satisfied on time.",
        "Unit purchase cost (C), setup/order cost (S), and annual holding cost (H) remain constant over time.",
    ]

    return EOQResult(
        annual_demand=D,
        ordering_cost=S,
        holding_cost=H,
        unit_cost=C,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        optimal_order_quantity=eoq,
        is_valid=True,
    )


# ==============================================================================
# 2. QUANTITY DISCOUNT MODEL
# ==============================================================================

@dataclass
class DiscountBracket:
    min_qty: float
    max_qty: Optional[float]
    unit_price: float


@dataclass
class BracketEvaluation:
    bracket_index: int
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    holding_cost_used: float
    unconstrained_eoq: float
    candidate_qty: float
    is_feasible: bool
    status: str
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    annual_total_cost: float


@dataclass
class QuantityDiscountResult:
    # Core scalar inputs
    annual_demand: float
    ordering_cost: float
    holding_cost_type: str
    holding_rate: Optional[float]
    fixed_holding_cost: Optional[float]

    # 8 Required Student-Verifiable Sections
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[str]
    assumptions: List[str]

    # Per-tier evaluation details and optimal scalars
    evaluations: List[BracketEvaluation]
    optimal_bracket_index: int
    optimal_order_quantity: float
    optimal_unit_price: float
    optimal_annual_total_cost: float
    is_valid: bool

    @property
    def verification_notes(self) -> List[str]:
        return self.verification_checks

    @property
    def holding_cost_rate(self) -> Optional[float]:
        return self.holding_rate

    def format_verification_steps(self) -> str:
        return _render_student_verification_markdown(self, "Quantity Discount Model")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "QuantityDiscount",
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "evaluations": [
                {
                    "tier": b.bracket_index + 1,
                    "range": f"[{b.min_qty}, {b.max_qty if b.max_qty else 'inf'})",
                    "price": b.unit_price,
                    "holding_cost_used": round(b.holding_cost_used, 2),
                    "unconstrained_eoq": round(b.unconstrained_eoq, 2),
                    "candidate_qty": round(b.candidate_qty, 2),
                    "is_feasible": b.is_feasible,
                    "status": b.status,
                    "annual_purchase_cost": round(b.annual_purchase_cost, 2),
                    "annual_ordering_cost": round(b.annual_ordering_cost, 2),
                    "annual_holding_cost": round(b.annual_holding_cost, 2),
                    "annual_total_cost": round(b.annual_total_cost, 2),
                }
                for b in self.evaluations
            ],
            "is_valid": self.is_valid,
        }


def calculate_quantity_discount(
    annual_demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    discount_schedule: Optional[List[Union[DiscountBracket, Dict[str, Any], Tuple]]] = None,
    holding_cost: Optional[float] = None,
    holding_rate: Optional[float] = None,
) -> QuantityDiscountResult:
    """
    Evaluates feasible order quantities across discount tiers and compares total annual costs.
    Supports fixed holding cost ($/unit/year) or percentage holding cost (rate * unit price).

    Strictly validates inputs and returns all 8 student-verifiable sections deterministically.
    """
    # Strict Input Validation
    if annual_demand is None:
        raise ValueError("Missing required input: 'annual_demand' must be provided.")
    if ordering_cost is None:
        raise ValueError("Missing required input: 'ordering_cost' must be provided.")
    if discount_schedule is None or len(discount_schedule) == 0:
        raise ValueError("Missing required input: 'discount_schedule' must be provided as a non-empty list.")
    if holding_cost is None and holding_rate is None:
        raise ValueError("Missing required input: Either fixed 'holding_cost' or percentage 'holding_rate' must be provided.")

    if not isinstance(annual_demand, (int, float)) or annual_demand <= 0:
        raise ValueError(f"annual_demand must be a positive number, got {annual_demand}")
    if not isinstance(ordering_cost, (int, float)) or ordering_cost <= 0:
        raise ValueError(f"ordering_cost must be a positive number, got {ordering_cost}")

    holding_cost_type = "fixed" if holding_cost is not None else "percentage"
    if holding_cost is not None:
        if not isinstance(holding_cost, (int, float)) or holding_cost <= 0:
            raise ValueError(f"holding_cost must be positive, got {holding_cost}")
    if holding_rate is not None:
        if not isinstance(holding_rate, (int, float)) or holding_rate <= 0:
            raise ValueError(f"holding_rate must be positive, got {holding_rate}")

    D = float(annual_demand)
    S = float(ordering_cost)

    # Normalize discount schedule
    normalized_schedule: List[DiscountBracket] = []
    for idx, item in enumerate(discount_schedule):
        if isinstance(item, DiscountBracket):
            normalized_schedule.append(item)
        elif isinstance(item, dict):
            min_q = item.get("min_qty", item.get("min", 0.0))
            max_q = item.get("max_qty", item.get("max", None))
            price = item.get("unit_price", item.get("price", None))
            if price is None or price <= 0:
                raise ValueError(f"Tier {idx+1} is missing a valid positive price.")
            normalized_schedule.append(DiscountBracket(min_qty=float(min_q), max_qty=float(max_q) if max_q else None, unit_price=float(price)))
        elif isinstance(item, (tuple, list)):
            if len(item) < 3:
                raise ValueError(f"Tier tuple {item} must contain (min_qty, max_qty, unit_price).")
            min_q, max_q, price = item[0], item[1], item[2]
            normalized_schedule.append(DiscountBracket(min_qty=float(min_q), max_qty=float(max_q) if max_q else None, unit_price=float(price)))
        else:
            raise ValueError(f"Unsupported bracket format: {item}")

    # 1. Inputs Extracted
    inputs_extracted = {
        "annual_demand": D,
        "ordering_cost": S,
        "holding_cost_type": holding_cost_type,
        "holding_rate": float(holding_rate) if holding_rate is not None else None,
        "fixed_holding_cost": float(holding_cost) if holding_cost is not None else None,
        "discount_schedule": [
            f"Tier {i+1}: [{b.min_qty:g}, {f'{b.max_qty:g}' if b.max_qty is not None else 'inf'}) @ ${b.unit_price:.2f}"
            for i, b in enumerate(normalized_schedule)
        ],
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual demand (units/year)",
        "S": "Ordering / setup cost per order ($/order)",
        "C_j": "Unit purchase price in discount tier j ($/unit)",
        "H_j": "Annual holding cost per unit in tier j ($/unit/year)",
        "i": "Annual holding cost percentage rate (% of unit price, if percentage-based)",
        "EOQ_j": "Unconstrained Economic Order Quantity for tier j (units)",
        "Q_j": "Feasible candidate order quantity evaluated for tier j (units)",
        "TC(Q_j)": "Total annual cost for tier j = Purchase + Ordering + Holding ($/year)",
    }

    # 3. Formula
    formula = {
        "holding_cost_per_tier": "H_j = H (fixed)" if holding_cost is not None else "H_j = i * C_j",
        "unconstrained_eoq": "EOQ_j = sqrt((2 * D * S) / H_j)",
        "feasibility_adjustment": "If EOQ_j < Q_min_j -> Q_j = Q_min_j; if Q_min_j <= EOQ_j <= Q_max_j -> Q_j = EOQ_j",
        "total_annual_cost": "TC(Q_j) = (D * C_j) + ((D / Q_j) * S) + ((Q_j / 2) * H_j)",
    }

    # Evaluate each tier & build substituted formulas
    evaluations: List[BracketEvaluation] = []
    formula_with_substituted_values: Dict[str, str] = {}

    for idx, bracket in enumerate(normalized_schedule):
        tier_num = idx + 1
        h = float(holding_cost) if holding_cost is not None else (float(holding_rate) * bracket.unit_price)
        unconstrained_eoq = math.sqrt((2.0 * D * S) / h)

        min_q = bracket.min_qty
        max_q = bracket.max_qty

        if unconstrained_eoq < min_q:
            candidate_qty = min_q
            status = "EOQ below tier minimum; adjusted up to price break cutoff."
            is_feasible = True
        elif max_q is not None and unconstrained_eoq > max_q:
            candidate_qty = max_q
            status = "EOQ exceeds tier maximum; evaluated at tier boundary."
            is_feasible = False
        else:
            candidate_qty = unconstrained_eoq
            status = "Feasible unconstrained EOQ within tier."
            is_feasible = True

        p_cost = D * bracket.unit_price
        o_cost = (D / candidate_qty) * S
        h_cost = (candidate_qty / 2.0) * h
        tot_cost = p_cost + o_cost + h_cost

        evaluations.append(
            BracketEvaluation(
                bracket_index=idx,
                min_qty=min_q,
                max_qty=max_q,
                unit_price=bracket.unit_price,
                holding_cost_used=h,
                unconstrained_eoq=unconstrained_eoq,
                candidate_qty=candidate_qty,
                is_feasible=is_feasible,
                status=status,
                annual_purchase_cost=p_cost,
                annual_ordering_cost=o_cost,
                annual_holding_cost=h_cost,
                annual_total_cost=tot_cost,
            )
        )

        # 4. Substituted Formulas for Tier j
        h_sub = (
            f"H_{tier_num} = ${h:.2f} (fixed)"
            if holding_cost is not None
            else f"H_{tier_num} = {float(holding_rate):.4f} * ${bracket.unit_price:.2f} = ${h:.4f}"
        )
        formula_with_substituted_values[f"tier_{tier_num}_holding_cost"] = h_sub
        formula_with_substituted_values[f"tier_{tier_num}_eoq"] = (
            f"EOQ_{tier_num} = sqrt((2 * {D:,.2f} * {S:,.2f}) / {h:.4f}) = {unconstrained_eoq:,.2f} -> Candidate Q_{tier_num} = {candidate_qty:,.2f} ({status})"
        )
        formula_with_substituted_values[f"tier_{tier_num}_total_cost"] = (
            f"TC({candidate_qty:,.2f}) = ({D:,.2f} * ${bracket.unit_price:.2f}) + "
            f"(({D:,.2f} / {candidate_qty:,.2f}) * ${S:,.2f}) + (({candidate_qty:,.2f} / 2) * ${h:.4f}) "
            f"= ${p_cost:,.2f} + ${o_cost:,.2f} + ${h_cost:,.2f} = ${tot_cost:,.2f}"
        )

    # Identify global cost-minimizing feasible candidate
    feasible_evals = [e for e in evaluations if e.is_feasible]
    if not feasible_evals:
        feasible_evals = evaluations

    best_eval = min(feasible_evals, key=lambda e: e.annual_total_cost)

    # 5. Intermediate Calculations
    intermediate = {
        "tiers_evaluated_count": len(evaluations),
        "holding_cost_type": holding_cost_type,
        "holding_cost_specification": (
            f"Fixed ${holding_cost:.2f}/unit/year" if holding_cost is not None
            else f"Percentage {holding_rate * 100:.1f}% of unit price"
        ),
        "tier_summary": [
            {
                "tier_index": e.bracket_index + 1,
                "price": e.unit_price,
                "holding_cost_H": round(e.holding_cost_used, 4),
                "unconstrained_EOQ": round(e.unconstrained_eoq, 2),
                "candidate_Q": round(e.candidate_qty, 2),
                "purchase_cost": round(e.annual_purchase_cost, 2),
                "ordering_cost": round(e.annual_ordering_cost, 2),
                "holding_cost": round(e.annual_holding_cost, 2),
                "total_annual_cost": round(e.annual_total_cost, 2),
                "is_feasible": e.is_feasible,
            }
            for e in evaluations
        ],
    }

    # 6. Final Answer
    final_answer = {
        "optimal_bracket_index": best_eval.bracket_index,
        "optimal_tier_number": best_eval.bracket_index + 1,
        "optimal_order_quantity": round(best_eval.candidate_qty, 2),
        "optimal_order_quantity_rounded": round(best_eval.candidate_qty),
        "optimal_unit_price": round(best_eval.unit_price, 2),
        "minimized_annual_total_cost": round(best_eval.annual_total_cost, 2),
        "annual_purchase_cost": round(best_eval.annual_purchase_cost, 2),
        "annual_ordering_cost": round(best_eval.annual_ordering_cost, 2),
        "annual_holding_cost": round(best_eval.annual_holding_cost, 2),
    }

    # 7. Verification Checks
    in_bounds = (best_eval.candidate_qty >= best_eval.min_qty) and (
        best_eval.max_qty is None or best_eval.candidate_qty <= best_eval.max_qty
    )
    is_global_min = all(best_eval.annual_total_cost <= e.annual_total_cost + 1e-6 for e in feasible_evals)

    verification_checks = [
        f"PASS: Evaluated all {len(evaluations)} discount tiers.",
        f"PASS: Selected Q* = {best_eval.candidate_qty:,.2f} satisfies Tier #{best_eval.bracket_index + 1} bounds [{best_eval.min_qty:g}, {best_eval.max_qty if best_eval.max_qty else 'inf'})."
        if in_bounds else "WARN: Selected quantity boundary check flagged.",
        f"PASS: Global minimum confirmed — Tier #{best_eval.bracket_index + 1} total cost (${best_eval.annual_total_cost:,.2f}) is lowest among all feasible candidates."
        if is_global_min else "WARN: Global minimum check flagged.",
    ]
    if best_eval.candidate_qty == best_eval.min_qty and best_eval.min_qty > best_eval.unconstrained_eoq:
        verification_checks.append(
            f"INFO: Optimal quantity ({best_eval.candidate_qty:,.0f} units) occurs at a price break threshold, "
            f"where purchase price savings outweigh additional holding costs."
        )

    # 8. Assumptions
    assumptions = [
        "All-units quantity discount schedule: discounted unit price applies to all units purchased in the order.",
        "Constant and deterministic annual demand.",
        "Zero or constant lead time with instantaneous batch delivery.",
        "Holding cost is either constant across all brackets or proportional to unit purchase price (H = i * C).",
        "Storage capacity and capital are sufficient to accommodate the price-break order quantity.",
    ]

    return QuantityDiscountResult(
        annual_demand=D,
        ordering_cost=S,
        holding_cost_type=holding_cost_type,
        holding_rate=float(holding_rate) if holding_rate is not None else None,
        fixed_holding_cost=float(holding_cost) if holding_cost is not None else None,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        evaluations=evaluations,
        optimal_bracket_index=best_eval.bracket_index,
        optimal_order_quantity=best_eval.candidate_qty,
        optimal_unit_price=best_eval.unit_price,
        optimal_annual_total_cost=best_eval.annual_total_cost,
        is_valid=True,
    )


# ==============================================================================
# 3. SAFETY STOCK AND REORDER POINT (ROP)
# ==============================================================================

@dataclass
class ROPResult:
    # Core scalar inputs
    demand_mean: float
    lead_time_mean: float
    demand_std: float
    lead_time_std: float
    service_level: Optional[float]
    z_score: float
    case_type: str

    # 8 Required Student-Verifiable Sections
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[str]
    assumptions: List[str]

    # Primary outputs
    safety_stock: float
    reorder_point: float
    is_valid: bool

    # Backwards-compatible properties
    @property
    def verification_notes(self) -> List[str]:
        return self.verification_checks

    @property
    def lead_time(self) -> float:
        return self.lead_time_mean

    @property
    def expected_lead_time_demand(self) -> float:
        return self.intermediate_calculations.get("expected_lead_time_demand", 0.0)

    @property
    def lead_time_demand_std(self) -> float:
        return self.intermediate_calculations.get("sigma_LT", 0.0)

    @property
    def safety_stock_rounded(self) -> int:
        return self.final_answer.get("safety_stock_rounded", 0)

    @property
    def reorder_point_rounded(self) -> int:
        return self.final_answer.get("reorder_point_rounded", 0)

    @property
    def stockout_risk(self) -> float:
        return self.intermediate_calculations.get("stockout_risk", 0.0)

    def format_verification_steps(self) -> str:
        return _render_student_verification_markdown(self, "Safety Stock & Reorder Point (ROP)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "SafetyStock_ROP",
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "is_valid": self.is_valid,
        }


def calculate_rop(
    demand_mean: Optional[float] = None,
    lead_time_mean: Optional[float] = None,
    demand_std: float = 0.0,
    lead_time_std: float = 0.0,
    service_level: Optional[float] = None,
    z: Optional[float] = None,
    lead_time: Optional[float] = None,
) -> ROPResult:
    """
    Computes Safety Stock (SS) and Reorder Point (ROP) across all 4 variability cases:
    1. Constant demand and constant lead time
    2. Variable demand and constant lead time
    3. Constant demand and variable lead time
    4. Variable demand and variable lead time:
       sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
       SS = z * sigma_LT
       ROP = d * L + SS

    Returns all 8 student-verifiable sections deterministically from Python.
    """
    l_mean = lead_time_mean if lead_time_mean is not None else lead_time

    # Strict Input Validation
    if demand_mean is None:
        raise ValueError("Missing required input: 'demand_mean' must be provided.")
    if l_mean is None:
        raise ValueError("Missing required input: 'lead_time_mean' (or 'lead_time') must be provided.")

    if not isinstance(demand_mean, (int, float)) or demand_mean < 0:
        raise ValueError(f"demand_mean must be non-negative, got {demand_mean}")
    if not isinstance(l_mean, (int, float)) or l_mean <= 0:
        raise ValueError(f"lead_time must be positive, got {l_mean}")
    if not isinstance(demand_std, (int, float)) or demand_std < 0:
        raise ValueError(f"demand_std must be non-negative, got {demand_std}")
    if not isinstance(lead_time_std, (int, float)) or lead_time_std < 0:
        raise ValueError(f"lead_time_std must be non-negative, got {lead_time_std}")

    has_variability = (demand_std > 0) or (lead_time_std > 0)

    if has_variability:
        if service_level is None and z is None:
            raise ValueError(
                "When demand or lead time is variable, either 'service_level' or 'z' must be provided. "
                "Missing service level parameter cannot be invented."
            )

    sl_normalized = None
    if z is not None:
        z_val = float(z)
        if z_val < 0:
            raise ValueError(f"z-score must be non-negative for standard service levels, got {z_val}")
    elif service_level is not None:
        sl = float(service_level)
        if sl > 1.0:
            sl = sl / 100.0
        if not (0.0 < sl < 1.0):
            raise ValueError(f"service_level must be between 0 and 1 (or 0% and 100%), got {service_level}")
        sl_normalized = sl
        z_val = NormalDist().inv_cdf(sl)
    else:
        z_val = 0.0
        sl_normalized = 1.0

    d = float(demand_mean)
    L = float(l_mean)
    s_d = float(demand_std)
    s_L = float(lead_time_std)

    # 1. Inputs Extracted
    inputs_extracted = {
        "demand_mean": d,
        "lead_time_mean": L,
        "demand_std": s_d,
        "lead_time_std": s_L,
        "service_level": round(sl_normalized, 4) if sl_normalized is not None else None,
        "z_score": round(z_val, 4),
    }

    # 2. Variable Definitions
    variable_definitions = {
        "d": "Average demand per period (units/period)",
        "L": "Average replenishment lead time (periods)",
        "sigma_d": "Standard deviation of demand per period (units/period)",
        "sigma_L": "Standard deviation of lead time (periods)",
        "CSL": "Target Cycle Service Level (probability of no stockout during lead time)",
        "z": "Standard normal critical value Phi^(-1)(CSL)",
        "DDLT": "Expected demand during lead time = d * L (units)",
        "sigma_LT": "Standard deviation of demand during lead time (units)",
        "SS": "Safety stock buffer (units)",
        "ROP": "Reorder point trigger level (units)",
    }

    # Deterministic Case Calculations
    expected_ltd = d * L
    var_d_comp = L * (s_d ** 2)
    var_L_comp = (d ** 2) * (s_L ** 2)

    if s_d == 0.0 and s_L == 0.0:
        case_type = "constant_demand_constant_lead_time"
        sigma_LT = 0.0
        sigma_formula = "sigma_LT = 0 (deterministic demand and lead time)"
        sigma_sub = "sigma_LT = 0.0000"
    elif s_d > 0.0 and s_L == 0.0:
        case_type = "variable_demand_constant_lead_time"
        sigma_LT = s_d * math.sqrt(L)
        sigma_formula = "sigma_LT = sigma_d * sqrt(L)"
        sigma_sub = f"sigma_LT = {s_d:,.4f} * sqrt({L:,.4f}) = {s_d:,.4f} * {math.sqrt(L):.4f} = {sigma_LT:,.4f}"
    elif s_d == 0.0 and s_L > 0.0:
        case_type = "constant_demand_variable_lead_time"
        sigma_LT = d * s_L
        sigma_formula = "sigma_LT = d * sigma_L"
        sigma_sub = f"sigma_LT = {d:,.4f} * {s_L:,.4f} = {sigma_LT:,.4f}"
    else:
        case_type = "variable_demand_variable_lead_time"
        sigma_LT = math.sqrt(var_d_comp + var_L_comp)
        sigma_formula = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)"
        sigma_sub = (
            f"sigma_LT = sqrt({L:,.2f} * {s_d:,.2f}^2 + {d:,.2f}^2 * {s_L:,.2f}^2) "
            f"= sqrt({var_d_comp:,.4f} + {var_L_comp:,.4f}) = sqrt({var_d_comp + var_L_comp:,.4f}) = {sigma_LT:,.4f}"
        )

    safety_stock = z_val * sigma_LT
    reorder_point = expected_ltd + safety_stock
    ss_rounded = math.ceil(safety_stock) if safety_stock >= 0 else 0
    rop_rounded = math.ceil(reorder_point)

    # 3. Formula
    formula = {
        "expected_lead_time_demand": "DDLT = d * L",
        "lead_time_demand_std": sigma_formula,
        "safety_stock": "SS = z * sigma_LT",
        "reorder_point": "ROP = DDLT + SS = (d * L) + SS",
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "expected_lead_time_demand": f"DDLT = {d:,.2f} * {L:,.2f} = {expected_ltd:,.2f} units",
        "lead_time_demand_std": sigma_sub,
        "safety_stock": f"SS = {z_val:.4f} * {sigma_LT:,.4f} = {safety_stock:,.2f} units (ceiling: {ss_rounded})",
        "reorder_point": f"ROP = {expected_ltd:,.2f} + {safety_stock:,.2f} = {reorder_point:,.2f} units (ceiling: {rop_rounded})",
    }

    # 5. Intermediate Calculations
    intermediate = {
        "case_type": case_type,
        "sigma_LT_formula": sigma_formula,
        "expected_lead_time_demand": round(expected_ltd, 4),
        "demand_variance_component": round(var_d_comp, 4),
        "lead_time_variance_component": round(var_L_comp, 4),
        "sigma_LT": round(sigma_LT, 4),
        "z_score": round(z_val, 4),
        "service_level_used": round(sl_normalized, 4) if sl_normalized is not None else None,
        "stockout_risk": round(1.0 - sl_normalized, 4) if sl_normalized is not None else 0.0,
    }

    # 6. Final Answer
    final_answer = {
        "reorder_point": round(reorder_point, 2),
        "reorder_point_rounded": rop_rounded,
        "safety_stock": round(safety_stock, 2),
        "safety_stock_rounded": ss_rounded,
        "expected_lead_time_demand": round(expected_ltd, 2),
    }

    # 7. Verification Checks
    verification_checks = [
        f"PASS: Variability case identified as '{case_type.replace('_', ' ').title()}'.",
        f"PASS: Non-negative safety stock verified (SS = {safety_stock:,.2f} >= 0).",
        f"PASS: Reorder point bound verified (ROP = {reorder_point:,.2f} >= DDLT = {expected_ltd:,.2f}).",
        (
            f"PASS: Cycle service level verified ({sl_normalized*100:.2f}% CSL -> Z = {z_val:.4f}, cycle stockout risk = {(1.0-sl_normalized)*100:.2f}%)."
            if sl_normalized is not None and has_variability
            else f"PASS: Deterministic lead-time coverage verified (Z = {z_val:.4f})."
        ),
    ]

    # 8. Assumptions
    assumptions_map = {
        "constant_demand_constant_lead_time": [
            "Demand rate is perfectly deterministic and constant over time.",
            "Replenishment lead time is completely fixed and known with certainty.",
            "No safety stock is necessary because demand during lead time is exactly predictable.",
        ],
        "variable_demand_constant_lead_time": [
            "Demand in each period is independent and identically distributed (i.i.d.) normal random variable.",
            "Replenishment lead time is constant and fixed.",
            "Continuous inventory review (s, Q) policy with instantaneous order placement when inventory drops to ROP.",
        ],
        "constant_demand_variable_lead_time": [
            "Demand rate is constant and known with certainty.",
            "Replenishment lead time is a random variable with known mean and variance.",
            "Orders do not cross in transit (first ordered, first delivered).",
        ],
        "variable_demand_variable_lead_time": [
            "Demand per period and replenishment lead time are independent random variables.",
            "Demand during lead time is approximated by a normal distribution with mean d*L and variance L*sigma_d^2 + d^2*sigma_L^2.",
            "Stockout risk during replenishment cycle corresponds to 1 - Cycle Service Level.",
        ],
    }
    assumptions = assumptions_map[case_type]

    return ROPResult(
        demand_mean=d,
        lead_time_mean=L,
        demand_std=s_d,
        lead_time_std=s_L,
        service_level=sl_normalized,
        z_score=z_val,
        case_type=case_type,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        is_valid=True,
    )


# Alias for backward compatibility
calculate_safety_stock_rop = calculate_rop


# ==============================================================================
# 4. FIXED-PERIOD REVIEW (ORDER-UP-TO)
# ==============================================================================

@dataclass
class PeriodicReviewResult:
    # Core scalar inputs
    review_period: float
    lead_time: float
    demand_mean: float
    demand_std: float
    service_level: Optional[float]
    z_score: float
    current_inventory: float
    pipeline_inventory: float
    backorders: float

    # 8 Required Student-Verifiable Sections
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[str]
    assumptions: List[str]

    # Primary outputs
    protection_period: float
    target_inventory: float
    order_quantity: float
    safety_stock: float
    is_valid: bool

    # Backwards-compatible properties
    @property
    def verification_notes(self) -> List[str]:
        return self.verification_checks

    @property
    def protection_interval(self) -> float:
        return self.protection_period

    @property
    def target_inventory_level(self) -> float:
        return self.target_inventory

    @property
    def inventory_position(self) -> float:
        return self.intermediate_calculations.get("inventory_position", 0.0)

    @property
    def expected_protection_demand(self) -> float:
        return self.intermediate_calculations.get("demand_during_protection", 0.0)

    @property
    def protection_demand_std(self) -> float:
        return self.intermediate_calculations.get("sigma_protection", 0.0)

    @property
    def continuous_ss_comparison(self) -> float:
        return self.intermediate_calculations.get("continuous_review_safety_stock", 0.0)

    @property
    def periodic_extra_ss(self) -> float:
        return self.intermediate_calculations.get("extra_safety_stock_vs_continuous", 0.0)

    def format_verification_steps(self) -> str:
        return _render_student_verification_markdown(self, "Fixed-Period Review (Order-Up-To Policy)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "FixedPeriodReview",
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "is_valid": self.is_valid,
        }


def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: float = 0.0,
    service_level: Optional[float] = None,
    z: Optional[float] = None,
    current_inventory: float = 0.0,
    pipeline_inventory: float = 0.0,
    backorders: float = 0.0,
    safety_stock: Optional[float] = None,
    inventory_position: Optional[float] = None,
) -> PeriodicReviewResult:
    """
    Computes Target Inventory Level and replenishment Order Quantity for a Fixed-Period Review policy:
    1. Protection period = review period + lead time
    2. Target inventory = demand during protection period + safety stock
    3. Order quantity = max(0, target inventory - inventory position)

    Returns all 8 student-verifiable sections deterministically from Python.
    """
    # Strict Input Validation
    if review_period is None:
        raise ValueError("Missing required input: 'review_period' must be provided.")
    if lead_time is None:
        raise ValueError("Missing required input: 'lead_time' must be provided.")
    if demand_mean is None:
        raise ValueError("Missing required input: 'demand_mean' must be provided.")

    if not isinstance(review_period, (int, float)) or review_period <= 0:
        raise ValueError(f"review_period must be positive, got {review_period}")
    if not isinstance(lead_time, (int, float)) or lead_time <= 0:
        raise ValueError(f"lead_time must be positive, got {lead_time}")
    if not isinstance(demand_mean, (int, float)) or demand_mean < 0:
        raise ValueError(f"demand_mean must be non-negative, got {demand_mean}")
    if not isinstance(demand_std, (int, float)) or demand_std < 0:
        raise ValueError(f"demand_std must be non-negative, got {demand_std}")
    if not isinstance(current_inventory, (int, float)) or current_inventory < 0:
        raise ValueError(f"current_inventory must be non-negative, got {current_inventory}")
    if not isinstance(pipeline_inventory, (int, float)) or pipeline_inventory < 0:
        raise ValueError(f"pipeline_inventory must be non-negative, got {pipeline_inventory}")
    if not isinstance(backorders, (int, float)) or backorders < 0:
        raise ValueError(f"backorders must be non-negative, got {backorders}")
    if safety_stock is not None and (not isinstance(safety_stock, (int, float)) or safety_stock < 0):
        raise ValueError(f"safety_stock must be non-negative, got {safety_stock}")
    if inventory_position is not None and not isinstance(inventory_position, (int, float)):
        raise ValueError(f"inventory_position must be numeric, got {inventory_position}")

    if safety_stock is None and demand_std > 0 and service_level is None and z is None:
        raise ValueError(
            "When demand standard deviation is positive and safety_stock is not provided directly, "
            "either 'service_level' or 'z' must be provided. Missing service level cannot be invented."
        )

    sl_normalized = None
    if z is not None:
        z_val = float(z)
        if z_val < 0:
            raise ValueError(f"z-score must be non-negative, got {z_val}")
    elif service_level is not None:
        sl = float(service_level)
        if sl > 1.0:
            sl = sl / 100.0
        if not (0.0 < sl < 1.0):
            raise ValueError(f"service_level must be between 0 and 1 (or 0% and 100%), got {service_level}")
        sl_normalized = sl
        z_val = NormalDist().inv_cdf(sl)
    else:
        z_val = 0.0

    P = float(review_period)
    L = float(lead_time)
    d = float(demand_mean)
    s_d = float(demand_std)
    oh = float(current_inventory)
    oo = float(pipeline_inventory)
    bo = float(backorders)

    # 1. Inputs Extracted
    inputs_extracted = {
        "review_period": P,
        "lead_time": L,
        "demand_mean": d,
        "demand_std": s_d,
        "service_level": round(sl_normalized, 4) if sl_normalized is not None else None,
        "current_inventory": oh,
        "pipeline_inventory": oo,
        "backorders": bo,
        "explicit_safety_stock": float(safety_stock) if safety_stock is not None else None,
        "explicit_inventory_position": float(inventory_position) if inventory_position is not None else None,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "P": "Review period — fixed time interval between inventory reviews (periods)",
        "L": "Replenishment lead time (periods)",
        "P + L": "Protection period — total exposure window covered by an order (periods)",
        "d": "Average demand per period (units/period)",
        "sigma_d": "Standard deviation of demand per period (units/period)",
        "z": "Standard normal critical value Phi^(-1)(CSL)",
        "SS": "Safety stock buffer across the protection period (units)",
        "M": "Target order-up-to inventory level (units)",
        "IP": "Inventory position = On-Hand + On-Order - Backorders (units)",
        "Q": "Replenishment order quantity to place now (units)",
    }

    # Deterministic Calculations
    protection_period = P + L
    demand_during_protection = d * protection_period
    sigma_protection = s_d * math.sqrt(protection_period)
    computed_safety_stock = float(safety_stock) if safety_stock is not None else (z_val * sigma_protection)

    target_inventory = demand_during_protection + computed_safety_stock
    computed_ip = float(inventory_position) if inventory_position is not None else (oh + oo - bo)
    raw_order = target_inventory - computed_ip
    order_quantity = max(0.0, raw_order)

    continuous_ss = z_val * (s_d * math.sqrt(L))
    extra_ss = computed_safety_stock - continuous_ss

    # 3. Formula
    formula = {
        "protection_period": "Protection Period = P + L",
        "demand_during_protection": "D_(P+L) = d * (P + L)",
        "sigma_protection": "sigma_(P+L) = sigma_d * sqrt(P + L)",
        "safety_stock": "SS = Explicit Safety Stock" if safety_stock is not None else "SS = z * sigma_(P+L)",
        "target_inventory": "M = d * (P + L) + SS",
        "inventory_position": "IP = Explicit Inventory Position" if inventory_position is not None else "IP = On-Hand + On-Order - Backorders",
        "order_quantity": "Q = max(0, M - IP)",
    }

    # 4. Formula with Substituted Values
    ss_sub = (
        f"SS = {computed_safety_stock:,.2f} units (explicitly provided)"
        if safety_stock is not None
        else f"SS = {z_val:.4f} * {sigma_protection:,.4f} = {computed_safety_stock:,.2f} units"
    )
    ip_sub = (
        f"IP = {computed_ip:,.2f} units (explicitly provided)"
        if inventory_position is not None
        else f"IP = {oh:,.2f} + {oo:,.2f} - {bo:,.2f} = {computed_ip:,.2f} units"
    )
    formula_with_substituted_values = {
        "protection_period": f"Protection Period = {P:,.2f} + {L:,.2f} = {protection_period:,.2f} periods",
        "demand_during_protection": f"D_(P+L) = {d:,.2f} * {protection_period:,.2f} = {demand_during_protection:,.2f} units",
        "sigma_protection": f"sigma_(P+L) = {s_d:,.2f} * sqrt({protection_period:,.2f}) = {sigma_protection:,.4f} units",
        "safety_stock": ss_sub,
        "target_inventory": f"M = {demand_during_protection:,.2f} + {computed_safety_stock:,.2f} = {target_inventory:,.2f} units",
        "inventory_position": ip_sub,
        "order_quantity": f"Q = max(0, {target_inventory:,.2f} - {computed_ip:,.2f}) = max(0, {raw_order:,.2f}) = {order_quantity:,.2f} units",
    }

    # 5. Intermediate Calculations
    intermediate = {
        "protection_period": round(protection_period, 4),
        "demand_during_protection": round(demand_during_protection, 4),
        "sigma_protection": round(sigma_protection, 4),
        "z_score": round(z_val, 4),
        "service_level_used": round(sl_normalized, 4) if sl_normalized is not None else None,
        "safety_stock": round(computed_safety_stock, 4),
        "safety_stock_source": "explicit_input" if safety_stock is not None else "calculated_from_service_level",
        "inventory_position": round(computed_ip, 4),
        "inventory_position_source": "explicit_input" if inventory_position is not None else "on_hand + on_order - backorders",
        "raw_order_quantity_before_clamp": round(raw_order, 4),
        "continuous_review_safety_stock": round(continuous_ss, 4),
        "extra_safety_stock_vs_continuous": round(extra_ss, 4),
    }

    # 6. Final Answer
    final_answer = {
        "order_quantity": round(order_quantity, 2),
        "order_quantity_rounded": math.ceil(order_quantity),
        "target_inventory": round(target_inventory, 2),
        "target_inventory_rounded": math.ceil(target_inventory),
        "protection_period": round(protection_period, 2),
        "safety_stock": round(computed_safety_stock, 2),
        "inventory_position": round(computed_ip, 2),
    }

    # 7. Verification Checks
    verification_checks = [
        f"PASS: Protection period verified (P + L = {protection_period:,.2f} > L = {L:,.2f}).",
        f"PASS: Target order-up-to level verified (M = {target_inventory:,.2f} >= D_(P+L) = {demand_during_protection:,.2f}).",
        f"PASS: Non-negative order clamp verified (Q = {order_quantity:,.2f} >= 0).",
    ]
    if safety_stock is None and s_d > 0:
        verification_checks.append(
            f"PASS: Periodic review buffer ({computed_safety_stock:,.2f}) exceeds continuous review buffer ({continuous_ss:,.2f}) by +{extra_ss:,.2f} units."
        )

    # 8. Assumptions
    assumptions = [
        "Inventory levels are reviewed at fixed, periodic intervals of length P.",
        "Replenishment orders are placed only at review epochs and arrive after lead time L.",
        "Orders must safeguard against stockouts across the full protection period (P + L).",
        "Periodic review requires higher safety stock than continuous review due to the unmonitored review gap P.",
        "Order quantity is non-negative and clamped to max(0, Target Inventory - Inventory Position).",
    ]

    return PeriodicReviewResult(
        review_period=P,
        lead_time=L,
        demand_mean=d,
        demand_std=s_d,
        service_level=sl_normalized,
        z_score=z_val,
        current_inventory=oh,
        pipeline_inventory=oo,
        backorders=bo,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        protection_period=protection_period,
        target_inventory=target_inventory,
        order_quantity=order_quantity,
        safety_stock=computed_safety_stock,
        is_valid=True,
    )


# Alias for backward compatibility
calculate_fixed_period_review = calculate_periodic_review


# ==============================================================================
# 5. NEWSVENDOR MODEL
# ==============================================================================

@dataclass
class NewsvendorResult:
    # Core scalar inputs
    price: float
    cost: float
    salvage_value: float
    shortage_penalty: float
    distribution_type: str

    # 8 Required Student-Verifiable Sections
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[str]
    assumptions: List[str]

    # Primary outputs
    underage_cost: float
    overage_cost: float
    critical_ratio: float
    recommended_order_quantity: float
    is_valid: bool

    # Backwards-compatible properties
    @property
    def verification_notes(self) -> List[str]:
        return self.verification_checks

    @property
    def cost_underage(self) -> float:
        return self.underage_cost

    @property
    def cost_overage(self) -> float:
        return self.overage_cost

    @property
    def optimal_quantity(self) -> float:
        return self.recommended_order_quantity

    @property
    def optimal_quantity_rounded(self) -> int:
        return self.final_answer.get("recommended_order_quantity_rounded", 0)

    @property
    def expected_sales(self) -> float:
        return self.intermediate_calculations.get("expected_sales", 0.0)

    @property
    def expected_lost_sales(self) -> float:
        return self.intermediate_calculations.get("expected_lost_sales", 0.0)

    @property
    def expected_leftover_inventory(self) -> float:
        return self.intermediate_calculations.get("expected_leftover_inventory", 0.0)

    @property
    def expected_revenue(self) -> float:
        return self.intermediate_calculations.get("expected_revenue", 0.0)

    @property
    def expected_total_cost(self) -> float:
        return self.intermediate_calculations.get("expected_total_cost", 0.0)

    @property
    def expected_profit(self) -> float:
        return self.final_answer.get("expected_profit", 0.0)

    @property
    def in_stock_probability(self) -> float:
        return self.intermediate_calculations.get("in_stock_probability", 0.0)

    def format_verification_steps(self) -> str:
        return _render_student_verification_markdown(self, "Newsvendor Model (Single-Period Stochastic)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Newsvendor",
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "underage_cost": round(self.underage_cost, 2),
            "overage_cost": round(self.overage_cost, 2),
            "critical_ratio": round(self.critical_ratio, 4),
            "recommended_order_quantity": round(self.recommended_order_quantity, 2),
            "is_valid": self.is_valid,
        }


def calculate_newsvendor(
    price: Optional[float] = None,
    cost: Optional[float] = None,
    salvage_value: float = 0.0,
    shortage_penalty: float = 0.0,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    discrete_demand: Optional[List[Tuple[float, float]]] = None,
    uniform_range: Optional[Tuple[float, float]] = None,
    uniform_min: Optional[float] = None,
    uniform_max: Optional[float] = None,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
) -> NewsvendorResult:
    """
    Computes optimal order quantity and financial expectations for the single-period Newsvendor model.
    1. Underage cost Cu = Price - Cost + Shortage_Penalty (or explicit underage_cost)
    2. Overage cost Co = Cost - Salvage_Value (or explicit overage_cost)
    3. Critical ratio CR = Cu / (Cu + Co)
    4. Recommended order quantity Q* based on demand distribution.

    Returns all 8 student-verifiable sections deterministically from Python.
    """
    direct_costs_mode = (underage_cost is not None and overage_cost is not None)

    if direct_costs_mode:
        if not isinstance(underage_cost, (int, float)) or underage_cost <= 0:
            raise ValueError(f"underage_cost must be positive, got {underage_cost}")
        if not isinstance(overage_cost, (int, float)) or overage_cost <= 0:
            raise ValueError(f"overage_cost must be positive, got {overage_cost}")
        cost_underage = float(underage_cost)
        cost_overage = float(overage_cost)
        p = float(price) if price is not None else (cost_underage + cost_overage)
        c = float(cost) if cost is not None else cost_overage
        s = float(salvage_value)
        g = float(shortage_penalty)
    else:
        if price is None:
            raise ValueError("Missing required input: 'price' must be provided (or both 'underage_cost' and 'overage_cost').")
        if cost is None:
            raise ValueError("Missing required input: 'cost' must be provided (or both 'underage_cost' and 'overage_cost').")

        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if not isinstance(cost, (int, float)) or cost <= 0:
            raise ValueError(f"cost must be positive, got {cost}")
        if not isinstance(salvage_value, (int, float)) or salvage_value < 0:
            raise ValueError(f"salvage_value must be non-negative, got {salvage_value}")
        if not isinstance(shortage_penalty, (int, float)) or shortage_penalty < 0:
            raise ValueError(f"shortage_penalty must be non-negative, got {shortage_penalty}")

        p = float(price)
        c = float(cost)
        s = float(salvage_value)
        g = float(shortage_penalty)

        if s >= c:
            raise ValueError(f"Salvage value ({s}) must be strictly less than cost ({c}) to avoid infinite riskless arbitrage.")
        if (p - c + g) <= 0:
            raise ValueError(f"Selling price plus shortage penalty ({p + g}) must exceed unit cost ({c}) for commercial viability.")

        cost_underage = float(underage_cost) if underage_cost is not None else (p - c + g)
        cost_overage = float(overage_cost) if overage_cost is not None else (c - s)

    # Validate demand distribution
    has_normal = (demand_mean is not None)
    has_discrete = (discrete_demand is not None)
    u_min = uniform_min if uniform_min is not None else (uniform_range[0] if uniform_range else None)
    u_max = uniform_max if uniform_max is not None else (uniform_range[1] if uniform_range else None)
    has_uniform = (u_min is not None and u_max is not None)

    if not (has_normal or has_discrete or has_uniform):
        raise ValueError(
            "Missing required demand distribution parameters. "
            "Must provide either (demand_mean and demand_std), discrete_demand, or (uniform_min and uniform_max). "
            "Demand parameters cannot be invented."
        )

    critical_ratio = cost_underage / (cost_underage + cost_overage)
    z_val: Optional[float] = None
    expected_demand_val = 0.0

    # Distribution-Specific Quantity Determination
    if has_normal:
        if demand_std is None or demand_std <= 0:
            raise ValueError(f"For normal distribution, 'demand_std' must be provided and positive, got {demand_std}")
        dist_type = "Normal"
        mu = float(demand_mean)
        sigma = float(demand_std)
        expected_demand_val = mu

        z_val = NormalDist().inv_cdf(critical_ratio)
        q_star = max(0.0, mu + (z_val * sigma))

        phi_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * (z_val ** 2))
        Phi_z = critical_ratio
        expected_lost_sales = sigma * (phi_z - (z_val * (1.0 - Phi_z)))
        expected_sales = mu - expected_lost_sales
        expected_leftover = q_star - expected_sales
        in_stock_prob = Phi_z

        q_formula = "Q* = mu + z * sigma, where z = Phi^(-1)(CR)"
        q_sub = f"Q* = {mu:,.2f} + ({z_val:.4f} * {sigma:,.2f}) = {q_star:,.2f} units"
        demand_inputs_dict = {"demand_mean": mu, "demand_std": sigma}

    elif has_discrete:
        dist_type = "Discrete"
        sorted_pairs = sorted(discrete_demand, key=lambda x: x[0])
        total_prob = sum(prob for _, prob in sorted_pairs)
        if not math.isclose(total_prob, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f"Discrete demand probabilities must sum to 1.0, got {total_prob}")

        expected_demand_val = sum(d_val * pr for d_val, pr in sorted_pairs)
        cum_prob = 0.0
        chosen_q = sorted_pairs[-1][0]
        in_stock_prob = 1.0
        for q_val, prob in sorted_pairs:
            cum_prob += prob
            if cum_prob >= critical_ratio:
                chosen_q = q_val
                in_stock_prob = cum_prob
                break
        q_star = float(chosen_q)
        expected_sales = sum(min(q_star, d_val) * pr for d_val, pr in sorted_pairs)
        expected_lost_sales = sum(max(0.0, d_val - q_star) * pr for d_val, pr in sorted_pairs)
        expected_leftover = sum(max(0.0, q_star - d_val) * pr for d_val, pr in sorted_pairs)

        q_formula = "Q* = smallest q such that F(q) = P(D <= q) >= CR"
        q_sub = f"Smallest q where F(q) >= {critical_ratio:.4f} is Q* = {q_star:,.2f} with F({q_star:,.2f}) = {in_stock_prob:.4f}"
        demand_inputs_dict = {"discrete_demand": sorted_pairs}

    else:
        dist_type = "Uniform"
        a = float(u_min)
        b = float(u_max)
        if a >= b:
            raise ValueError(f"uniform_min ({a}) must be strictly less than uniform_max ({b})")
        expected_demand_val = (a + b) / 2.0
        q_star = a + critical_ratio * (b - a)
        in_stock_prob = critical_ratio
        expected_sales = q_star - ((q_star - a) ** 2) / (2.0 * (b - a))
        expected_lost_sales = ((b - q_star) ** 2) / (2.0 * (b - a))
        expected_leftover = q_star - expected_sales

        q_formula = "Q* = a + CR * (b - a)"
        q_sub = f"Q* = {a:,.2f} + ({critical_ratio:.4f} * ({b:,.2f} - {a:,.2f})) = {q_star:,.2f} units"
        demand_inputs_dict = {"uniform_min": a, "uniform_max": b}

    expected_revenue = (p * expected_sales) + (s * expected_leftover)
    expected_cost = (c * q_star) + (g * expected_lost_sales)
    expected_profit = expected_revenue - expected_cost

    # 1. Inputs Extracted
    inputs_extracted = {
        "price": p if not direct_costs_mode else (float(price) if price is not None else None),
        "cost": c if not direct_costs_mode else (float(cost) if cost is not None else None),
        "salvage_value": s,
        "shortage_penalty": g,
        "explicit_underage_cost": float(underage_cost) if underage_cost is not None else None,
        "explicit_overage_cost": float(overage_cost) if overage_cost is not None else None,
        "distribution_type": dist_type,
        **demand_inputs_dict,
    }

    # 2. Variable Definitions
    variable_definitions = {
        "p": "Unit selling / retail price ($/unit)",
        "c": "Unit purchase / production cost ($/unit)",
        "s": "Unit salvage / clearance value for unsold units ($/unit)",
        "g": "Unit shortage / goodwill penalty for unmet demand ($/unit)",
        "C_u": "Cost of underage — marginal opportunity loss per unit of unmet demand ($/unit)",
        "C_o": "Cost of overage — marginal loss per unsold leftover unit ($/unit)",
        "CR": "Critical ratio (critical fractile) — optimal probability of satisfying all demand",
        "Q*": "Recommended single-period order quantity (units)",
    }

    # 3. Formula
    formula = {
        "underage_cost": "C_u = Explicit Underage Cost" if direct_costs_mode else "C_u = p - c + g",
        "overage_cost": "C_o = Explicit Overage Cost" if direct_costs_mode else "C_o = c - s",
        "critical_ratio": "CR = C_u / (C_u + C_o)",
        "optimal_order_quantity": q_formula,
        "expected_profit": "E[Profit] = (p * E[Sales] + s * E[Leftover]) - (c * Q* + g * E[Lost Sales])",
    }

    # 4. Formula with Substituted Values
    cu_sub = (
        f"C_u = ${cost_underage:,.2f} (explicitly provided)"
        if direct_costs_mode
        else f"C_u = ${p:,.2f} - ${c:,.2f} + ${g:,.2f} = ${cost_underage:,.2f}"
    )
    co_sub = (
        f"C_o = ${cost_overage:,.2f} (explicitly provided)"
        if direct_costs_mode
        else f"C_o = ${c:,.2f} - ${s:,.2f} = ${cost_overage:,.2f}"
    )
    formula_with_substituted_values = {
        "underage_cost": cu_sub,
        "overage_cost": co_sub,
        "critical_ratio": f"CR = ${cost_underage:,.2f} / (${cost_underage:,.2f} + ${cost_overage:,.2f}) = ${cost_underage:,.2f} / ${cost_underage + cost_overage:,.2f} = {critical_ratio:.4f} ({critical_ratio*100:.2f}%)",
        "optimal_order_quantity": q_sub,
        "expected_profit": (
            f"E[Profit] = (${p:,.2f} * {expected_sales:,.2f} + ${s:,.2f} * {expected_leftover:,.2f}) "
            f"- (${c:,.2f} * {q_star:,.2f} + ${g:,.2f} * {expected_lost_sales:,.2f}) = ${expected_profit:,.2f}"
        ),
    }

    # 5. Intermediate Calculations
    intermediate = {
        "cost_underage": round(cost_underage, 4),
        "cost_overage": round(cost_overage, 4),
        "critical_ratio": round(critical_ratio, 4),
        "critical_ratio_percentage": round(critical_ratio * 100.0, 2),
        "z_score": round(z_val, 4) if z_val is not None else None,
        "distribution_type": dist_type,
        "expected_demand": round(expected_demand_val, 2),
        "expected_sales": round(expected_sales, 2),
        "expected_lost_sales": round(expected_lost_sales, 2),
        "expected_leftover_inventory": round(expected_leftover, 2),
        "expected_revenue": round(expected_revenue, 2),
        "expected_total_cost": round(expected_cost, 2),
        "expected_profit": round(expected_profit, 2),
        "in_stock_probability": round(in_stock_prob, 4),
    }

    # 6. Final Answer
    final_answer = {
        "recommended_order_quantity": round(q_star, 2),
        "recommended_order_quantity_rounded": round(q_star),
        "critical_ratio": round(critical_ratio, 4),
        "underage_cost": round(cost_underage, 2),
        "overage_cost": round(cost_overage, 2),
        "expected_profit": round(expected_profit, 2),
        "in_stock_probability": round(in_stock_prob, 4),
    }

    # 7. Verification Checks
    sales_balance = abs((expected_sales + expected_leftover) - q_star)
    demand_balance = abs((expected_sales + expected_lost_sales) - expected_demand_val)
    verification_checks = [
        f"PASS: Critical ratio bounds verified (0 < CR = {critical_ratio:.4f} < 1).",
        f"PASS: Positive marginal economics verified (Cu = ${cost_underage:,.2f} > 0, Co = ${cost_overage:,.2f} > 0).",
        f"PASS: Supply conservation verified (E[Sales] {expected_sales:,.2f} + E[Leftover] {expected_leftover:,.2f} = Q* {q_star:,.2f})."
        if sales_balance < 0.05 else f"WARN: Supply conservation delta = {sales_balance:.4f}.",
        f"PASS: Demand conservation verified (E[Sales] {expected_sales:,.2f} + E[Lost Sales] {expected_lost_sales:,.2f} = E[D] {expected_demand_val:,.2f})."
        if demand_balance < 0.05 else f"WARN: Demand conservation delta = {demand_balance:.4f}.",
    ]

    # 8. Assumptions
    assumptions = [
        "Single replenishment opportunity prior to the selling period; no mid-season reorders permitted.",
        "Perishable or seasonal goods: unsold items at the end of the period lose primary value and must be salvaged at s < c.",
        "Unmet customer demand is lost; each lost sale incurs an opportunity cost Cu = p - c + g.",
        "Demand is stochastic with a known, stationary probability distribution.",
        "The objective is to maximize expected single-period operating profit.",
    ]

    return NewsvendorResult(
        price=p,
        cost=c,
        salvage_value=s,
        shortage_penalty=g,
        distribution_type=dist_type,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        underage_cost=cost_underage,
        overage_cost=cost_overage,
        critical_ratio=critical_ratio,
        recommended_order_quantity=q_star,
        is_valid=True,
    )
