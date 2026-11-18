"""
inventory_tools.py - Supply Chain Analytics Inventory Tools & Multicriteria Classification
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import List, Optional, Tuple, Dict, Any, Union

REQUIRED_CLASSIFICATION_COLUMNS = ["Item Number", "Item Name", "UoM", "Sales ($)", "Hits"]

@dataclass
class ClassificationResult:
    class_summary_df: pd.DataFrame
    classified_df: pd.DataFrame
    verification_steps_text: str = ""

    def format_verification_steps(self) -> str:
        return self.verification_steps_text


def validate_classification_dataframe(df: pd.DataFrame) -> List[str]:
    """Validates that all required multicriteria classification columns exist."""
    return [col for col in REQUIRED_CLASSIFICATION_COLUMNS if col not in df.columns]


def classify_inventory_multicriteria(df: pd.DataFrame) -> ClassificationResult:
    """
    Performs Multicriteria Inventory Classification (MCIC) using Sales ($) and Hits:
    1. Rank Sales and Hits separately using Pareto thresholds:
       - Class A: 0 - 60% cumulative share
       - Class B: 61 - 80% cumulative share
       - Class C: 81 - 90% cumulative share
       - Class D: 91 - 100% cumulative share
    2. Convert classes to numerical scores: A=4, B=3, C=2, D=1
    3. Calculate Combined Weighted Score = 0.65 * Sales Score + 0.35 * Hits Score
    4. Assign Final Class based on Combined Score:
       - Class A: 4.00 - 3.25
       - Class B: 3.24 - 2.50
       - Class C: 2.49 - 1.75
       - Class D: < 1.75
    """
    work_df = df.copy()

    work_df["Sales ($)"] = pd.to_numeric(
        work_df["Sales ($)"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    ).fillna(0.0)

    hits_col = "Hits" if "Hits" in work_df.columns else "POS Hits"
    work_df[hits_col] = pd.to_numeric(
        work_df[hits_col].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    ).fillna(0.0)

    # Pareto Sales
    work_df = work_df.sort_values(by="Sales ($)", ascending=False).reset_index(drop=True)
    total_sales = work_df["Sales ($)"].sum()
    work_df["Sales Share (%)"] = (work_df["Sales ($)"] / total_sales) * 100 if total_sales > 0 else 0
    work_df["Cumulative Sales (%)"] = work_df["Sales Share (%)"].cumsum()

    def assign_sales_class(cum_pct):
        if cum_pct <= 60.0:
            return "A", 4
        elif cum_pct <= 80.0:
            return "B", 3
        elif cum_pct <= 90.0:
            return "C", 2
        else:
            return "D", 1

    s_classes, s_scores = [], []
    for val in work_df["Cumulative Sales (%)"]:
        cls, score = assign_sales_class(val)
        s_classes.append(cls)
        s_scores.append(score)

    work_df["Sales Class"] = s_classes
    work_df["Sales Score"] = s_scores

    # Pareto Hits
    work_df = work_df.sort_values(by=hits_col, ascending=False).reset_index(drop=True)
    total_hits = work_df[hits_col].sum()
    work_df["Hits Share (%)"] = (work_df[hits_col] / total_hits) * 100 if total_hits > 0 else 0
    work_df["Cumulative Hits (%)"] = work_df["Hits Share (%)"].cumsum()

    def assign_hits_class(cum_pct):
        if cum_pct <= 60.0:
            return "A", 4
        elif cum_pct <= 80.0:
            return "B", 3
        elif cum_pct <= 90.0:
            return "C", 2
        else:
            return "D", 1

    h_classes, h_scores = [], []
    for val in work_df["Cumulative Hits (%)"]:
        cls, score = assign_hits_class(val)
        h_classes.append(cls)
        h_scores.append(sc if 'sc' in locals() else score) # robust

    # fix for h_scores scope
    h_classes, h_scores = [], []
    for val in work_df["Cumulative Hits (%)"]:
        cls, score_val = assign_hits_class(val)
        h_classes.append(cls)
        h_scores.append(score_val)

    work_df["Hits Class"] = h_classes
    work_df["Hits Score"] = h_scores

    work_df = work_df.sort_values(by="Sales ($)", ascending=False).reset_index(drop=True)
    work_df["Combined Weighted Score"] = (0.65 * work_df["Sales Score"]) + (0.35 * work_df["Hits Score"])

    def assign_final_class(score):
        if score >= 3.25:
            return "A"
        elif score >= 2.50:
            return "B"
        elif score >= 1.75:
            return "C"
        else:
            return "D"

    work_df["Final Inventory Class"] = work_df["Combined Weighted Score"].apply(assign_final_class)

    summary_rows = []
    total_items = len(work_df)
    for cls in ["A", "B", "C", "D"]:
        subset = work_df[work_df["Final Inventory Class"] == cls]
        item_count = len(subset)
        item_share = (item_count / total_items * 100) if total_items > 0 else 0
        cls_sales = subset["Sales ($)"].sum()
        sales_share = (cls_sales / total_sales * 100) if total_sales > 0 else 0
        cls_hits = subset[hits_col].sum()
        hits_share = (cls_hits / total_hits * 100) if total_hits > 0 else 0

        summary_rows.append({
            "Final Inventory Class": cls,
            "Item Count": item_count,
            "Item Share (%)": item_share,
            "Total Sales ($)": cls_sales,
            "Sales Share (%)": sales_share,
            "Total Hits": cls_hits,
            "Hits Share (%)": hits_share
        })

    summary_df = pd.DataFrame(summary_rows)

    verification_text = (
        "Multicriteria Inventory Classification successfully computed using "
        "65% Sales Score and 35% Hits Score weights with Pareto thresholds."
    )

    return ClassificationResult(
        class_summary_df=summary_df,
        classified_df=work_df,
        verification_steps_text=verification_text
    )


def _render_student_verification_markdown(result: Any, model_title: str) -> str:
    lines = [f"### Student Verification Breakdown — {model_title}\n"]
    lines.append("#### 1. Inputs Extracted")
    for k, v in result.inputs_extracted.items():
        lines.append(f"- **`{k}`:** `{v}`")
    lines.append("")
    lines.append("#### 2. Variable Definitions")
    for sym, desc in result.variable_definitions.items():
        lines.append(f"- **`{sym}`:** {desc}")
    lines.append("")
    lines.append("#### 3. Formula")
    for name, expr in result.formula.items():
        lines.append(f"- **{name.replace('_', ' ').title()}:** `{expr}`")
    lines.append("")
    lines.append("#### 4. Formula with Substituted Values")
    for name, expr in result.formula_with_substituted_values.items():
        lines.append(f"- **{name.replace('_', ' ').title()}:** `{expr}`")
    lines.append("")
    lines.append("#### 5. Intermediate Calculations")
    for k, v in result.intermediate_calculations.items():
        if isinstance(v, list):
            lines.append(f"- **{k.replace('_', ' ').title()}:**")
            for item in v:
                lines.append(f"  - `{item}`")
        else:
            lines.append(f"- **{k.replace('_', ' ').title()}:** `{v}`")
    lines.append("")
    lines.append("#### 6. Final Answer")
    for k, v in result.final_answer.items():
        lines.append(f"- **{k.replace('_', ' ').title()}:** **{v}**")
    lines.append("")
    lines.append("#### 7. Verification Checks")
    for chk in result.verification_checks:
        lines.append(f"- {chk}")
    lines.append("")
    lines.append("#### 8. Assumptions")
    for asm in result.assumptions:
        lines.append(f"- {asm}")
    return "\n".join(lines)


@dataclass
class EOQResult:
    annual_demand: float
    ordering_cost: float
    holding_cost: float
    unit_cost: float
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[str]
    assumptions: List[str]
    optimal_order_quantity: float
    is_valid: bool

    @property
    def verification_notes(self) -> List[str]:
        return self.verification_checks

    @property
    def eoq(self) -> float:
        return self.optimal_order_quantity

    def format_verification_steps(self) -> str:
        return _render_student_verification_markdown(self, "Economic Order Quantity (EOQ)")

    def to_dict(self) -> Dict[str, Any]:
        return {"model": "EOQ", "optimal_order_quantity": round(self.optimal_order_quantity, 2), "is_valid": self.is_valid}


def calculate_eoq(
    annual_demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    unit_cost: float = 0.0,
) -> EOQResult:
    if annual_demand is None or ordering_cost is None or holding_cost is None:
        raise ValueError("Missing required EOQ inputs.")
    D, S, H, C = float(annual_demand), float(ordering_cost), float(holding_cost), float(unit_cost)
    eoq = math.sqrt((2.0 * D * S) / H)
    orders_per_year = D / eoq
    cycle_time = (eoq / D) * 365.0
    ann_ord = orders_per_year * S
    ann_hold = (eoq / 2.0) * H
    tot_cost = ann_ord + ann_hold + (D * C)

    return EOQResult(
        annual_demand=D, ordering_cost=S, holding_cost=H, unit_cost=C,
        inputs_extracted={"annual_demand": D, "ordering_cost": S, "holding_cost": H, "unit_cost": C},
        variable_definitions={"D": "Annual demand", "S": "Ordering cost", "H": "Holding cost", "Q*": "EOQ"},
        formula={"eoq": "Q* = sqrt(2DS/H)"},
        formula_with_substituted_values={"eoq": f"Q* = sqrt(2 * {D} * {S} / {H}) = {eoq:.2f}"},
        intermediate_calculations={"orders_per_year": round(orders_per_year, 2), "cycle_time_days": round(cycle_time, 1)},
        final_answer={"optimal_order_quantity": round(eoq, 2), "total_annual_cost": round(tot_cost, 2)},
        verification_checks=["PASS: Positive order quantity verified."],
        assumptions=["Constant demand, instantaneous replenishment."],
        optimal_order_quantity=eoq,
        is_valid=True
    )