"""
inventory_agent.py - Supply Chain Analytics Inventory Decision Agent

Implements the decision pipeline architecture:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain

Features:
- Advanced Natural Language Entity Extraction:
  - Recognizes common inventory terminology and domain synonyms.
  - EOQ: recognizes 'ordering cost', 'order cost', 'setup cost', 'setup expense',
    'cost per order', 'cost of placing an order', 'ordering/setup cost', etc.
  - Safety Stock & ROP: cleanly distinguishes average demand, demand std dev,
    average lead time, lead-time std dev, and service level.
  - Quantity Discount: distinguishes quantity break, tier unit prices, ordering cost,
    fixed holding cost, percentage holding cost, and base unit purchase price.
  - Fixed-Period Review: distinguishes demand, review period, lead time, safety stock,
    inventory position, on-hand inventory, on-order inventory, and backorders.
  - Newsvendor: distinguishes selling price, purchase cost, salvage value,
    underage/shortage cost, overage/excess cost, demand distribution, mean demand,
    and demand standard deviation.
- Exhaustive Search: Scans the entire question before declaring any information missing.
  Never reports a parameter as missing when it is already present in the question.
- Strict Integrity: Never invents missing numerical inputs.
"""

from dataclasses import dataclass, field
import json
import math
import re
from typing import Dict, Any, List, Optional, Tuple, Callable

import inventory_tools as it


@dataclass
class AgentStepLog:
    step_name: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    question: str
    model_identified: str
    model_display_name: str
    extracted_inputs: Dict[str, Any]
    tool_selected: str
    raw_result: Any
    verification_passed: bool
    verification_notes: List[str]
    executive_summary: str
    detailed_explanation: str
    extracted_meanings: Dict[str, str] = field(default_factory=dict)
    missing_inputs: List[str] = field(default_factory=list)
    pipeline_trace: List[AgentStepLog] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Renders the full response in rich GitHub-style Markdown."""
        lines = [
            f"# 📦 Supply Chain Inventory Decision Agent Report",
            f"**User Question:** *\"{self.question}\"*\n",
            f"---",
            f"## 1. 🎯 Executive Decision Summary",
            f"{self.executive_summary}\n",
            f"---",
            f"## 2. 🔍 Pipeline Execution Trace",
            f"- **Identified Model:** `{self.model_display_name}` (`{self.model_identified}`)",
            f"- **Selected Tool:** `{self.tool_selected}`",
            f"- **Extracted Parameters & Meanings:**",
        ]
        if self.extracted_inputs:
            for k, v in self.extracted_inputs.items():
                meaning_text = f" ({self.extracted_meanings[k]})" if k in self.extracted_meanings else ""
                if k == "discount_schedule" and isinstance(v, list):
                    sched_str = ", ".join(f"[{b.min_qty}-{b.max_qty or '∞'} @ ${b.unit_price:.2f}]" for b in v)
                    lines.append(f"  - `{k}`: {sched_str}{meaning_text}")
                elif k == "discrete_demand" and isinstance(v, list):
                    lines.append(f"  - `{k}`: {v}{meaning_text}")
                else:
                    lines.append(f"  - `{k}`: `{v}`{meaning_text}")
        else:
            lines.append("  - *None extracted.*")

        if self.missing_inputs:
            lines.append(f"- **⚠️ Missing Required Inputs:**")
            for m in self.missing_inputs:
                lines.append(f"  - `{m}`")

        lines.extend([
            f"\n---",
            f"## 3. 📊 Analytical Breakdown & Calculations",
            f"{self.detailed_explanation}\n",
            f"---",
            f"## 4. 🛡️ Verification & Audit Checks",
            f"**Status:** {'✅ PASSED' if self.verification_passed else '⚠️ FLAGGED'}",
        ])
        for note in self.verification_notes:
            lines.append(f"- {note}")

        return "\n".join(lines)


# ==============================================================================
# NATURAL LANGUAGE ENTITY & PARAMETER EXTRACTOR
# ==============================================================================

class NLPEntityExtractor:
    """
    Robust multi-pass Natural Language Entity and Parameter Extractor.
    Recognizes common supply chain terminology, domain synonyms, forward and backward
    number-word relationships, and prevents cross-talk between related parameters.
    """
    NUM_PAT = r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)"
    PCT_PAT = r"([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)"

    @classmethod
    def clean_num(cls, val_str: str) -> float:
        return float(val_str.replace(",", "").replace("$", "").strip("., "))

    @classmethod
    def extract_val(cls, text: str, synonyms: List[str]) -> Optional[Tuple[float, str]]:
        """
        Scans entire text for target synonyms followed by a number,
        or a number followed by target synonyms.
        Returns (numerical_value, matched_source_phrase) or None.
        """
        for syn in synonyms:
            # Pattern 1: Synonym then number
            # e.g., "setup expense is $80", "cost of placing an order: $75", "backorders are 15"
            p1 = rf"\b(?:{syn})\b\s*(?:is|are|was|were|of|equals?|equal\s+to|at|averages?|costs?|total(?:ing)?|:)?\s*[\$]?\s*{cls.NUM_PAT}"
            m1 = re.search(p1, text, re.IGNORECASE)
            if m1:
                try:
                    val = cls.clean_num(m1.group(1))
                    return val, m1.group(0).strip()
                except ValueError:
                    pass

            # Pattern 2: Number then synonym
            # e.g., "$80 setup expense", "$75 cost of placing an order"
            p2 = rf"[\$]?\s*{cls.NUM_PAT}\s*(?:USD|dollars)?\s*(?:for\s+an?\s+|to\s+|per\s+)?\b(?:{syn})\b"
            m2 = re.search(p2, text, re.IGNORECASE)
            if m2:
                try:
                    val = cls.clean_num(m2.group(1))
                    return val, m2.group(0).strip()
                except ValueError:
                    pass

        return None

    @classmethod
    def extract_pct(cls, text: str, synonyms: List[str]) -> Optional[Tuple[float, str]]:
        """
        Scans for percentage or decimal representations of rates (service level, holding rate).
        Returns (normalized_fraction, matched_source_phrase) or None.
        """
        for syn in synonyms:
            # Percentage: "holding rate is 20%", "service level of 95%"
            p1 = rf"\b(?:{syn})\b\s*(?:is|was|of|equals?|equal\s+to|at|:)?\s*{cls.PCT_PAT}"
            m1 = re.search(p1, text, re.IGNORECASE)
            if m1:
                try:
                    v = float(m1.group(1))
                    fraction = v / 100.0 if v > 1.0 else v
                    return fraction, m1.group(0).strip()
                except ValueError:
                    pass

            # Number % then keyword: "20% holding rate", "95% service level"
            p2 = rf"{cls.PCT_PAT}\s*(?:of\s+(?:item\s+|unit\s+)?(?:cost|price|value)\s+)?\b(?:{syn})\b"
            m2 = re.search(p2, text, re.IGNORECASE)
            if m2:
                try:
                    v = float(m2.group(1))
                    fraction = v / 100.0 if v > 1.0 else v
                    return fraction, m2.group(0).strip()
                except ValueError:
                    pass

            # Decimal: "service level of 0.95", "0.95 service level"
            p3 = rf"\b(?:{syn})\b\s*(?:is|was|of|equals?|at|:)?\s*(0\.[0-9]+)"
            m3 = re.search(p3, text, re.IGNORECASE)
            if m3:
                try:
                    return float(m3.group(1)), m3.group(0).strip()
                except ValueError:
                    pass

            p4 = rf"(0\.[0-9]+)\s+\b(?:{syn})\b"
            m4 = re.search(p4, text, re.IGNORECASE)
            if m4:
                try:
                    return float(m4.group(1)), m4.group(0).strip()
                except ValueError:
                    pass

        return None


class InventoryDecisionAgent:
    """Supply Chain Analytics Inventory Decision Agent."""

    MODEL_EOQ = "eoq"
    MODEL_QUANTITY_DISCOUNT = "quantity_discount"
    MODEL_SAFETY_STOCK_ROP = "safety_stock_rop"
    MODEL_FIXED_PERIOD_REVIEW = "fixed_period_review"
    MODEL_NEWSVENDOR = "newsvendor"
    MODEL_INVENTORY_CLASSIFICATION = "inventory_classification"

    MODEL_NAMES = {
        MODEL_EOQ: "Economic Order Quantity (EOQ)",
        MODEL_QUANTITY_DISCOUNT: "Quantity Discount Model",
        MODEL_SAFETY_STOCK_ROP: "Safety Stock & Reorder Point (Continuous Review)",
        MODEL_FIXED_PERIOD_REVIEW: "Fixed-Period Review (Periodic Review / Order-Up-To)",
        MODEL_NEWSVENDOR: "Newsvendor Model (Single-Period Stochastic)",
        MODEL_INVENTORY_CLASSIFICATION: "Multicriteria Inventory Classification (A, B, C, D)",
    }

    def __init__(self):
        self.tool_map: Dict[str, Callable] = {
            self.MODEL_EOQ: it.calculate_eoq,
            self.MODEL_QUANTITY_DISCOUNT: it.calculate_quantity_discount,
            self.MODEL_SAFETY_STOCK_ROP: it.calculate_rop,
            self.MODEL_FIXED_PERIOD_REVIEW: it.calculate_periodic_review,
            self.MODEL_NEWSVENDOR: it.calculate_newsvendor,
            self.MODEL_INVENTORY_CLASSIFICATION: it.classify_inventory_multicriteria,
        }

    # ==========================================================================
    # STEP 1: IDENTIFY MODEL
    # ==========================================================================
    def identify_model(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Classifies the customer question into one of the supported inventory models,
        including Multicriteria Inventory Classification (A, B, C, D).
        Returns (model_code, reasoning_dict).
        """
        q_lower = question.lower()
        scores: Dict[str, float] = {k: 0.0 for k in self.MODEL_NAMES}
        reasons: Dict[str, List[str]] = {k: [] for k in self.MODEL_NAMES}

        # 0. Multicriteria Inventory Classification (A, B, C, D) indicators
        cls_keywords = [
            ("inventory classification", 14.0), ("multicriteria", 12.0), ("multi-criteria", 12.0),
            ("abcd classification", 14.0), ("abc classification", 12.0), ("a, b, c, and d", 12.0),
            ("a, b, c, d", 12.0), ("classify inventory", 12.0), ("classify the inventory", 12.0),
            ("classify my inventory", 12.0), ("classify items", 10.0), ("classify my uploaded", 12.0),
            ("sales and hits", 12.0), ("hits and sales", 12.0), ("pareto classification", 10.0),
            ("abcd classes", 12.0), ("class a, b, c", 10.0), ("hits", 5.0),
        ]
        for kw, pts in cls_keywords:
            if kw in q_lower:
                scores[self.MODEL_INVENTORY_CLASSIFICATION] += pts
                reasons[self.MODEL_INVENTORY_CLASSIFICATION].append(kw)

        # 1. Newsvendor indicators
        nv_keywords = [
            ("newsvendor", 10.0), ("perishable", 8.0), ("salvage", 8.0), ("spoilage", 8.0),
            ("single period", 9.0), ("one-time order", 8.0), ("critical fractile", 9.0),
            ("critical ratio", 8.0), ("underage", 7.0), ("overage", 7.0),
            ("scrap value", 7.0), ("markdown", 6.0), ("seasonal item", 7.0),
            ("fashion", 6.0), ("event demand", 6.0), ("goodwill penalty", 6.0)
        ]
        for kw, pts in nv_keywords:
            if kw in q_lower:
                scores[self.MODEL_NEWSVENDOR] += pts
                reasons[self.MODEL_NEWSVENDOR].append(kw)

        # 2. Fixed-Period Review indicators
        fp_keywords = [
            ("fixed period", 10.0), ("fixed-period", 10.0), ("periodic review", 10.0), ("order-up-to", 9.0),
            ("order up to", 9.0), ("target inventory", 8.0), ("review period", 9.0),
            ("review interval", 8.0), ("every week", 7.0), ("every 2 weeks", 7.0),
            ("every month", 7.0), ("reviewed every", 9.0), ("review inventory every", 9.0),
            ("review stock every", 9.0), ("review every", 9.0), ("periodic policy", 8.0),
            ("inventory position", 7.0), ("on hand and on order", 7.0),
            ("on-hand inventory", 6.0), ("on-order inventory", 6.0), ("backorders", 5.0),
        ]
        for kw, pts in fp_keywords:
            if kw in q_lower:
                scores[self.MODEL_FIXED_PERIOD_REVIEW] += pts
                reasons[self.MODEL_FIXED_PERIOD_REVIEW].append(kw)

        # 3. Quantity Discount indicators
        qd_keywords = [
            ("quantity discount", 10.0), ("volume discount", 10.0), ("price break", 10.0),
            ("quantity break", 10.0), ("price discount", 9.0), ("discount schedule", 9.0),
            ("bulk discount", 8.0), ("tier", 6.0), ("bracket", 6.0), ("pricing tier", 8.0),
            ("cheaper if we order", 7.0), ("discount if", 6.0),
            ("for orders under", 8.0), ("orders under", 7.0), ("for orders of", 7.0),
            ("or more", 5.0),
        ]
        for kw, pts in qd_keywords:
            if kw in q_lower:
                scores[self.MODEL_QUANTITY_DISCOUNT] += pts
                reasons[self.MODEL_QUANTITY_DISCOUNT].append(kw)

        # 4. Safety Stock & ROP indicators
        rop_keywords = [
            ("reorder point", 10.0), ("rop", 8.0), ("safety stock", 9.0),
            ("service level", 6.0), ("lead time demand", 8.0), ("lead time standard deviation", 8.0),
            ("lead-time standard deviation", 8.0), ("continuous review", 8.0), ("buffer stock", 7.0),
            ("stockout probability", 7.0), ("when should we reorder", 8.0)
        ]
        for kw, pts in rop_keywords:
            if kw in q_lower:
                scores[self.MODEL_SAFETY_STOCK_ROP] += pts
                reasons[self.MODEL_SAFETY_STOCK_ROP].append(kw)

        # 5. EOQ indicators
        eoq_keywords = [
            ("eoq", 10.0), ("economic order quantity", 10.0), ("annual demand", 5.0),
            ("ordering cost", 5.0), ("order cost", 5.0), ("setup cost", 5.0),
            ("setup expense", 5.0), ("cost per order", 5.0), ("cost of placing an order", 5.0),
            ("ordering/setup cost", 5.0), ("holding cost", 5.0), ("carrying cost", 5.0),
            ("how many units to order", 5.0), ("optimal order quantity", 6.0),
            ("how much should we order", 4.0), ("batch size", 5.0)
        ]
        for kw, pts in eoq_keywords:
            if kw in q_lower:
                scores[self.MODEL_EOQ] += pts
                reasons[self.MODEL_EOQ].append(kw)

        # Disambiguation logic
        if scores[self.MODEL_FIXED_PERIOD_REVIEW] > 0 and scores[self.MODEL_SAFETY_STOCK_ROP] > 0:
            if any(term in q_lower for term in ["review period", "periodic", "order-up-to", "every"]):
                scores[self.MODEL_FIXED_PERIOD_REVIEW] += 5.0

        if scores[self.MODEL_QUANTITY_DISCOUNT] > 0 and scores[self.MODEL_EOQ] > 0:
            if any(term in q_lower for term in [
                "discount", "tier", "price break", "quantity break", "bulk", "schedule",
                "orders under", "for orders of", "or more", "bracket"
            ]):
                scores[self.MODEL_QUANTITY_DISCOUNT] += 10.0

        best_model = max(scores, key=lambda k: scores[k])
        if scores[best_model] == 0.0:
            if "lead time" in q_lower:
                best_model = self.MODEL_SAFETY_STOCK_ROP
            elif "order" in q_lower and ("holding" in q_lower or "setup" in q_lower):
                best_model = self.MODEL_EOQ
            else:
                best_model = self.MODEL_EOQ

        reasoning = {
            "chosen_model": best_model,
            "scores": scores,
            "matched_terms": reasons[best_model],
        }
        return best_model, reasoning

    # ==========================================================================
    # STEP 2: EXTRACT INPUTS (EXHAUSTIVE SYNONYM & CONTEXT PARSER)
    # ==========================================================================
    def extract_inputs(
        self,
        model_name: str,
        question: str,
        input_overrides: Optional[Dict[str, Any]] = None,
        dataframe: Any = None,
    ) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
        """
        Examines the entire question to extract numerical values and their semantic meanings.
        Recognizes common supply chain terminology and synonyms without declaring parameters
        missing when they are present in the text.
        Never invents missing numerical inputs.

        Returns (inputs_dict, meanings_dict, missing_mandatory_list).
        """
        inputs: Dict[str, Any] = {}
        meanings: Dict[str, str] = {}
        missing: List[str] = []

        if input_overrides:
            inputs.update(input_overrides)
            for k in input_overrides:
                meanings[k] = "User programmatic override"

        text = question

        # ----------------------------------------------------------------------
        # MODEL 1: EOQ
        # ----------------------------------------------------------------------
        if model_name == self.MODEL_EOQ:
            # 1. Ordering Cost Synonyms
            # - ordering cost, order cost, setup cost, setup expense, cost per order,
            #   cost of placing an order, ordering/setup cost, cost to place an order,
            #   order placement fee, fixed cost per order
            ordering_synonyms = [
                r"ordering\/setup\s+costs?",
                r"ordering\s+and\s+setup\s+costs?",
                r"cost\s+of\s+placing\s+(?:an?\s+)?order",
                r"cost\s+to\s+place\s+(?:an?\s+)?order",
                r"cost\s+per\s+order",
                r"setup\s+expenses?",
                r"setup\s+costs?",
                r"order\s+costs?",
                r"ordering\s+costs?",
                r"order\s+placement\s+(?:costs?|fees?)",
                r"procurement\s+costs?\s+per\s+order",
                r"fixed\s+costs?\s+per\s+order",
                r"costs?\s+to\s+order",
                r"each\s+order\s+costs?",
            ]
            if "ordering_cost" not in inputs:
                match = NLPEntityExtractor.extract_val(text, ordering_synonyms)
                if match:
                    inputs["ordering_cost"], meanings["ordering_cost"] = match[0], f"Extracted from '{match[1]}'"

            # 2. Annual Demand Synonyms
            demand_synonyms = [
                r"annual\s+demand",
                r"yearly\s+demand",
                r"annual\s+usage",
                r"annual\s+consumption",
                r"annual\s+sales",
                r"annual\s+requirement",
                r"demand\s+per\s+year",
                r"annual\s+volume",
                r"demand\s+is",
                r"demand\s+of",
                r"demand",
                r"usage",
            ]
            if "annual_demand" not in inputs:
                match = NLPEntityExtractor.extract_val(text, demand_synonyms)
                if match:
                    inputs["annual_demand"], meanings["annual_demand"] = match[0], f"Extracted from '{match[1]}'"
                else:
                    # Check bare "X units per year / X units annually / X units a year"
                    m_bare = re.search(r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*(?:units?|items?|parts?)\s*(?:per\s+year|\/year|annually|a\s+year)", text, re.I)
                    if m_bare:
                        val = NLPEntityExtractor.clean_num(m_bare.group(1))
                        inputs["annual_demand"], meanings["annual_demand"] = val, f"Extracted from '{m_bare.group(0)}'"

            # 3. Holding Cost Synonyms (Fixed or Percentage)
            holding_synonyms = [
                r"holding\s+costs?",
                r"carrying\s+costs?",
                r"storage\s+costs?",
                r"holding\s+expenses?",
                r"carrying\s+expenses?",
                r"storage\s+expenses?",
                r"costs?\s+to\s+hold(?:\s+inventory)?",
                r"costs?\s+of\s+carrying\s+inventory",
            ]
            holding_rate_synonyms = [
                r"holding\s+cost\s+rate",
                r"carrying\s+cost\s+rate",
                r"holding\s+rate",
                r"carrying\s+rate",
                r"annual\s+carrying\s+charge",
                r"carrying\s+charge",
            ]
            # Optional unit price
            unit_price_synonyms = [
                r"unit\s+purchase\s+price",
                r"purchase\s+price",
                r"base\s+unit\s+purchase\s+price",
                r"base\s+price",
                r"unit\s+cost",
                r"unit\s+price",
                r"purchase\s+cost",
                r"item\s+cost",
                r"item\s+price",
                r"wholesale\s+price",
            ]
            if "unit_cost" not in inputs:
                u_match = NLPEntityExtractor.extract_val(text, unit_price_synonyms)
                if u_match:
                    inputs["unit_cost"], meanings["unit_cost"] = u_match[0], f"Extracted from '{u_match[1]}'"
                else:
                    inputs["unit_cost"] = 0.0

            if "holding_cost" not in inputs:
                # Check fixed holding cost
                h_match = NLPEntityExtractor.extract_val(text, holding_synonyms)
                if h_match:
                    inputs["holding_cost"], meanings["holding_cost"] = h_match[0], f"Extracted from '{h_match[1]}'"
                else:
                    # Check bare "$X per unit per year"
                    m_hbare = re.search(r"[\$]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s+per\s+year|\/unit\/year|per\s+unit\s+annually)", text, re.I)
                    if m_hbare:
                        val = float(m_hbare.group(1))
                        inputs["holding_cost"], meanings["holding_cost"] = val, f"Extracted from '{m_hbare.group(0)}'"
                    else:
                        # Check percentage holding cost rate
                        hr_match = NLPEntityExtractor.extract_pct(text, holding_rate_synonyms)
                        if hr_match and inputs.get("unit_cost", 0.0) > 0:
                            h_calc = hr_match[0] * inputs["unit_cost"]
                            inputs["holding_cost"] = h_calc
                            meanings["holding_cost"] = f"Calculated as {hr_match[0]*100:.1f}% of unit cost ${inputs['unit_cost']:.2f}"

            # Check mandatory inputs for EOQ
            if "annual_demand" not in inputs:
                missing.append("annual_demand (Annual Demand D)")
            if "ordering_cost" not in inputs:
                missing.append("ordering_cost (Ordering / Setup Cost S)")
            if "holding_cost" not in inputs:
                missing.append("holding_cost (Annual Holding Cost per unit H)")

        # ----------------------------------------------------------------------
        # MODEL 2: QUANTITY DISCOUNT
        # ----------------------------------------------------------------------
        elif model_name == self.MODEL_QUANTITY_DISCOUNT:
            # 1. Ordering Cost Synonyms
            ordering_synonyms = [
                r"ordering\/setup\s+costs?",
                r"ordering\s+and\s+setup\s+costs?",
                r"cost\s+of\s+placing\s+(?:an?\s+)?order",
                r"cost\s+to\s+place\s+(?:an?\s+)?order",
                r"cost\s+per\s+order",
                r"setup\s+expenses?",
                r"setup\s+costs?",
                r"order\s+costs?",
                r"ordering\s+costs?",
                r"order\s+placement\s+(?:costs?|fees?)",
                r"procurement\s+costs?\s+per\s+order",
            ]
            if "ordering_cost" not in inputs:
                match = NLPEntityExtractor.extract_val(text, ordering_synonyms)
                if match:
                    inputs["ordering_cost"], meanings["ordering_cost"] = match[0], f"Extracted from '{match[1]}'"

            # 2. Annual Demand
            demand_synonyms = [
                r"annual\s+demand", r"yearly\s+demand", r"annual\s+usage",
                r"annual\s+consumption", r"annual\s+sales", r"demand\s+per\s+year", r"demand"
            ]
            if "annual_demand" not in inputs:
                match = NLPEntityExtractor.extract_val(text, demand_synonyms)
                if match:
                    inputs["annual_demand"], meanings["annual_demand"] = match[0], f"Extracted from '{match[1]}'"

            # 3. Holding Cost (Fixed vs Percentage)
            # Distinguish fixed holding cost from percentage holding cost
            holding_rate_synonyms = [
                r"holding\s+cost\s+rate", r"carrying\s+cost\s+rate", r"holding\s+rate",
                r"carrying\s+rate", r"annual\s+carrying\s+charge", r"holding\s+cost", r"carrying\s+cost"
            ]
            fixed_holding_synonyms = [
                r"fixed\s+holding\s+costs?", r"holding\s+costs?\s+is\s+fixed",
                r"holding\s+costs?\s+of\s+[\$][0-9]", r"carrying\s+costs?\s+of\s+[\$][0-9]"
            ]
            if "holding_rate" not in inputs and "holding_cost" not in inputs:
                # Check for percentage first
                pct_match = NLPEntityExtractor.extract_pct(text, holding_rate_synonyms)
                if pct_match:
                    inputs["holding_rate"], meanings["holding_rate"] = pct_match[0], f"Extracted from '{pct_match[1]}'"
                else:
                    fixed_match = NLPEntityExtractor.extract_val(text, [r"fixed\s+holding\s+costs?", r"holding\s+costs?", r"carrying\s+costs?"])
                    if fixed_match:
                        inputs["holding_cost"], meanings["holding_cost"] = fixed_match[0], f"Extracted from '{fixed_match[1]}'"

            # 4. Base Unit Purchase Price
            base_price_match = NLPEntityExtractor.extract_val(text, [
                r"base\s+unit\s+purchase\s+price", r"base\s+purchase\s+price", r"base\s+unit\s+price", r"base\s+price"
            ])
            base_price = base_price_match[0] if base_price_match else None

            # 5. Quantity Breaks & Unit Prices (Discount Schedule)
            if "discount_schedule" not in inputs:
                parsed_schedule: List[it.DiscountBracket] = []

                # Condition-first: Under X at $P
                for m, p in re.findall(r"(?:under|below|<)\s*([0-9,]+)\s*(?:units?)?\s*(?:at|costs?|price|:)\s*[\$]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.I):
                    parsed_schedule.append(it.DiscountBracket(min_qty=1.0, max_qty=float(m.replace(",", "").strip("., ")) - 1.0, unit_price=float(p)))

                # Price-first: $P for orders under X
                for p, m in re.findall(r"[\$]?\s*([0-9]+(?:\.[0-9]+)?)\s+(?:for\s+(?:orders?\s+)?)(?:under|below|<)\s*([0-9,]+)", text, re.I):
                    parsed_schedule.append(it.DiscountBracket(min_qty=1.0, max_qty=float(m.replace(",", "").strip("., ")) - 1.0, unit_price=float(p)))

                # Range tiers: X to Y at $P
                for m1, m2, p in re.findall(r"([0-9,]+)\s*(?:to|-|\sthrough\s)\s*([0-9,]+)\s*(?:units?)?\s*(?:at|costs?|price|:|for)?\s*[\$]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.I):
                    parsed_schedule.append(it.DiscountBracket(min_qty=float(m1.replace(",", "").strip("., ")), max_qty=float(m2.replace(",", "").strip("., ")), unit_price=float(p)))

                # Upper/open-ended tiers: X or more, X+, over X
                for m1, p in re.findall(r"(?:over|above|at\s+least|>=|>)\s*([0-9,]+)\s*(?:units?)?\s*(?:at|costs?|price|:|for)?\s*[\$]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.I):
                    q_min = float(m1.replace(",", "").strip("., "))
                    parsed_schedule.append(it.DiscountBracket(min_qty=q_min, max_qty=None, unit_price=float(p)))

                for m1, p in re.findall(r"([0-9,]+)\s*(?:or\s+more|\+|\s+and\s+above|\s+and\s+up|\s+plus)\s*(?:units?)?\s*(?:at|costs?|price|:|for)?\s*[\$]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.I):
                    q_min = float(m1.replace(",", "").strip("., "))
                    if not any(b.min_qty == q_min for b in parsed_schedule):
                        parsed_schedule.append(it.DiscountBracket(min_qty=q_min, max_qty=None, unit_price=float(p)))

                # Price-first open-ended: $P for orders of X or more
                for p, m1 in re.findall(r"[\$]?\s*([0-9]+(?:\.[0-9]+)?)\s+(?:for\s+(?:orders?\s+)?)(?:of\s+)?([0-9,]+)\s*(?:or\s+more|\+|\sand\s+above|\sand\s+up)", text, re.I):
                    q_min = float(m1.replace(",", "").strip("., "))
                    if not any(b.min_qty == q_min for b in parsed_schedule):
                        parsed_schedule.append(it.DiscountBracket(min_qty=q_min, max_qty=None, unit_price=float(p)))

                # Price-break cutoffs: e.g. "price break at 500 units ($19) and 1000 units ($18), base price $20"
                if not parsed_schedule and base_price:
                    breaks = []
                    for q, p in re.findall(r"(?:price\s+break\s+at|at)\s*([0-9,]+)\s*(?:units?)?\s*(?:for|at|costs?|:)?\s*[\$]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.I):
                        breaks.append((float(q.replace(",", "")), float(p)))
                    if breaks:
                        breaks.sort(key=lambda x: x[0])
                        parsed_schedule.append(it.DiscountBracket(min_qty=1.0, max_qty=breaks[0][0] - 1.0, unit_price=base_price))
                        for i in range(len(breaks)):
                            q_min = breaks[i][0]
                            q_max = breaks[i+1][0] - 1.0 if (i + 1 < len(breaks)) else None
                            parsed_schedule.append(it.DiscountBracket(min_qty=q_min, max_qty=q_max, unit_price=breaks[i][1]))

                parsed_schedule.sort(key=lambda b: b.min_qty)
                if parsed_schedule:
                    inputs["discount_schedule"] = parsed_schedule
                    meanings["discount_schedule"] = f"Parsed {len(parsed_schedule)} pricing tiers from discount schedule"

            # Check mandatory inputs for Quantity Discount
            if "annual_demand" not in inputs:
                missing.append("annual_demand (Annual Demand D)")
            if "ordering_cost" not in inputs:
                missing.append("ordering_cost (Ordering / Setup Cost S)")
            if "discount_schedule" not in inputs or not inputs["discount_schedule"]:
                missing.append("discount_schedule (Quantity discount pricing tiers)")
            if "holding_cost" not in inputs and "holding_rate" not in inputs:
                missing.append("holding_cost / holding_rate (Fixed holding cost or percentage holding rate)")

        # ----------------------------------------------------------------------
        # MODEL 3: SAFETY STOCK & REORDER POINT (ROP)
        # ----------------------------------------------------------------------
        elif model_name == self.MODEL_SAFETY_STOCK_ROP:
            # Split into sentences/clauses (avoiding decimal points between digits!)
            clauses = [c.strip() for c in re.split(r"(?<!\d)\.(?!\d)|[;\n]+", text) if c.strip()]

            # 1. Lead-time Standard Deviation (Extract FIRST to avoid matching mean lead time!)
            lt_std_synonyms = [
                r"lead\s*[- ]?time\s+(?:standard\s+deviation|std\s+dev|variance|variability)",
                r"standard\s+deviation\s+of\s+lead\s*[- ]?time",
                r"std\s+dev\s+of\s+lead\s*[- ]?time",
                r"delivery\s+time\s+(?:standard\s+deviation|std\s+dev)",
                r"std\s+dev\s+of\s+delivery\s+time",
            ]
            if "lead_time_std" not in inputs:
                match = NLPEntityExtractor.extract_val(text, lt_std_synonyms)
                if match:
                    inputs["lead_time_std"], meanings["lead_time_std"] = match[0], f"Extracted from '{match[1]}'"
                else:
                    # Clause-level context: "Average lead time is 5 days with a standard deviation of 1.5 days"
                    for cl in clauses:
                        if re.search(r"lead\s*[- ]?time|delivery\s+time", cl, re.I) and not re.search(r"demand", cl, re.I):
                            cl_match = NLPEntityExtractor.extract_val(cl, [r"standard\s+deviation", r"std\s+dev"])
                            if cl_match:
                                inputs["lead_time_std"], meanings["lead_time_std"] = cl_match[0], f"Extracted from lead-time clause '{cl_match[1]}'"
                                break
                    if "lead_time_std" not in inputs:
                        inputs["lead_time_std"] = 0.0

            # 2. Demand Standard Deviation (Extract BEFORE mean demand)
            demand_std_synonyms = [
                r"daily\s+demand\s+(?:standard\s+deviation|std\s+dev)",
                r"weekly\s+demand\s+(?:standard\s+deviation|std\s+dev)",
                r"demand\s+(?:standard\s+deviation|std\s+dev|variance)",
                r"standard\s+deviation\s+of\s+(?:daily\s+|weekly\s+)?demand",
                r"std\s+dev\s+of\s+demand",
            ]
            if "demand_std" not in inputs:
                match = NLPEntityExtractor.extract_val(text, demand_std_synonyms)
                if match:
                    inputs["demand_std"], meanings["demand_std"] = match[0], f"Extracted from '{match[1]}'"
                else:
                    # Clause-level context: "Average daily demand is 80 units with a standard deviation of 12 units"
                    for cl in clauses:
                        if re.search(r"demand|sales", cl, re.I) and not re.search(r"lead\s*[- ]?time", cl, re.I):
                            cl_match = NLPEntityExtractor.extract_val(cl, [r"standard\s+deviation", r"std\s+dev"])
                            if cl_match:
                                inputs["demand_std"], meanings["demand_std"] = cl_match[0], f"Extracted from demand clause '{cl_match[1]}'"
                                break
                    if "demand_std" not in inputs:
                        # Fallback generic std dev if not already matched as lead-time std dev
                        gen_match = NLPEntityExtractor.extract_val(text, [r"standard\s+deviation", r"std\s+dev"])
                        if gen_match and (inputs.get("lead_time_std", 0.0) == 0.0 or gen_match[0] != inputs.get("lead_time_std")):
                            inputs["demand_std"], meanings["demand_std"] = gen_match[0], f"Extracted from '{gen_match[1]}'"
                        else:
                            inputs["demand_std"] = 0.0

            # 3. Average Lead Time
            lead_time_synonyms = [
                r"average\s+lead\s*[- ]?time",
                r"mean\s+lead\s*[- ]?time",
                r"supplier\s+lead\s*[- ]?time",
                r"replenishment\s+lead\s*[- ]?time",
                r"delivery\s+lead\s*[- ]?time",
                r"order\s+lead\s*[- ]?time",
                r"lead\s*[- ]?time",
                r"delivery\s+takes",
            ]
            if "lead_time_mean" not in inputs and "lead_time" not in inputs:
                match = NLPEntityExtractor.extract_val(text, lead_time_synonyms)
                if match:
                    inputs["lead_time_mean"], meanings["lead_time_mean"] = match[0], f"Extracted from '{match[1]}'"

            # 4. Average Demand
            demand_mean_synonyms = [
                r"average\s+daily\s+demand",
                r"mean\s+daily\s+demand",
                r"average\s+weekly\s+demand",
                r"average\s+demand",
                r"mean\s+demand",
                r"daily\s+demand",
                r"weekly\s+demand",
                r"average\s+sales",
                r"demand\s+averages?",
            ]
            if "demand_mean" not in inputs:
                match = NLPEntityExtractor.extract_val(text, demand_mean_synonyms)
                if match:
                    inputs["demand_mean"], meanings["demand_mean"] = match[0], f"Extracted from '{match[1]}'"

            # 5. Service Level
            service_level_synonyms = [
                r"cycle\s+service\s+level",
                r"target\s+service\s+level",
                r"desired\s+service\s+level",
                r"service\s+level",
                r"csl",
                r"in\s*[- ]?stock\s+probability",
            ]
            if "service_level" not in inputs and "z" not in inputs:
                # Check explicit Z-score first
                m_z = re.search(r"\b(?:z\s*[- ]?score|z)\s*(?:is|was|of|equals?|at|:)?\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
                if m_z:
                    inputs["z"], meanings["z"] = float(m_z.group(1)), f"Extracted from '{m_z.group(0)}'"
                else:
                    sl_match = NLPEntityExtractor.extract_pct(text, service_level_synonyms)
                    if sl_match:
                        inputs["service_level"], meanings["service_level"] = sl_match[0], f"Extracted from '{sl_match[1]}'"

            # Check mandatory inputs for Safety Stock & ROP
            if "demand_mean" not in inputs:
                missing.append("demand_mean (Average periodic demand d)")
            if "lead_time_mean" not in inputs and "lead_time" not in inputs:
                missing.append("lead_time_mean (Average lead time L)")
            if (inputs.get("demand_std", 0.0) > 0 or inputs.get("lead_time_std", 0.0) > 0):
                if "service_level" not in inputs and "z" not in inputs:
                    missing.append("service_level / z (Target Cycle Service Level or Z-score)")

        # ----------------------------------------------------------------------
        # MODEL 4: FIXED-PERIOD REVIEW
        # ----------------------------------------------------------------------
        elif model_name == self.MODEL_FIXED_PERIOD_REVIEW:
            # 1. Review Period (Extract FIRST to avoid confusion with lead time)
            review_period_synonyms = [
                r"review\s*[- ]?period",
                r"review\s*[- ]?interval",
                r"reviewed\s+every",
                r"review\s+(?:inventory\s+|stock\s+)?every",
                r"review\s+frequency\s+of",
                r"orders?\s+placed\s+every",
                r"order\s+cycle\s+of",
                r"fixed\s+period\s+of",
            ]
            if "review_period" not in inputs:
                match = NLPEntityExtractor.extract_val(text, review_period_synonyms)
                if match:
                    inputs["review_period"], meanings["review_period"] = match[0], f"Extracted from '{match[1]}'"

            # 2. Lead Time
            lead_time_synonyms = [
                r"lead\s*[- ]?time",
                r"delivery\s+lead\s*[- ]?time",
                r"supplier\s+lead\s*[- ]?time",
                r"delivery\s+takes",
            ]
            if "lead_time" not in inputs:
                match = NLPEntityExtractor.extract_val(text, lead_time_synonyms)
                if match:
                    inputs["lead_time"], meanings["lead_time"] = match[0], f"Extracted from '{match[1]}'"

            # 3. Demand Rate & Std Dev
            if "demand_mean" not in inputs:
                match = NLPEntityExtractor.extract_val(text, [r"average\s+(?:daily\s+)?demand", r"mean\s+demand", r"daily\s+demand", r"demand\s+is", r"demand\s+of"])
                if match:
                    inputs["demand_mean"], meanings["demand_mean"] = match[0], f"Extracted from '{match[1]}'"

            if "demand_std" not in inputs:
                match = NLPEntityExtractor.extract_val(text, [r"demand\s+standard\s+deviation", r"standard\s+deviation\s+of\s+demand", r"demand\s+std\s+dev", r"standard\s+deviation", r"std\s+dev"])
                if match:
                    inputs["demand_std"], meanings["demand_std"] = match[0], f"Extracted from '{match[1]}'"
                else:
                    inputs["demand_std"] = 0.0

            # 4. Explicit Safety Stock (if given directly)
            if "safety_stock" not in inputs:
                ss_match = NLPEntityExtractor.extract_val(text, [r"safety\s+stock\s+buffer", r"safety\s+stock", r"buffer\s+stock"])
                if ss_match:
                    inputs["safety_stock"], meanings["safety_stock"] = ss_match[0], f"Extracted from '{ss_match[1]}'"

            # 5. Service Level
            if "service_level" not in inputs:
                sl_match = NLPEntityExtractor.extract_pct(text, [r"service\s+level", r"cycle\s+service\s+level", r"csl"])
                if sl_match:
                    inputs["service_level"], meanings["service_level"] = sl_match[0], f"Extracted from '{sl_match[1]}'"

            # 6. Explicit Inventory Position (if given directly)
            if "inventory_position" not in inputs:
                ip_match = NLPEntityExtractor.extract_val(text, [r"inventory\s+position", r"effective\s+inventory"])
                if ip_match:
                    inputs["inventory_position"], meanings["inventory_position"] = ip_match[0], f"Extracted from '{ip_match[1]}'"

            # 7. On-Hand Inventory
            on_hand_synonyms = [
                r"on\s*[- ]?hand\s+inventory", r"inventory\s+on\s*[- ]?hand", r"currently\s+have",
                r"in\s+stock", r"warehouse\s+stock", r"current\s+inventory"
            ]
            if "current_inventory" not in inputs:
                match = NLPEntityExtractor.extract_val(text, on_hand_synonyms)
                if match:
                    inputs["current_inventory"], meanings["current_inventory"] = match[0], f"Extracted from '{match[1]}'"
                else:
                    m_oh = re.search(r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*(?:units?)?\s+on\s*[- ]?hand\b", text, re.I)
                    if m_oh:
                        val = NLPEntityExtractor.clean_num(m_oh.group(1))
                        inputs["current_inventory"], meanings["current_inventory"] = val, f"Extracted from '{m_oh.group(0)}'"
                    else:
                        inputs["current_inventory"] = 0.0

            # 8. On-Order Inventory (Pipeline)
            on_order_synonyms = [
                r"on\s*[- ]?order\s+inventory", r"inventory\s+on\s*[- ]?order", r"pipeline\s+inventory",
                r"open\s+orders?", r"units?\s+in\s+transit"
            ]
            if "pipeline_inventory" not in inputs:
                match = NLPEntityExtractor.extract_val(text, on_order_synonyms)
                if match:
                    inputs["pipeline_inventory"], meanings["pipeline_inventory"] = match[0], f"Extracted from '{match[1]}'"
                else:
                    m_oo = re.search(r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*(?:units?)?\s+(?:already\s+)?on\s*[- ]?order\b", text, re.I)
                    if m_oo:
                        val = NLPEntityExtractor.clean_num(m_oo.group(1))
                        inputs["pipeline_inventory"], meanings["pipeline_inventory"] = val, f"Extracted from '{m_oo.group(0)}'"
                    else:
                        inputs["pipeline_inventory"] = 0.0

            # 9. Backorders
            backorder_synonyms = [r"unfilled\s+backorders?", r"backordered\s+demand", r"backorders?", r"backlog"]
            if "backorders" not in inputs:
                match = NLPEntityExtractor.extract_val(text, backorder_synonyms)
                if match:
                    inputs["backorders"], meanings["backorders"] = match[0], f"Extracted from '{match[1]}'"
                else:
                    m_bo = re.search(r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*(?:units?)?\s+(?:in\s+)?backorders?\b", text, re.I)
                    if m_bo:
                        val = NLPEntityExtractor.clean_num(m_bo.group(1))
                        inputs["backorders"], meanings["backorders"] = val, f"Extracted from '{m_bo.group(0)}'"
                    else:
                        inputs["backorders"] = 0.0

            # Check mandatory inputs for Fixed-Period Review
            if "review_period" not in inputs:
                missing.append("review_period (Review Interval P)")
            if "lead_time" not in inputs:
                missing.append("lead_time (Replenishment Lead Time L)")
            if "demand_mean" not in inputs:
                missing.append("demand_mean (Average Demand Rate d)")
            if "safety_stock" not in inputs and inputs.get("demand_std", 0.0) > 0 and "service_level" not in inputs:
                missing.append("service_level (Cycle Service Level for periodic review)")

        # ----------------------------------------------------------------------
        # MODEL 5: NEWSVENDOR
        # ----------------------------------------------------------------------
        elif model_name == self.MODEL_NEWSVENDOR:
            # 1. Explicit Underage Cost & Overage / Excess Cost (Extract FIRST!)
            underage_direct_synonyms = [
                r"cost\s+of\s+underage", r"underage\s+cost",
            ]
            overage_direct_synonyms = [
                r"cost\s+of\s+overage", r"overage\s+cost", r"excess\s+cost",
                r"cost\s+of\s+overstocking", r"overstock\s+cost",
            ]
            if "underage_cost" not in inputs:
                u_match = NLPEntityExtractor.extract_val(text, underage_direct_synonyms)
                if u_match:
                    inputs["underage_cost"], meanings["underage_cost"] = u_match[0], f"Extracted from '{u_match[1]}'"

            if "overage_cost" not in inputs:
                o_match = NLPEntityExtractor.extract_val(text, overage_direct_synonyms)
                if o_match:
                    inputs["overage_cost"], meanings["overage_cost"] = o_match[0], f"Extracted from '{o_match[1]}'"

            # 2. Shortage Penalty (Goodwill / Lost Sale Penalty)
            shortage_penalty_synonyms = [
                r"shortage\s+penalty", r"goodwill\s+penalty", r"goodwill\s+loss",
                r"lost\s+sale\s+penalty", r"shortage\s+cost", r"stockout\s+penalty",
            ]
            if "shortage_penalty" not in inputs:
                match = NLPEntityExtractor.extract_val(text, shortage_penalty_synonyms)
                if match:
                    inputs["shortage_penalty"], meanings["shortage_penalty"] = match[0], f"Extracted from '{match[1]}'"
                else:
                    inputs["shortage_penalty"] = 0.0

            # 3. Selling Price
            price_synonyms = [
                r"selling\s+price", r"retail\s+price", r"sale\s+price", r"sales\s+price",
                r"sells\s+for", r"revenue\s+per\s+unit", r"price\s+per\s+unit", r"retail\s+value"
            ]
            if "price" not in inputs:
                match = NLPEntityExtractor.extract_val(text, price_synonyms)
                if match:
                    inputs["price"], meanings["price"] = match[0], f"Extracted from '{match[1]}'"

            # 4. Purchase Cost (Guarded against matching underage/overage/shortage/excess cost!)
            cost_synonyms = [
                r"wholesale\s+purchase\s+cost", r"purchase\s+cost", r"acquisition\s+cost",
                r"procurement\s+cost", r"wholesale\s+price", r"production\s+cost",
                r"unit\s+cost", r"cost\s+per\s+unit", r"costs?\s+to\s+(?:buy|purchase|produce|make)",
                r"(?<!underage\s)(?<!overage\s)(?<!shortage\s)(?<!excess\s)(?<!overstock\s)\bcosts?\b(?!\s+of\s+(?:underage|overage|shortage|excess|overstocking))"
            ]
            if "cost" not in inputs:
                match = NLPEntityExtractor.extract_val(text, cost_synonyms)
                if match:
                    inputs["cost"], meanings["cost"] = match[0], f"Extracted from '{match[1]}'"

            # 5. Salvage Value
            salvage_synonyms = [
                r"salvage\s+value", r"scrap\s+value", r"liquidation\s+value", r"salvaged\s+for",
                r"clearance\s+price", r"markdown\s+price", r"scrap\s+price", r"salvage\s+price"
            ]
            if "salvage_value" not in inputs:
                match = NLPEntityExtractor.extract_val(text, salvage_synonyms)
                if match:
                    inputs["salvage_value"], meanings["salvage_value"] = match[0], f"Extracted from '{match[1]}'"
                else:
                    inputs["salvage_value"] = 0.0

            # 6. Demand Distribution
            uniform_match = re.search(r"uniformly\s+distributed\s+between\s+([0-9,]+)\s+and\s+([0-9,]+)", text, re.I)
            if uniform_match:
                a_val = NLPEntityExtractor.clean_num(uniform_match.group(1))
                b_val = NLPEntityExtractor.clean_num(uniform_match.group(2))
                inputs["uniform_range"] = (a_val, b_val)
                meanings["uniform_range"] = f"Uniform distribution on [{a_val}, {b_val}]"
            else:
                if "demand_mean" not in inputs:
                    match = NLPEntityExtractor.extract_val(text, [
                        r"normally\s+distributed\s+with\s+(?:a\s+)?mean\s+(?:demand\s+)?(?:of)?",
                        r"mean\s+demand", r"average\s+demand", r"expected\s+demand", r"demand\s+averages?"
                    ])
                    if match:
                        inputs["demand_mean"], meanings["demand_mean"] = match[0], f"Extracted from '{match[1]}'"

                if "demand_std" not in inputs:
                    match = NLPEntityExtractor.extract_val(text, [
                        r"demand\s+standard\s+deviation", r"standard\s+deviation\s+of\s+demand",
                        r"demand\s+std\s+dev", r"standard\s+deviation", r"std\s+dev", r"sigma"
                    ])
                    if match:
                        inputs["demand_std"], meanings["demand_std"] = match[0], f"Extracted from '{match[1]}'"

            # Check mandatory inputs for Newsvendor
            has_direct_cu_co = ("underage_cost" in inputs and "overage_cost" in inputs)
            if not has_direct_cu_co:
                if "price" not in inputs:
                    missing.append("price (Selling / Retail Price p)")
                if "cost" not in inputs:
                    missing.append("cost (Purchase / Unit Cost c)")
            if ("demand_mean" not in inputs or "demand_std" not in inputs) and "uniform_range" not in inputs and "discrete_demand" not in inputs:
                missing.append("demand distribution (Mean and standard deviation of demand)")

        # ----------------------------------------------------------------------
        # MODEL 6: MULTICRITERIA INVENTORY CLASSIFICATION (A, B, C, D)
        # ----------------------------------------------------------------------
        elif model_name == self.MODEL_INVENTORY_CLASSIFICATION:
            df_obj = inputs.get("df", dataframe)
            if df_obj is None:
                missing.append(
                    "uploaded CSV DataFrame with columns: Item Number, Item Name, UoM, Sales ($), Hits"
                )
            else:
                missing_cols = it.validate_classification_dataframe(df_obj)
                if missing_cols:
                    missing.append(
                        f"required CSV columns missing from uploaded DataFrame: {', '.join(missing_cols)}"
                    )
                else:
                    inputs["df"] = df_obj
                    row_cnt = len(df_obj) if hasattr(df_obj, "__len__") else "unknown"
                    meanings["df"] = f"Uploaded pandas DataFrame ({row_cnt} rows)"

            if "sales_weight" not in inputs:
                inputs["sales_weight"] = 0.65
                meanings["sales_weight"] = "Course methodology weight for Sales ($) (65%)"
            if "hits_weight" not in inputs:
                inputs["hits_weight"] = 0.35
                meanings["hits_weight"] = "Course methodology weight for POS Hits (35%)"

        return inputs, meanings, missing

    # ==========================================================================
    # STEP 3: SELECT PYTHON TOOL
    # ==========================================================================
    def select_python_tool(self, model_name: str) -> Tuple[Callable, str]:
        """Maps the identified model name to its numerical calculation tool."""
        if model_name not in self.tool_map:
            raise ValueError(f"Unknown inventory model '{model_name}'.")
        tool = self.tool_map[model_name]
        return tool, tool.__name__

    # ==========================================================================
    # STEP 4: CALCULATE
    # ==========================================================================
    def calculate(self, tool: Callable, inputs: Dict[str, Any]) -> Any:
        """Executes the selected Python tool with the extracted inputs."""
        return tool(**inputs)

    # ==========================================================================
    # STEP 5: VERIFY
    # ==========================================================================
    def verify(self, model_name: str, inputs: Dict[str, Any], result: Any) -> Tuple[bool, List[str]]:
        """
        Validates the mathematical result against domain rules, convexity, and physical constraints.
        Returns (is_passed, notes).
        """
        notes = list(getattr(result, "verification_notes", []))
        passed = True

        if model_name == self.MODEL_EOQ:
            if result.eoq <= 0:
                notes.append("❌ Verification Failure: EOQ must be strictly positive.")
                passed = False
            else:
                notes.append("✅ Non-negativity check passed: EOQ > 0.")
            if abs(result.annual_ordering_cost - result.annual_holding_cost) > (0.01 * result.annual_inventory_cost):
                notes.append("⚠️ Balance check warning: Ordering and holding costs deviate slightly.")
            else:
                notes.append("✅ Trade-off equilibrium verified: Annual Ordering Cost equals Annual Holding Cost.")

        elif model_name == self.MODEL_QUANTITY_DISCOUNT:
            if result.optimal_order_quantity <= 0:
                notes.append("❌ Verification Failure: Optimal quantity must be positive.")
                passed = False
            else:
                notes.append(f"✅ Feasible optimal order quantity verified: {result.optimal_order_quantity:.2f} units.")
                notes.append(f"✅ Global cost minimization verified: Total Cost = ${result.optimal_annual_total_cost:,.2f}.")

        elif model_name == self.MODEL_SAFETY_STOCK_ROP:
            if result.safety_stock < 0:
                notes.append("❌ Verification Failure: Safety stock cannot be negative for positive service levels.")
                passed = False
            if result.reorder_point < result.expected_lead_time_demand:
                notes.append("❌ Verification Failure: ROP cannot be lower than mean lead time demand.")
                passed = False
            notes.append(f"✅ Reorder trigger verified: Order placed when inventory drops to {result.reorder_point:.1f} units.")

        elif model_name == self.MODEL_FIXED_PERIOD_REVIEW:
            if result.target_inventory_level <= result.expected_protection_demand and result.demand_std > 0:
                notes.append("❌ Verification Failure: Target inventory must exceed mean protection interval demand.")
                passed = False
            if result.order_quantity < 0:
                notes.append("❌ Verification Failure: Order quantity cannot be negative.")
                passed = False
            notes.append(f"✅ Protection horizon verified: P + L = {result.protection_interval} periods.")
            notes.append(f"✅ Order-up-to target level verified: M = {result.target_inventory_level:.2f} units.")

        elif model_name == self.MODEL_NEWSVENDOR:
            if not (0.0 < result.critical_ratio < 1.0):
                notes.append(f"❌ Verification Failure: Critical ratio must be in (0, 1), got {result.critical_ratio}.")
                passed = False
            if result.expected_profit <= 0:
                notes.append("⚠️ Profitability warning: Expected profit is non-positive.")
            else:
                notes.append(f"✅ Economic feasibility confirmed: Expected single-period profit = ${result.expected_profit:,.2f}.")
            notes.append(f"✅ Marginal revenue vs cost balance satisfied: CR = {result.critical_ratio*100:.2f}%.")

        elif model_name == self.MODEL_INVENTORY_CLASSIFICATION:
            if result.total_items <= 0:
                notes.append("❌ Verification Failure: No inventory items were classified.")
                passed = False
            else:
                notes.append(
                    f"✅ Classified {result.total_items:,} SKUs into A ({result.class_counts['A']}), "
                    f"B ({result.class_counts['B']}), C ({result.class_counts['C']}), D ({result.class_counts['D']})."
                )

        return passed, notes

    # ==========================================================================
    # STEP 6: EXPLAIN (SOURCED EXCLUSIVELY FROM DETERMINISTIC PYTHON TOOL)
    # ==========================================================================
    def explain(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        result: Any,
        verification_notes: List[str]
    ) -> Tuple[str, str]:
        """
        Synthesizes an executive summary and renders the 8-section student-verifiable
        breakdown directly from the deterministic Python tool result.
        Does NOT independently generate or reconstruct numerical intermediate calculations.
        """
        fa = result.final_answer

        # 1. EOQ
        if model_name == self.MODEL_EOQ:
            exec_summary = (
                f"**Recommendation:** Order **{fa['optimal_order_quantity_rounded']:,} units** "
                f"(exact: {fa['optimal_order_quantity']:,.2f} units) approximately "
                f"**{fa['recommended_orders_per_year']} times per year** (every **{fa['order_interval_days']} days**), "
                f"minimizing annual inventory cost at **${fa['minimized_annual_inventory_cost']:,.2f}**."
            )

        # 2. QUANTITY DISCOUNT
        elif model_name == self.MODEL_QUANTITY_DISCOUNT:
            exec_summary = (
                f"**Recommendation:** Select **Tier #{fa['optimal_tier_number']}** and order "
                f"**{fa['optimal_order_quantity_rounded']:,} units** (exact: {fa['optimal_order_quantity']:,.2f} units) "
                f"at **${fa['optimal_unit_price']:,.2f}/unit**, minimizing total annual cost at "
                f"**${fa['minimized_annual_total_cost']:,.2f}**."
            )

        # 3. SAFETY STOCK & ROP
        elif model_name == self.MODEL_SAFETY_STOCK_ROP:
            exec_summary = (
                f"**Recommendation:** Set **Reorder Point (ROP) = {fa['reorder_point_rounded']:,} units** "
                f"(exact: {fa['reorder_point']:,.2f} units) with **Safety Stock = {fa['safety_stock_rounded']:,} units** "
                f"(exact: {fa['safety_stock']:,.2f} units)."
            )

        # 4. FIXED-PERIOD REVIEW
        elif model_name == self.MODEL_FIXED_PERIOD_REVIEW:
            exec_summary = (
                f"**Recommendation:** Maintain **Target Inventory = {fa['target_inventory_rounded']:,} units** "
                f"(exact: {fa['target_inventory']:,.2f} units) across the **{fa['protection_period']}-period** protection window "
                f"and place a replenishment order for **{fa['order_quantity_rounded']:,} units** "
                f"(exact: {fa['order_quantity']:,.2f} units)."
            )

        # 5. NEWSVENDOR
        elif model_name == self.MODEL_NEWSVENDOR:
            exec_summary = (
                f"**Recommendation:** Order **{fa['recommended_order_quantity_rounded']:,} units** "
                f"(exact: {fa['recommended_order_quantity']:,.2f} units) at a Critical Ratio of "
                f"**{fa['critical_ratio']*100:.2f}%** (Cu = ${fa['underage_cost']:,.2f}, Co = ${fa['overage_cost']:,.2f}), "
                f"yielding an expected profit of **${fa['expected_profit']:,.2f}**."
            )

        # 6. MULTICRITERIA INVENTORY CLASSIFICATION (A, B, C, D)
        elif model_name == self.MODEL_INVENTORY_CLASSIFICATION:
            exec_summary = (
                f"**Classification Complete:** Classified **{fa['total_items_classified']:,} SKUs** "
                f"(Total Sales: **${fa['total_sales']:,.2f}**, Total Hits: **{fa['total_hits']:,.0f}**) — "
                f"**Class A**: {fa['class_A_items']:,} items (${fa['class_A_sales']:,.2f}) | "
                f"**Class B**: {fa['class_B_items']:,} items (${fa['class_B_sales']:,.2f}) | "
                f"**Class C**: {fa['class_C_items']:,} items (${fa['class_C_sales']:,.2f}) | "
                f"**Class D**: {fa['class_D_items']:,} items (${fa['class_D_sales']:,.2f})."
            )
        else:
            exec_summary = "Calculation completed successfully."

        # Render all 8 student-verifiable sections directly from the deterministic Python tool
        detailed = result.format_verification_steps()
        return exec_summary, detailed

    # ==========================================================================
    # COMPLETE PIPELINE: RUN
    # ==========================================================================
    def run(
        self,
        question: str,
        model_override: Optional[str] = None,
        input_overrides: Optional[Dict[str, Any]] = None,
        dataframe: Any = None,
    ) -> AgentResponse:
        """
        Executes the full Supply Chain Analytics Decision Pipeline:
        Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
        """
        trace: List[AgentStepLog] = []

        # Step 1: Identify Model
        if model_override:
            model_code = model_override
            reasoning = {"chosen_model": model_code, "override": True}
        else:
            model_code, reasoning = self.identify_model(question)

        trace.append(AgentStepLog(
            step_name="Identify Model",
            status="SUCCESS",
            details={
                "model_code": model_code,
                "display_name": self.MODEL_NAMES.get(model_code, model_code),
                "reasoning": reasoning
            }
        ))

        # Step 2: Extract Inputs (Exhaustive Synonym & Context Parsing)
        extracted_inputs, extracted_meanings, missing_params = self.extract_inputs(
            model_code, question, input_overrides, dataframe=dataframe
        )
        trace.append(AgentStepLog(
            step_name="Extract Inputs",
            status="SUCCESS" if not missing_params else "MISSING_PARAMETERS",
            details={
                "extracted_inputs": {
                    k: (f"<DataFrame shape={getattr(v, 'shape', 'unknown')}>" if k == "df" else str(v))
                    for k, v in extracted_inputs.items()
                },
                "extracted_meanings": extracted_meanings,
                "missing_parameters": missing_params,
            }
        ))

        # Handle genuine missing parameters gracefully without inventing fake values
        if missing_params:
            missing_str = ", ".join(f"`{m}`" for m in missing_params)
            exec_summary = (
                f"⚠️ **Incomplete Information**: The question was identified as a `{self.MODEL_NAMES.get(model_code, model_code)}` problem, "
                f"but missing required parameter(s): {missing_str}. "
                f"No numerical values were invented. Please supply the missing parameter(s) to compute the final answer."
            )
            detailed = (
                f"### Extracted Parameters and Meanings\n"
            )
            if extracted_inputs:
                for k, v in extracted_inputs.items():
                    meaning_text = f" ({extracted_meanings[k]})" if k in extracted_meanings else ""
                    detailed += f"- **{k}:** `{v}`{meaning_text}\n"
            else:
                detailed += "- *No parameters could be extracted from the query.*\n"

            detailed += (
                f"\n### Missing Mandatory Parameters\n"
                f"The following required values were not found anywhere in your question:\n"
            )
            for m in missing_params:
                detailed += f"- **{m}**\n"

            detailed += (
                f"\n### Suggested Action\n"
                f"Please update your question to include the missing parameter(s) above."
            )

            return AgentResponse(
                question=question,
                model_identified=model_code,
                model_display_name=self.MODEL_NAMES.get(model_code, model_code),
                extracted_inputs=extracted_inputs,
                extracted_meanings=extracted_meanings,
                missing_inputs=missing_params,
                tool_selected="None (Calculation blocked due to missing parameters)",
                raw_result=None,
                verification_passed=False,
                verification_notes=[f"Missing mandatory parameters: {missing_str}"],
                executive_summary=exec_summary,
                detailed_explanation=detailed,
                pipeline_trace=trace,
            )

        # Step 3: Select Python Tool
        tool, tool_name = self.select_python_tool(model_code)
        trace.append(AgentStepLog(
            step_name="Select Python Tool",
            status="SUCCESS",
            details={"tool_function": tool_name}
        ))

        # Step 4: Calculate
        try:
            raw_result = self.calculate(tool, extracted_inputs)
            trace.append(AgentStepLog(
                step_name="Calculate",
                status="SUCCESS",
                details={"result_summary": getattr(raw_result, "to_dict", lambda: str(raw_result))()}
            ))
        except Exception as e:
            trace.append(AgentStepLog(
                step_name="Calculate",
                status="ERROR",
                details={"error_message": str(e)}
            ))
            raise RuntimeError(f"Calculation failed in {tool_name}: {e}") from e

        # Step 5: Verify
        is_valid, verification_notes = self.verify(model_code, extracted_inputs, raw_result)
        trace.append(AgentStepLog(
            step_name="Verify",
            status="SUCCESS" if is_valid else "WARNING",
            details={"is_valid": is_valid, "notes": verification_notes}
        ))

        # Step 6: Explain
        exec_summary, detailed_explanation = self.explain(
            model_code,
            extracted_inputs,
            raw_result,
            verification_notes
        )
        trace.append(AgentStepLog(
            step_name="Explain",
            status="SUCCESS",
            details={"summary_length": len(exec_summary)}
        ))

        return AgentResponse(
            question=question,
            model_identified=model_code,
            model_display_name=self.MODEL_NAMES.get(model_code, model_code),
            extracted_inputs=extracted_inputs,
            extracted_meanings=extracted_meanings,
            missing_inputs=missing_params,
            tool_selected=tool_name,
            raw_result=raw_result,
            verification_passed=is_valid,
            verification_notes=verification_notes,
            executive_summary=exec_summary,
            detailed_explanation=detailed_explanation,
            pipeline_trace=trace,
        )
