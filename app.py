#!/usr/bin/env python3
"""
app.py - Supply Chain Analytics Inventory Decision Agent Application

Entry point for the Inventory Decision Agent. Accepts natural language
inventory questions, processes them through the 6-stage architecture:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain

Also provides a Streamlit web interface with CSV data upload and pandas-based
inventory data analysis.

Usage:
  streamlit run app.py              # Launches Streamlit web interface with CSV upload & analysis
  ./python3 app.py                  # Launches interactive CLI conversation mode
  ./python3 app.py --demo           # Runs demonstration across all 5 inventory models
  ./python3 app.py --question "..." # Evaluates a specific inventory query
  ./python3 app.py --question "..." --json # Outputs raw JSON decision payload
"""

import argparse
import json
import sys
from typing import List, Dict, Any

import inventory_tools as it
from inventory_agent import InventoryDecisionAgent, AgentResponse


REQUIRED_CSV_COLUMNS = ["Product", "Category", "Units", "Unit Cost"]

DEMO_QUESTIONS = [
    {
        "title": "Case 1: Economic Order Quantity (EOQ)",
        "question": (
            "Our company distributes medical supplies. We have an annual demand of 12,000 units. "
            "Each order costs $100 to place, and holding cost is $2.40 per unit per year. "
            "The unit purchase price is $10.00. What is our economic order quantity, how often should we order, "
            "and what are our total annual inventory costs?"
        ),
    },
    {
        "title": "Case 2: Quantity Discount Schedule",
        "question": (
            "We purchase industrial pumps with an annual demand of 5,000 units and ordering cost of $49 per order. "
            "Our supplier offers a volume discount schedule: 1 to 499 units at $20.00, 500 to 999 units at $19.00, "
            "and 1000 or more units at $18.00. Our holding cost rate is 20% of the unit price. "
            "Which discount tier minimizes our total annual procurement and inventory cost?"
        ),
    },
    {
        "title": "Case 3: Safety Stock & Reorder Point (Continuous Review)",
        "question": (
            "We sell high-velocity electronics with an average daily demand of 100 units and standard deviation "
            "of 15 units. Supplier replenishment lead time is 4 days. Management requires a 95% cycle service level. "
            "What should our safety stock buffer and reorder point (ROP) be to prevent stockouts?"
        ),
    },
    {
        "title": "Case 4: Fixed-Period Review Model (Periodic Review / Order-Up-To)",
        "question": (
            "Our distribution center operates a periodic review policy reviewed every 7 days with a 3-day lead time. "
            "Average daily demand is 50 units with standard deviation of 10 units. We maintain a 95% service level. "
            "Currently, we have 120 units on hand and 50 units already on order. What is our target order-up-to level "
            "and how many units should we order right now?"
        ),
    },
    {
        "title": "Case 5: Newsvendor Model (Single-Period Stochastic Demand)",
        "question": (
            "We are preparing our seasonal procurement for a holiday gift set. The item sells for $100.00 per unit "
            "and costs $40.00 to purchase. Unsold inventory at the end of the season can only be salvaged for $10.00 per unit. "
            "Demand follows a normal distribution with a mean of 1,000 units and standard deviation of 200 units. "
            "What is our critical fractile, how many units should we order, and what is our expected profit?"
        ),
    },
]


# ==============================================================================
# CSV DATA PROCESSING & PANDAS ANALYSIS HELPERS
# ==============================================================================

def validate_csv_dataframe(df: Any) -> List[str]:
    """
    Validates that the required columns (Product, Category, Units, Unit Cost)
    are present in the uploaded pandas DataFrame.
    Returns a list of missing column names (empty list if all required columns exist).
    """
    return [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]


def analyze_csv_dataframe(df: Any) -> Dict[str, Any]:
    """
    Performs all inventory CSV calculations strictly using pandas on the
    DataFrame created from the uploaded CSV file. Never hardcodes dataset values.

    Calculations:
      1. Total Units
      2. Average Unit Cost
      3. Product with the highest Units
      4. Group by Category -> Total Units per Category
      5. New calculated column: Inventory Value = Units * Unit Cost
      6. Total Inventory Value across all products
    """
    import pandas as pd

    if df.empty:
        raise ValueError("The uploaded CSV file contains no data rows.")

    analysis_df = df.copy()

    # Parse numeric values cleanly using pandas (supports numeric cells or formatted strings)
    units_numeric = pd.to_numeric(
        analysis_df["Units"].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    unit_cost_numeric = pd.to_numeric(
        analysis_df["Unit Cost"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )

    if units_numeric.isna().any():
        raise ValueError("Column 'Units' contains missing or non-numeric values.")
    if unit_cost_numeric.isna().any():
        raise ValueError("Column 'Unit Cost' contains missing or non-numeric values.")

    analysis_df["Units"] = units_numeric
    analysis_df["Unit Cost"] = unit_cost_numeric

    # 1. Total Units
    total_units = analysis_df["Units"].sum()

    # 2. Average Unit Cost
    average_unit_cost = analysis_df["Unit Cost"].mean()

    # 3. Product with highest Units
    highest_idx = analysis_df["Units"].idxmax()
    highest_row = analysis_df.loc[highest_idx]
    highest_units_product = str(highest_row["Product"])
    highest_units_value = highest_row["Units"]

    # 4. Group by Category and calculate total Units for each Category
    units_by_category = (
        analysis_df.groupby("Category", as_index=False)["Units"]
        .sum()
        .rename(columns={"Units": "Total Units"})
    )

    # 5. Create new calculated column: Inventory Value = Units × Unit Cost
    analysis_df["Inventory Value"] = analysis_df["Units"] * analysis_df["Unit Cost"]

    # 6. Total Inventory Value across all products
    total_inventory_value = analysis_df["Inventory Value"].sum()

    return {
        "row_count": int(df.shape[0]),
        "col_count": int(df.shape[1]),
        "column_names": list(df.columns),
        "total_units": total_units,
        "average_unit_cost": float(average_unit_cost),
        "highest_units_product": highest_units_product,
        "highest_units_value": highest_units_value,
        "units_by_category": units_by_category,
        "analysis_df": analysis_df,
        "total_inventory_value": float(total_inventory_value),
    }


def _format_units(val: Any) -> str:
    """Formats unit counts cleanly (integer format if whole number, else 2 decimals)."""
    try:
        fval = float(val)
        return f"{int(round(fval)):,}" if abs(fval - round(fval)) < 1e-9 else f"{fval:,.2f}"
    except Exception:
        return str(val)


# ==============================================================================
# STREAMLIT APPLICATION INTERFACE
# ==============================================================================

def is_running_in_streamlit() -> bool:
    """Detects whether app.py is currently running under `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        try:
            from streamlit import runtime
            return runtime.exists()
        except Exception:
            return False


def run_streamlit_app():
    """
    Renders the Supply Chain Analytics Inventory Decision Agent Streamlit UI
    along with the sidebar CSV Data Upload section, existing pandas-based Data Analysis view,
    and Multicriteria Inventory Classification (A, B, C, D) using Sales ($) and Hits.
    """
    import pandas as pd
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Analytics Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
    )

    agent = InventoryDecisionAgent()

    streamlit_samples = list(DEMO_QUESTIONS) + [
        {
            "title": "Case 6: Multicriteria Inventory Classification (A, B, C, D)",
            "question": (
                "Perform multicriteria inventory classification on the uploaded CSV file "
                "using Sales ($) and Hits to classify all inventory items into A, B, C, and D classes."
            ),
        }
    ]

    # ------------------------------------------------------------------
    # LEFT SIDEBAR (SINGLE CSV FILE UPLOADER AT BOTTOM)
    # ------------------------------------------------------------------
    with st.sidebar:
        st.title("📦 Inventory Agent")
        st.caption("Supply Chain Analytics Decision Support")

        st.header("Supported Models")
        st.markdown(
            "- **Economic Order Quantity (EOQ)**\n"
            "- **Quantity Discount**\n"
            "- **Safety Stock & Reorder Point**\n"
            "- **Fixed-Period Review**\n"
            "- **Newsvendor**\n"
            "- **Multicriteria Inventory Classification (A, B, C, D)**"
        )

        st.divider()

        st.header("Sample Questions")
        sample_titles = ["Custom Question"] + [c["title"] for c in streamlit_samples]
        selected_sample = st.selectbox(
            "Load a sample inventory question",
            options=sample_titles,
            index=0,
        )

        st.divider()

        # LAST SECTION AT THE BOTTOM OF THE LEFT SIDEBAR: Data Upload
        # Single CSV uploader shared across inventory analysis, classification, and agent queries.
        st.header("Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            key="sidebar_csv_uploader",
        )

    # Read the single uploaded CSV into a pandas DataFrame ONCE so every capability shares it
    uploaded_df = None
    upload_read_error = None
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            uploaded_df.columns = [str(c).strip() for c in uploaded_df.columns]
        except Exception as exc:
            upload_read_error = str(exc)

    # ------------------------------------------------------------------
    # MAIN STREAMLIT PAGE — EXISTING INVENTORY DECISION AGENT
    # ------------------------------------------------------------------
    st.title("Supply Chain Analytics Inventory Decision Agent")
    st.markdown(
        "Ask natural-language inventory questions across **EOQ**, **Quantity Discount**, "
        "**Safety Stock & Reorder Point**, **Fixed-Period Review**, **Newsvendor**, and "
        "**Multicriteria Inventory Classification (A, B, C, D)**."
    )

    default_question = ""
    if selected_sample != "Custom Question":
        for case in streamlit_samples:
            if case["title"] == selected_sample:
                default_question = case["question"]
                break

    user_question = st.text_area(
        "Enter your inventory question:",
        value=default_question,
        height=110,
        placeholder="e.g., Our annual demand is 12,000 units, ordering cost is $100, and holding cost is $2.40 per unit per year. What is our EOQ?",
    )

    if st.button("Analyze Inventory Question", type="primary"):
        if not user_question.strip():
            st.warning("Please enter an inventory question before running the agent.")
        else:
            try:
                response = agent.run(user_question.strip(), dataframe=uploaded_df)
                if response.verification_passed:
                    st.success(response.executive_summary)
                else:
                    st.warning(response.executive_summary)

                with st.expander("Pipeline Trace & Extracted Parameters", expanded=True):
                    st.write(f"**Identified Model:** {response.model_display_name} (`{response.model_identified}`)")
                    st.write(f"**Selected Python Tool:** `{response.tool_selected}`")
                    if response.extracted_inputs:
                        for k, v in response.extracted_inputs.items():
                            meaning = f" — *{response.extracted_meanings[k]}*" if k in response.extracted_meanings else ""
                            display_v = f"pandas DataFrame ({v.shape[0]} rows x {v.shape[1]} cols)" if k == "df" else v
                            st.write(f"- `{k}`: `{display_v}`{meaning}")
                    if response.missing_inputs:
                        st.error("Missing required inputs: " + ", ".join(response.missing_inputs))

                if response.model_identified == agent.MODEL_INVENTORY_CLASSIFICATION and response.raw_result is not None:
                    cls_res = response.raw_result
                    st.subheader("Final Summary Counts & Total Sales by Class (A, B, C, D)")
                    st.dataframe(
                        cls_res.class_summary_df.style.format(
                            {
                                "Item Count": "{:,}",
                                "Item Share (%)": "{:.2f}%",
                                "Total Sales ($)": "${:,.2f}",
                                "Sales Share (%)": "{:.2f}%",
                                "Total Hits": "{:,.0f}",
                                "Hits Share (%)": "{:.2f}%",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.subheader("Classified Inventory DataFrame")
                    hits_display_col = "Hits" if "Hits" in cls_res.classified_df.columns else "POS Hits"
                    st.dataframe(
                        cls_res.classified_df.style.format(
                            {
                                "Sales ($)": "${:,.2f}",
                                hits_display_col: "{:,.0f}",
                                "Sales Share (%)": "{:.2f}%",
                                "Cumulative Sales (%)": "{:.2f}%",
                                "Sales Score": "{:d}",
                                "Hits Share (%)": "{:.2f}%",
                                "Cumulative Hits (%)": "{:.2f}%",
                                "Hits Score": "{:d}",
                                "Combined Weighted Score": "{:.2f}",
                            }
                        ),
                        use_container_width=True,
                    )

                st.markdown(response.detailed_explanation)
            except Exception as exc:
                st.error(f"Error processing inventory question: {exc}")

    # ------------------------------------------------------------------
    # MAIN STREAMLIT PAGE — UPLOADED CSV DATA ANALYSIS & CLASSIFICATION
    # ------------------------------------------------------------------
    if uploaded_file is not None:
        st.divider()
        st.header("Uploaded CSV Data Analysis & Classification")

        if upload_read_error is not None:
            st.error(f"Failed to read uploaded CSV file: {upload_read_error}")
            return

        df = uploaded_df

        # Always show Uploaded DataFrame Preview & Metadata
        st.subheader("Uploaded DataFrame")
        st.dataframe(df, use_container_width=True)

        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.metric("Number of Rows", f"{df.shape[0]:,}")
        with meta_col2:
            st.metric("Number of Columns", f"{df.shape[1]:,}")
        st.markdown(f"**Column Names:** {', '.join(f'`{col}`' for col in df.columns)}")

        # Check which supported CSV schemas are satisfied by the uploaded DataFrame
        missing_analysis_cols = validate_csv_dataframe(df)
        missing_classification_cols = it.validate_classification_dataframe(df)

        has_analysis_schema = len(missing_analysis_cols) == 0
        has_classification_schema = len(missing_classification_cols) == 0

        if not has_analysis_schema and not has_classification_schema:
            st.error(
                "The uploaded CSV file does not match either supported schema.\n\n"
                f"- **Inventory Value Analysis** missing column(s): **{', '.join(missing_analysis_cols)}** "
                f"(Required: {', '.join(REQUIRED_CSV_COLUMNS)})\n"
                f"- **Multicriteria Inventory Classification (A, B, C, D)** missing column(s): "
                f"**{', '.join(missing_classification_cols)}** "
                f"(Required: {', '.join(it.REQUIRED_CLASSIFICATION_COLUMNS)})"
            )
            return

        # --------------------------------------------------------------
        # A. EXISTING INVENTORY DATA ANALYSIS (Product, Category, Units, Unit Cost)
        # --------------------------------------------------------------
        if has_analysis_schema:
            st.divider()
            st.subheader("Inventory Data Analysis Results")
            try:
                results = analyze_csv_dataframe(df)
            except Exception as exc:
                st.error(f"Unable to analyze uploaded dataset: {exc}")
            else:
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                with kpi1:
                    st.metric("Total Units", _format_units(results["total_units"]))
                with kpi2:
                    st.metric("Average Unit Cost", f"${results['average_unit_cost']:,.2f}")
                with kpi3:
                    st.metric(
                        "Product with Highest Units",
                        results["highest_units_product"],
                        delta=f"{_format_units(results['highest_units_value'])} units",
                        delta_color="off",
                    )
                with kpi4:
                    st.metric("Total Inventory Value", f"${results['total_inventory_value']:,.2f}")

                st.subheader("Total Units by Category")
                st.dataframe(
                    results["units_by_category"],
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("Inventory Data with Calculated Inventory Value")
                styled_df = results["analysis_df"].style.format(
                    {
                        "Unit Cost": "${:,.2f}",
                        "Inventory Value": "${:,.2f}",
                    }
                )
                st.dataframe(styled_df, use_container_width=True)

        # --------------------------------------------------------------
        # B. MULTICRITERIA INVENTORY CLASSIFICATION (A, B, C, D) — COURSE METHODOLOGY
        # --------------------------------------------------------------
        if has_classification_schema:
            st.divider()
            st.subheader("Multicriteria Inventory Classification (A, B, C, D)")
            st.caption(
                "Course Methodology: Pareto thresholds **A (0–60%), B (61–80%), C (81–90%), D (91–100%)** → "
                "Scores **A=4, B=3, C=2, D=1** → **Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score)** → "
                "Final Class **A (4.00–3.25), B (3.24–2.50), C (2.49–1.75), D (< 1.75)**."
            )

            try:
                cls_result = it.classify_inventory_multicriteria(df)
            except Exception as exc:
                st.error(f"Unable to perform inventory classification: {exc}")
            else:
                # KPI Summary Row showing Counts & Total Sales ($) for each Final Class (A, B, C, D)
                c1, c2, c3, c4 = st.columns(4)
                summary_lookup = cls_result.class_summary_df.set_index("Final Inventory Class")
                for col_widget, cls_label in zip([c1, c2, c3, c4], ["A", "B", "C", "D"]):
                    row_s = summary_lookup.loc[cls_label]
                    with col_widget:
                        st.metric(
                            f"Class {cls_label}",
                            f"{int(row_s['Item Count']):,} items",
                            delta=f"${row_s['Total Sales ($)']:,.2f} ({row_s['Sales Share (%)']:.1f}% Sales)",
                            delta_color="off",
                        )

                # Final Summary Counts & Total Sales Table
                st.subheader("Final Summary Counts & Total Sales by Class (A, B, C, D)")
                styled_summary = cls_result.class_summary_df.style.format(
                    {
                        "Item Count": "{:,}",
                        "Item Share (%)": "{:.2f}%",
                        "Total Sales ($)": "${:,.2f}",
                        "Sales Share (%)": "{:.2f}%",
                        "Total Hits": "{:,.0f}",
                        "Hits Share (%)": "{:.2f}%",
                    }
                )
                st.dataframe(styled_summary, use_container_width=True, hide_index=True)

                # Full DataFrame with Course Classification Columns Added
                st.subheader("Inventory DataFrame with Multicriteria Classification Results")
                hits_display_col = "Hits" if "Hits" in cls_result.classified_df.columns else "POS Hits"
                styled_classified = cls_result.classified_df.style.format(
                    {
                        "Sales ($)": "${:,.2f}",
                        hits_display_col: "{:,.0f}",
                        "Sales Share (%)": "{:.2f}%",
                        "Cumulative Sales (%)": "{:.2f}%",
                        "Sales Score": "{:d}",
                        "Hits Share (%)": "{:.2f}%",
                        "Cumulative Hits (%)": "{:.2f}%",
                        "Hits Score": "{:d}",
                        "Combined Weighted Score": "{:.2f}",
                    }
                )
                st.dataframe(styled_classified, use_container_width=True)

                with st.expander("Student Verification Breakdown (Formulas, Substitutions & Checks)", expanded=False):
                    st.markdown(cls_result.format_verification_steps())


# ==============================================================================
# EXISTING CLI APPLICATION (PRESERVED UNCHANGED)
# ==============================================================================

def print_banner():
    banner = """
================================================================================
   SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT
   Architecture: Question → Identify → Extract → Tool → Calculate → Verify → Explain
================================================================================
"""
    print(banner)


def format_cli_response(response: AgentResponse) -> str:
    """Formats the AgentResponse into a readable terminal output."""
    separator = "-" * 80
    lines = [
        separator,
        f"🎯 EXECUTIVE RECOMMENDATION",
        separator,
        response.executive_summary,
        "",
        separator,
        f"🔍 DECISION PIPELINE ARCHITECTURE TRACE",
        separator,
        f"1. Customer Question : \"{response.question}\"",
        f"2. Identified Model  : {response.model_display_name} (Code: '{response.model_identified}')",
        f"3. Extracted Inputs  :",
    ]

    for k, v in response.extracted_inputs.items():
        meaning = f"  [{response.extracted_meanings[k]}]" if k in response.extracted_meanings else ""
        if k == "discount_schedule" and isinstance(v, list):
            sched_str = ", ".join(f"[{b.min_qty}-{b.max_qty or '∞'} @ ${b.unit_price:.2f}]" for b in v)
            lines.append(f"     • {k}: {sched_str}{meaning}")
        elif k == "discrete_demand" and isinstance(v, list):
            lines.append(f"     • {k}: {v}{meaning}")
        else:
            lines.append(f"     • {k}: {v}{meaning}")

    if response.missing_inputs:
        lines.append("   ⚠️ Missing Inputs  :")
        for m in response.missing_inputs:
            lines.append(f"     • {m}")

    calc_state = "Skipped (Missing Required Inputs)" if response.missing_inputs else "Completed via Python Engine"
    lines.extend([
        f"4. Selected Tool     : {response.tool_selected}",
        f"5. Calculation State : {calc_state}",
        f"6. Verification Check: {'✅ PASSED' if response.verification_passed else '⚠️ FLAGGED'}",
    ])
    for note in response.verification_notes:
        lines.append(f"     • {note}")

    lines.extend([
        "",
        separator,
        f"📊 ANALYTICAL & MANAGERIAL BREAKDOWN",
        separator,
        response.detailed_explanation,
        separator,
    ])
    return "\n".join(lines)


def run_demo(agent: InventoryDecisionAgent):
    """Executes all 5 supply chain showcase models."""
    print_banner()
    print("🚀 Running demonstration across all 5 supply chain inventory decision models...\n")

    for i, case in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n{'='*80}")
        print(f"👉 DEMO CASE {i}: {case['title']}")
        print(f"{'='*80}")
        print(f"Question: \"{case['question']}\"\n")

        response = agent.run(case["question"])
        print(format_cli_response(response))
        print("\n")


def run_interactive(agent: InventoryDecisionAgent):
    """Runs interactive command line REPL."""
    print_banner()
    print("Welcome to the Supply Chain Analytics Inventory Decision Agent!")
    print("Ask any natural language question about inventory management.")
    print("Supported models: EOQ, Quantity Discount, Safety Stock & ROP, Fixed-Period Review, Newsvendor.")
    print("Commands:")
    print("  'demo'    - Run all 5 model case studies")
    print("  'sample'  - View sample questions")
    print("  'quit'    - Exit the program\n")

    while True:
        try:
            user_input = input("\n📦 Enter your inventory question (or 'sample', 'demo', 'quit'):\n> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting Inventory Decision Agent. Goodbye!")
                break
            if user_input.lower() == "demo":
                run_demo(agent)
                continue
            if user_input.lower() == "sample":
                print("\nAvailable sample prompts:")
                for idx, c in enumerate(DEMO_QUESTIONS, 1):
                    print(f"\n[{idx}] {c['title']}:")
                    print(f"    {c['question']}")
                continue

            # Process user question
            print("\n⚙️ Processing through decision pipeline...")
            response = agent.run(user_input)
            print(format_cli_response(response))

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Supply Chain Analytics Inventory Decision Agent CLI"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Supply chain inventory question to evaluate"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run comprehensive demo across all 5 inventory models"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output response as a JSON payload"
    )
    args = parser.parse_args()

    agent = InventoryDecisionAgent()

    if args.demo:
        run_demo(agent)
    elif args.question:
        response = agent.run(args.question)
        if args.json:
            out = {
                "question": response.question,
                "model_identified": response.model_identified,
                "model_display_name": response.model_display_name,
                "extracted_inputs": {
                    k: [b.__dict__ for b in v] if k == "discount_schedule" and isinstance(v, list) else v
                    for k, v in response.extracted_inputs.items()
                },
                "tool_selected": response.tool_selected,
                "raw_result": getattr(response.raw_result, "to_dict", lambda: str(response.raw_result))(),
                "verification_passed": response.verification_passed,
                "verification_notes": response.verification_notes,
                "executive_summary": response.executive_summary,
            }
            print(json.dumps(out, indent=2))
        else:
            print(format_cli_response(response))
    else:
        run_interactive(agent)


if __name__ == "__main__":
    if is_running_in_streamlit():
        run_streamlit_app()
    else:
        main()
