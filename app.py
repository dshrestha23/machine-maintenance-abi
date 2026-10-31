import sys
from pathlib import Path


import json
import math
import joblib
import pandas as pd
import streamlit as st
from typing import Any
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.pdm.data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
HIGH_THRESHOLD = 0.50
REVIEW_THRESHOLD = 0.20

MODE_NAMES = {
    "TWF": "Tool wear",
    "HDF": "Heat dissipation",
    "PWF": "Power limits",
    "OSF": "Overstrain",
}

st.set_page_config(page_title="MaintenIQ | Maintenance Intelligence", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background:#0d1421;color:#e6ebf2;}
    [data-testid="stSidebar"] {background:#131f30;}
    [data-testid="stMetric"] {background:#182639;border:1px solid #2a3b51;
        border-radius:9px;padding:14px;}
    [data-testid="stMetricLabel"] {color:#aab9cb;}
    div.stButton > button[kind="primary"] {background:#167d80;border:0;color:white;}
    h1,h2,h3 {letter-spacing:-.025em;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_dataset(ROOT / "data" / "ai4i2020.csv")


@st.cache_resource
def get_model() -> Any:
    return joblib.load(ARTIFACTS / "random_forest.joblib")


@st.cache_data
def get_metrics() -> dict[str, Any]:
    with (ARTIFACTS / "metrics.json").open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def get_importance() -> pd.DataFrame:
    path = ARTIFACTS / "feature_importance.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def make_replay(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the held-out 15% test partition, then shuffle its order."""
    x = data.loc[:, list(FEATURE_COLUMNS)]
    y = data[TARGET_COLUMN]
    _, remaining_x, _, remaining_y = train_test_split(
        x, y, train_size=0.70, random_state=42, stratify=y
    )
    _, test_x, _, _ = train_test_split(
        remaining_x,
        remaining_y,
        test_size=0.50,
        random_state=42,
        stratify=remaining_y,
    )
    return test_x.sample(frac=1, random_state=42).reset_index(drop=True)


def priority_for(probability: float) -> str:
    if probability >= HIGH_THRESHOLD:
        return "High"
    if probability >= REVIEW_THRESHOLD:
        return "Review"
    return "Routine"


def predict_probability(frame: pd.DataFrame) -> float:
    return float(model.predict_proba(frame.loc[:, list(FEATURE_COLUMNS)])[0, 1])


def screen_modes(row: pd.Series) -> list[dict[str, str]]:
    """UCI AI4I-inspired checks; not learned failure-mode classifiers."""
    kind = str(row["Type"])
    air = float(row["Air temperature [K]"])
    process = float(row["Process temperature [K]"])
    speed = float(row["Rotational speed [rpm]"])
    torque = float(row["Torque [Nm]"])
    wear = float(row["Tool wear [min]"])
    delta = process - air
    power = torque * speed * 2 * math.pi / 60
    interaction = wear * torque
    findings = []

    if wear >= 200:
        findings.append(
            {
                "code": "TWF",
                "title": "Tool wear",
                "evidence": (
                    f"Wear is {wear:.0f} min, reaching the 200–240 min interval "
                    "in which the dataset's tool replacement or failure point is selected. "
                    "This is a concern, not a confirmed tool failure."
                ),
                "action": "Inspect tool condition and review the replacement schedule.",
            }
        )

    if delta < 8.6 and speed < 1380:
        findings.append(
            {
                "code": "HDF",
                "title": "Heat dissipation",
                "evidence": (
                    f"Process-to-air temperature difference is {delta:.2f} K "
                    f"(below 8.6 K), while speed is {speed:,.0f} rpm "
                    "(below 1,380 rpm)."
                ),
                "action": "Check cooling conditions, temperature measurements and speed.",
            }
        )

    if power < 3500 or power > 9000:
        findings.append(
            {
                "code": "PWF",
                "title": "Power limits",
                "evidence": (
                    f"Calculated mechanical power is {power:,.0f} W, outside "
                    "the dataset's 3,500–9,000 W operating band."
                ),
                "action": "Verify torque, rotational speed and process loading.",
            }
        )

    limit = {"L": 11000, "M": 12000, "H": 13000}.get(kind)
    if limit is not None and interaction > limit:
        findings.append(
            {
                "code": "OSF",
                "title": "Overstrain",
                "evidence": (
                    f"Wear × torque is {wear:.0f} × {torque:.1f} = "
                    f"{interaction:,.0f} min·Nm, above the {limit:,.0f} min·Nm "
                    f"limit for type {kind}."
                ),
                "action": "Inspect tool condition and assess mechanical loading.",
            }
        )
    return findings


def record_row(record: dict[str, Any]) -> pd.Series:
    return pd.Series({column: record[column] for column in FEATURE_COLUMNS})


def assess_record(row: pd.Series, probability: float, identifier: str) -> None:
    """Render an individual assessment: measurement evidence then next checks."""
    findings = screen_modes(row)
    status = priority_for(probability)
    a, b = st.columns(2)
    a.metric("Estimated failure risk", f"{probability:.1%}")
    b.metric("Assessment", status)
    st.progress(max(0.0, min(1.0, probability)))

    st.subheader("Risk Drivers")
    if findings:
        for finding in findings:
            st.markdown(f"**{finding['title']}** — {finding['evidence']}")
    else:
        st.write(
            "The screening rules did not identify a specific failure mechanism "
            "in these measurements. The model's risk estimate is a separate result."
        )

    st.subheader("Inspection Guidance")
    actions = list(dict.fromkeys(item["action"] for item in findings))
    if status == "High":
        st.error(
            f"{identifier}: prioritise an inspection before the next operating period."
        )
    elif status == "Review":
        st.warning(
            f"{identifier}: verify operating conditions and consider inspection."
        )
    else:
        st.success(
            f"{identifier}: continue normal monitoring and scheduled maintenance."
        )
    if actions:
        for action in actions:
            st.write(f"- {action}")
    elif status != "Routine":
        st.write(
            "Check the operating measurements and assess the equipment using standard procedures."
        )


def next_reading() -> None:
    position = st.session_state.replay_position
    if position >= len(replay):
        st.session_state.monitoring = False
        return
    row = replay.iloc[position]
    probability = predict_probability(replay.iloc[[position]])
    modes = screen_modes(row)
    record = {column: row[column] for column in FEATURE_COLUMNS}
    record.update(
        {
            "Observation": f"OBS-{position + 1:04d}",
            "Sequence": position + 1,
            "Risk (%)": round(probability * 100, 3),
            "Priority": priority_for(probability),
            "Conditions": ", ".join(item["code"] for item in modes) or "—",
        }
    )
    st.session_state.readings.append(record)
    st.session_state.replay_position += 1
    st.session_state.shift_report = None
    if st.session_state.replay_position >= len(replay):
        st.session_state.monitoring = False


def create_shift_report(readings: list[dict[str, Any]]) -> dict[str, Any]:
    high = [r for r in readings if r["Priority"] == "High"]
    review = [r for r in readings if r["Priority"] == "Review"]
    attention = high + review
    ranked = sorted(attention, key=lambda r: r["Risk (%)"], reverse=True)
    counts = {code: 0 for code in MODE_NAMES}
    actions: dict[str, int] = {}
    for record in high:
        for finding in screen_modes(record_row(record)):
            counts[finding["code"]] += 1
            actions[finding["action"]] = actions.get(finding["action"], 0) + 1

    if high:
        status = "High"
        summary = (
            f"{len(high)} of {len(readings)} processed observations need priority attention. "
            f"A further {len(review)} fall into the review category."
        )
    elif review:
        status = "Review"
        summary = (
            f"No high-risk observations were recorded. "
            f"{len(review)} of {len(readings)} observations warrant verification."
        )
    else:
        status = "Routine"
        summary = (
            f"All {len(readings)} observations remained below the review threshold. "
            "Continue normal monitoring."
        )

    return {
        "status": status,
        "summary": summary,
        "high": len(high),
        "review": len(review),
        "total": len(readings),
        "maximum": max(r["Risk (%)"] for r in readings),
        "ranked": ranked,
        "counts": counts,
        "actions": actions,
    }


try:
    data = get_data()
    model = get_model()
    metrics_file = get_metrics()
    importance = get_importance()
    replay = make_replay(data)
except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
    st.error(f"Unable to load project data or model: {exc}")
    st.stop()

for key, default in {
    "monitoring": False,
    "monitoring_started": False,
    "replay_position": 0,
    "readings": [],
    "shift_report": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.title("MaintenIQ")
    st.caption("PREDICTIVE MAINTENANCE INTELLIGENCE")
    st.divider()
    page = st.radio(
        "Workspace",
        [
            "Operations Overview",
            "Live Monitoring",
            "Risk Assessment",
            "Model Intelligence",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("MONITORING STATUS")
    if st.session_state.monitoring:
        st.success("Monitoring active")
        st.caption("Receiving replay observations")
    elif st.session_state.replay_position >= len(replay):
        st.info("Monitoring complete")
        st.caption(f"{len(st.session_state.readings)} observations processed")
    elif st.session_state.monitoring_started:
        st.warning("Monitoring stopped")
        st.caption(f"{len(st.session_state.readings)} observations processed")
    else:
        st.info("Monitoring standby")
    st.caption("MODEL STATUS")
    st.success("Random Forest ready")

st.title(page)
st.caption("Maintenance risk assessment and operational decision support")

if page == "Operations Overview":
    total = len(data)
    failed = int(data[TARGET_COLUMN].sum())
    a, b, c, d = st.columns(4)
    a.metric("Historical observations", f"{total:,}")
    b.metric("Recorded failures", f"{failed:,}")
    c.metric("Historical failure rate", f"{failed / total:.2%}")
    d.metric("Observations processed", len(st.session_state.readings))

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Historical Outcomes")
        st.bar_chart(
            pd.DataFrame(
                {
                    "Outcome": ["Normal", "Failure"],
                    "Observations": [total - failed, failed],
                }
            ).set_index("Outcome")
        )
    with right:
        st.subheader("Failure Rate by Product Type")
        grouped = data.groupby("Type")[TARGET_COLUMN].agg(["size", "sum"])
        grouped["Failure rate (%)"] = grouped["sum"] / grouped["size"] * 100
        st.bar_chart(grouped["Failure rate (%)"])

    if st.session_state.readings:
        recent = pd.DataFrame(st.session_state.readings)
        st.subheader("Current Monitoring Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("High-priority readings", int((recent["Priority"] == "High").sum()))
        c2.metric("Review readings", int((recent["Priority"] == "Review").sum()))
        c3.metric("Peak estimated risk", f"{recent['Risk (%)'].max():.1f}%")
    else:
        st.info("Start monitoring to populate the current operational summary.")

elif page == "Live Monitoring":
    st.subheader("Operations Monitoring")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.session_state.monitoring:
            if st.button("Stop Monitoring", type="primary", width="stretch"):
                st.session_state.monitoring = False
                st.rerun()
        else:
            label = (
                "Resume Monitoring"
                if st.session_state.monitoring_started
                else "Start Monitoring"
            )
            if st.button(
                label,
                type="primary",
                disabled=st.session_state.replay_position >= len(replay),
                width="stretch",
            ):
                st.session_state.monitoring = True
                st.session_state.monitoring_started = True
                st.rerun()
    with c2:
        if st.button(
            "Process 25 Readings",
            disabled=st.session_state.monitoring
            or st.session_state.replay_position >= len(replay),
            width="stretch",
        ):
            st.session_state.monitoring_started = True
            for _ in range(min(25, len(replay) - st.session_state.replay_position)):
                next_reading()
            st.rerun()
    with c3:
        if st.button("Reset Feed", width="stretch"):
            st.session_state.monitoring = False
            st.session_state.monitoring_started = False
            st.session_state.replay_position = 0
            st.session_state.readings = []
            st.session_state.shift_report = None
            st.rerun()

    @st.fragment(run_every="3s")
    def monitoring_panel() -> None:
        if st.session_state.monitoring:
            next_reading()
        readings = st.session_state.readings
        high = sum(r["Priority"] == "High" for r in readings)
        review = sum(r["Priority"] == "Review" for r in readings)
        a, b, c, d = st.columns(4)
        a.metric(
            "Feed",
            (
                "Active"
                if st.session_state.monitoring
                else (
                    "Complete"
                    if st.session_state.replay_position >= len(replay)
                    else "Stopped" if st.session_state.monitoring_started else "Standby"
                )
            ),
        )
        b.metric("Processed", len(readings))
        c.metric("High priority", high)
        d.metric("Review", review)

        if not readings:
            st.info("Start monitoring or process a batch to view incoming assessments.")
            return

        frame = pd.DataFrame(readings)
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Risk Across Processed Readings")
            st.line_chart(frame.set_index("Sequence")["Risk (%)"])
        with right:
            latest = readings[-1]
            st.subheader("Latest Observation")
            st.metric("Estimated failure risk", f"{latest['Risk (%)']:.1f}%")
            st.write(f"**Reference:** {latest['Observation']}")
            st.write(f"**Priority:** {latest['Priority']}")
            st.write(f"**Screening:** {latest['Conditions']}")

        st.subheader("Observation Log")
        st.dataframe(
            frame[["Observation", "Risk (%)", "Priority", "Conditions"]]
            .tail(12)
            .iloc[::-1],
            hide_index=True,
            width="stretch",
        )
        st.divider()
        st.subheader("Shift Intelligence")
        if st.button(
            "Generate Shift Report",
            type="primary",
            disabled=st.session_state.monitoring,
            width="stretch",
        ):
            st.session_state.shift_report = create_shift_report(readings)

        report = st.session_state.shift_report
        if report is None:
            st.caption(
                "Stop monitoring to generate a summary of processed observations."
            )
            return

        a, b, c = st.columns(3)
        a.metric("Observations analysed", report["total"])
        b.metric("High-priority observations", report["high"])
        c.metric("Maximum estimated risk", f"{report['maximum']:.1f}%")

        st.subheader("Shift Summary")
        if report["status"] == "High":
            st.error(report["summary"])
        elif report["status"] == "Review":
            st.warning(report["summary"])
        else:
            st.success(report["summary"])

        active_modes = [
            (code, count) for code, count in report["counts"].items() if count
        ]
        if active_modes:
            description = "; ".join(
                f"{MODE_NAMES[code]} ({count})" for code, count in active_modes
            )
            st.write(f"**Conditions among high-risk readings:** {description}.")
        elif report["high"]:
            st.write(
                "The high-risk model predictions did not coincide with a named "
                "failure-mode screening condition."
            )

        st.subheader("Maintenance Priorities")
        if report["ranked"]:
            priority_frame = pd.DataFrame(report["ranked"])
            st.dataframe(
                priority_frame[
                    [
                        "Observation",
                        "Priority",
                        "Risk (%)",
                        "Conditions",
                        "Type",
                        "Torque [Nm]",
                        "Tool wear [min]",
                    ]
                ].head(15),
                hide_index=True,
                width="stretch",
            )
            if len(report["ranked"]) > 15:
                st.caption(
                    f"Showing the 15 highest-risk of {len(report['ranked'])} flagged observations."
                )
            top = report["ranked"][0]
            st.write(
                f"**First review:** {top['Observation']} "
                f"({top['Risk (%)']:.1f}% estimated failure risk)."
            )
        else:
            st.write(
                "No immediate inspection priorities were identified in this batch."
            )

        st.subheader("Next-Shift Advisory")
        if report["status"] == "High":
            st.write(
                "Review high-priority observations before the next operating period."
            )
        elif report["status"] == "Review":
            st.write("Verify flagged measurements during the next maintenance review.")
        else:
            st.write("Continue routine checks and standard maintenance scheduling.")

        if report["actions"]:
            for action, count in sorted(
                report["actions"].items(), key=lambda item: -item[1]
            ):
                st.write(f"- {action}")
        elif report["high"]:
            st.write(
                "- Check flagged operating conditions using standard inspection procedures."
            )

        st.caption(
            "Mode checks identify conditions worth investigating, not confirmed causes. "
            "The advisory is a maintenance priority, not a dated failure forecast."
        )

    monitoring_panel()

elif page == "Risk Assessment":
    st.subheader("Operating Condition Assessment")
    with st.form("assessment"):
        left, right = st.columns(2)
        with left:
            kind = st.selectbox("Product type", ["L", "M", "H"])
            air = st.number_input("Air temperature (K)", min_value=0.0, value=300.0)
            process = st.number_input(
                "Process temperature (K)", min_value=0.0, value=310.0
            )
        with right:
            speed = st.number_input("Rotational speed (rpm)", min_value=0, value=1500)
            torque = st.number_input("Torque (Nm)", min_value=0.0, value=40.0)
            wear = st.number_input("Tool wear (min)", min_value=0, value=100)
        submitted = st.form_submit_button(
            "Assess Conditions", type="primary", width="stretch"
        )

    if submitted:
        sample = pd.DataFrame(
            [
                {
                    "Type": kind,
                    "Air temperature [K]": air,
                    "Process temperature [K]": process,
                    "Rotational speed [rpm]": speed,
                    "Torque [Nm]": torque,
                    "Tool wear [min]": wear,
                }
            ],
            columns=list(FEATURE_COLUMNS),
        )
        probability = predict_probability(sample)
        st.session_state["planner_assessment"] = {
            "record": sample.iloc[0].to_dict(),
            "probability": float(probability),
        }

    # Render the saved assessment on every Streamlit rerun, including
    # reruns triggered by the adaptive planner's select boxes.
    assessment = st.session_state.get("planner_assessment")
    if assessment is not None:
        st.divider()
        assess_record(
            pd.Series(assessment["record"]),
            assessment["probability"],
            "Manual assessment",
        )
        st.session_state["planner_prediction"] = assessment["probability"]

    if assessment is not None:
        st.divider()
        st.subheader("Adaptive Maintenance Planner")

        st.caption(
            "Context-aware, rule-based maintenance guidance. "
            "Business inputs change the recommendation, "
            "not the Random Forest prediction."
        )

        criticality = st.selectbox(
            "Production criticality",
            ["Low", "Medium", "High"],
            index=1,
            key="planner_criticality",
        )

        col_tech, col_parts = st.columns(2)

        with col_tech:
            technician = st.selectbox(
                "Qualified technician",
                ["Available", "Unavailable"],
                key="planner_technician",
            )

        with col_parts:
            parts = st.selectbox(
                "Required spare parts",
                ["Available", "Unavailable"],
                key="planner_parts",
            )

        risk = st.session_state["planner_prediction"]

        if risk >= HIGH_THRESHOLD:
            priority = "High"

            if technician == "Unavailable":
                recommendation = (
                    "Escalate urgently to the maintenance supervisor "
                    "and arrange qualified technical support. "
                    "Follow approved site safety procedures."
                )
            elif parts == "Unavailable":
                recommendation = (
                    "Arrange a qualified inspection, escalate the "
                    "parts shortage, and assess safe operating status "
                    "under site procedures."
                )
            elif criticality == "High":
                recommendation = (
                    "Prioritise qualified inspection of this "
                    "production-critical asset and review whether "
                    "continued operation is safe."
                )
            else:
                recommendation = (
                    "Arrange prompt qualified inspection and "
                    "coordinate maintenance with production."
                )

        elif risk >= REVIEW_THRESHOLD:
            priority = "Review"

            if criticality == "High":
                recommendation = (
                    "Prioritise a condition review for this "
                    "production-critical asset and plan an inspection."
                )
            elif technician == "Unavailable":
                recommendation = (
                    "Record the review requirement and schedule "
                    "qualified inspection when support is available."
                )
            else:
                recommendation = (
                    "Review operating conditions and schedule "
                    "an inspection according to maintenance priorities."
                )

        else:
            priority = "Routine"
            recommendation = (
                "Continue routine monitoring and planned maintenance. "
                "Reassess if operating conditions change."
            )

        st.info(
            f"**{priority} risk — Adaptive recommendation:** "
            f"{recommendation}"
        )

        st.caption(
            "Decision-support guidance only. Not a validated "
            "industrial safety rule or automated maintenance instruction."
        )

        # Feedback-driven policy learning: session-only, no model retraining.
        st.divider()
        st.subheader("Maintenance Learning & Optimisation")
        st.caption(
            "Prototype feedback loop. Record outcomes for maintenance actions, "
            "then compare observed results in matching risk and operating contexts. "
            "Use simulated outcomes only for demonstration; these are not factory results."
        )

        if "maintenance_feedback" not in st.session_state:
            st.session_state["maintenance_feedback"] = []

        # Only compare two inspection-planning approaches where qualified staff
        # and required parts are available. Never learn around safety escalation.
        eligible = (
            priority in ("High", "Review")
            and technician == "Available"
            and parts == "Available"
        )
        options = (
            "Prioritised qualified inspection",
            "Coordinated qualified inspection",
        )
        context = (priority, criticality, technician, parts)
        history = st.session_state["maintenance_feedback"]
        matching = [item for item in history if item["context"] == context]

        total_records = len(history)
        successful_records = sum(item["successful"] for item in history)
        m1, m2, m3 = st.columns(3)
        m1.metric("Feedback records (this session)", total_records)
        m2.metric("Successful outcomes", successful_records)
        m3.metric(
            "Observed success rate",
            f"{successful_records / total_records:.0%}" if total_records else "—",
        )

        if not eligible:
            st.warning(
                "Learning-based action ranking is unavailable in this context. "
                "Follow the risk-based guidance above. Comparative inspection "
                "planning requires Review/High risk, an available qualified "
                "technician, and available parts."
            )
        else:
            st.write(
                f"**Matching context:** {priority} risk · {criticality} production "
                "criticality · technician and parts available"
            )
            stats = {}
            for option in options:
                records = [item for item in matching if item["action"] == option]
                successes = sum(item["successful"] for item in records)
                count = len(records)
                # Laplace smoothing reduces extreme estimates from tiny samples.
                stats[option] = (count, successes, (successes + 1) / (count + 2))

            # Require evidence for both options before allowing feedback to
            # change the preferred approach. Default is a conservative heuristic.
            evidence_ready = all(stats[option][0] >= 2 for option in options)
            preferred = options[0]
            if evidence_ready and stats[options[1]][2] > stats[options[0]][2]:
                preferred = options[1]

            for option in options:
                count, successes, score = stats[option]
                label = " (currently preferred)" if option == preferred else ""
                st.write(
                    f"**{option}{label}** — {successes}/{count} reported successful; "
                    f"smoothed outcome score {score:.0%}"
                )

            if evidence_ready:
                st.success(
                    f"**Feedback-informed preference:** {preferred}. "
                    "Ranking is based on the outcomes recorded for this exact context."
                )
            else:
                st.info(
                    f"**Initial preference:** {preferred}. "
                    "Record at least two outcomes for EACH approach in this "
                    "context to enable feedback-informed ranking."
                )

            with st.form("maintenance_feedback_form", clear_on_submit=False):
                action = st.selectbox(
                    "Maintenance approach taken",
                    list(options),
                    key="learning_action",
                )
                outcome = st.selectbox(
                    "Reported follow-up outcome",
                    ["Issue resolved", "Issue not resolved"],
                    key="learning_outcome",
                )
                confirmed_demo = st.checkbox(
                    "This is simulated feedback entered for the prototype demo",
                    key="learning_demo_confirmation",
                )
                save_feedback = st.form_submit_button(
                    "Record feedback", type="primary", width="stretch"
                )
            if save_feedback:
                if not confirmed_demo:
                    st.error("Confirm that the feedback is simulated before saving.")
                else:
                    st.session_state["maintenance_feedback"].append(
                        {
                            "context": context,
                            "action": action,
                            "successful": outcome == "Issue resolved",
                            "outcome": outcome,
                            "risk_probability": risk,
                            "source": "Simulated demo entry",
                        }
                    )
                    st.rerun()

        with st.expander("Feedback history (this session)"):
            if history:
                rows = [
                    {
                        "Risk": item["context"][0],
                        "Criticality": item["context"][1],
                        "Technician": item["context"][2],
                        "Parts": item["context"][3],
                        "Action": item["action"],
                        "Outcome": item["outcome"],
                        "Source": item["source"],
                    }
                    for item in history
                ]
                st.dataframe(pd.DataFrame(rows).iloc[::-1], hide_index=True, width="stretch")
            else:
                st.write("No feedback has been recorded in this session.")

        if st.button("Clear demo feedback", key="clear_maintenance_feedback"):
            st.session_state["maintenance_feedback"] = []
            st.rerun()

        st.caption(
            "Learning updates the inspection-action ranking, not the Random "
            "Forest failure probability. Feedback is stored in this browser "
            "session only and is lost when the session resets. Small samples, "
            "unverified outcomes, and selection bias limit the ranking's reliability. "
            "This is not a validated maintenance optimisation or safety system."
        )

elif page == "Model Intelligence":
    st.subheader("Model Performance")
    test = metrics_file.get("test_metrics", {})

    def metric_value(*names: str) -> float | None:
        for name in names:
            if name in test:
                return float(test[name])
        return None

    a, b, c, d = st.columns(4)
    for container, label, keys in [
        (a, "Accuracy", ("accuracy",)),
        (b, "Precision", ("precision",)),
        (c, "Recall", ("recall",)),
        (d, "F1-score", ("f1_score", "f1")),
    ]:
        value = metric_value(*keys)
        container.metric(
            label, f"{value:.2%}" if value is not None else "Not available"
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Test Confusion Matrix")
        matrix = test.get("confusion_matrix")
        if matrix is not None:
            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=["Actual normal", "Actual failure"],
                    columns=["Predicted normal", "Predicted failure"],
                ),
                width="stretch",
            )
            st.caption(
                f"Detected {matrix[1][1]} failures; missed {matrix[1][0]}; "
                f"raised {matrix[0][1]} false alerts."
            )
        else:
            st.info("Confusion matrix is unavailable in the saved metrics.")
    with right:
        st.subheader("Feature Importance")
        if not importance.empty:
            feature_column = next(
                (
                    name
                    for name in ["feature", "Feature", "feature_name"]
                    if name in importance
                ),
                None,
            )
            value_column = next(
                (
                    name
                    for name in ["importance_mean", "mean_importance", "importance"]
                    if name in importance
                ),
                None,
            )
            if feature_column and value_column:
                st.bar_chart(
                    importance.set_index(feature_column)[value_column].sort_values(
                        ascending=False
                    )
                )
            else:
                st.dataframe(importance, width="stretch")
        else:
            st.info("Feature importance is not available.")

    with st.expander("Evaluation Details"):
        for label, names in [
            ("ROC-AUC", ("roc_auc",)),
            ("PR-AUC", ("pr_auc",)),
        ]:
            value = metric_value(*names)
            if value is not None:
                st.write(f"**{label}:** {value:.4f}")
        for key, label in [
            ("selected_feature_set", "Selected features"),
            ("selected_cross_validation_f1", "Cross-validation F1"),
            ("best_parameters", "Best hyperparameters"),
        ]:
            if key in metrics_file:
                st.write(f"**{label}:**")
                st.write(metrics_file[key])

    # with st.expander("About the System"):
    #     st.write(
    #         "MaintenIQ is a predictive maintenance decision-support prototype. "
    #         "It uses a trained Random Forest to estimate overall machine-failure "
    #         "risk from six operating inputs."
    #     )
    #     st.write(
    #         "**Data provenance:** AI4I 2020 is a synthetic manufacturing benchmark "
    #         "dataset. The monitoring interface replays shuffled, held-out test "
    #         "observations; it is not connected to physical equipment and does not "
    #         "represent a chronological machine history."
    #     )
    #     st.write(
    #         "**Interpretation:** Tool wear, heat dissipation, power and overstrain "
    #         "checks use published AI4I mechanisms as screening rules. They are not "
    #         "separately trained failure-mode classifiers or confirmed diagnoses. "
    #         "Random failure cannot be inferred reliably from the input measurements."
    #     )
    #     st.write(
    #         "**Scope:** Risk estimates concern the supplied operating conditions. "
    #         "The model does not predict time to failure or failures in a specified "
    #         "future period. The displayed maintenance thresholds are prototype "
    #         "decision rules rather than validated industrial safety standards."
    #     )

st.divider()
st.caption("MaintenIQ | Maintenance decision intelligence")
