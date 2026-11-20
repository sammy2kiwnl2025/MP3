"""
Machine Problem 3 - Student pass/fail predictor (Shiny for Python)

Run locally:  shiny run app.py
"""
import json

import joblib
import pandas as pd
import sklearn
from shiny import App, reactive, render, ui

import train_model as tm

# feature -> (input id, label, short explanation, max allowed value)
FIELDS = {
    "Submission.Actions": ("submission", "Submission actions", "Assignment submissions and related actions.", 10_000),
    "Quiz.Actions": ("quiz", "Quiz actions", "Quiz attempts and related actions.", 10_000),
    "Forum.Actions": ("forum", "Forum actions", "Posts, replies and other forum activity.", 10_000),
    "Create.Actions": ("create", "Create actions", "Times the student created content in the LMS.", 10_000),
    "Read.Actions": ("read", "Read actions", "Times the student viewed or read course content.", 1_000_000),
    "Update.Actions": ("update", "Update actions", "Times the student edited existing content.", 10_000),
    "Delete.Actions": ("delete", "Delete actions", "Times the student deleted content.", 10_000),
    "Days.Logged.In": ("days", "Days logged in", "Number of different days the student logged in.", 366),
}


# ------------------------------------------------------------------ model
def load_model():
    """Load the saved model. If it is missing or was saved with a different
    scikit-learn version, retrain from the bundled CSV instead."""
    try:
        with open(tm.METRICS_PATH) as f:
            metrics = json.load(f)
        if metrics.get("sklearn_version") != sklearn.__version__:
            raise RuntimeError("scikit-learn version differs from the saved model")
        model = joblib.load(tm.MODEL_PATH)
    except Exception:
        model, metrics = tm.train(save=False)
    return model, metrics


model, metrics = load_model()  # runs once when the app starts


# ------------------------------------------------------------- static parts
def make_input(feature):
    input_id, label, help_text, max_value = FIELDS[feature]
    default = int(round(metrics["feature_median"][feature]))
    return ui.input_numeric(
        input_id,
        ui.TagList(
            label,
            ui.tags.br(),
            ui.tags.small(f"{help_text} Median: {default}.", class_="text-muted fw-normal"),
        ),
        value=default,
        min=0,
        max=max_value,
        step=1,
    )


def table_html(df):
    return ui.HTML(df.to_html(classes="table table-sm w-auto", border=0))


def importance_bars():
    items = list(metrics["feature_importance"].items())
    top = max(v for _, v in items)
    rows = []
    for feature, value in items:
        rows.append(
            ui.tags.div(
                ui.tags.div(FIELDS[feature][1], class_="small"),
                ui.tags.div(
                    ui.tags.div(
                        f"{value:.0%}",
                        class_="progress-bar",
                        role="progressbar",
                        style=f"width: {max(value / top * 100, 8):.0f}%",
                    ),
                    class_="progress mb-2",
                ),
            )
        )
    return ui.TagList(*rows)

# ----------------------------------------------------------------- server
def server(input, output, session):
    @render.ui
    @reactive.event(input.predict, ignore_none=False)
    def result():
        if input.predict() == 0:
            return ui.p("Fill in the values and click Predict outcome.", class_="text-muted")

        values = {f: input[FIELDS[f][0]]() for f in features}

        bad = [
            FIELDS[f][1]
            for f, v in values.items()
            if v is None or v < 0 or v > FIELDS[f][3]
        ]
        if bad:
            return ui.tags.div(
                "Enter a number within the allowed range for: " + ", ".join(bad) + ".",
                class_="alert alert-warning",
            )

        row = pd.DataFrame([values], columns=features)
        proba = model.predict_proba(row)[0]
        p_pass = float(proba[list(model.classes_).index("High")])  # High -> Pass
        p_fail = 1.0 - p_pass

        if p_pass >= 0.5:
            banner = ui.tags.div(
                ui.tags.strong("Likely to pass"),
                f" ({p_pass:.0%} confidence)",
                class_="alert alert-success",
            )
        else:
            banner = ui.tags.div(
                ui.tags.strong("Likely to fail"),
                f" ({p_fail:.0%} confidence)",
                class_="alert alert-danger",
            )

        return ui.TagList(
            banner,
            ui.tags.div(
                ui.tags.div(
                    class_="progress-bar",
                    role="progressbar",
                    style=f"width: {p_pass * 100:.0f}%",
                ),
                class_="progress",
            ),
            ui.p(f"Probability of passing: {p_pass:.0%}", class_="small text-muted mt-1"),
            ui.p(
                "This is a screening aid based on LMS activity only. It should prompt a "
                "conversation with the student, not replace an instructor's judgment.",
                class_="small text-muted",
            ),
        )


app = App(app_ui, server)
