"""
Smart Grid Stability Analyzer - Class 12 Capstone
Libraries used: pandas, numpy, scipy, matplotlib, tkinter (+ our own ml.py)

Run:  python app.py       (keep ml.py in the same folder)
"""
import pickle
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ml import (Scaler, train_test_split, LogisticRegressionScratch, NeuralNetScratch,
                KNNScratch, metrics, permutation_importance, feature_correlations)

GREEN, RED, BLUE, BG = "#16a34a", "#dc2626", "#2563eb", "#f1f5f9"


# ------------------------------------------------------------------ helpers
def add_time_features(df):
    """If the CSV has a 'date' column, create 'hour' and 'weekday' features from it."""
    df = df.copy()
    if "date" in df.columns and "hour" not in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        if d.notna().all():
            df["hour"] = d.dt.hour + d.dt.minute / 60
            df["weekday"] = d.dt.weekday
    return df


def fill_tree(tree, df, max_rows=3000):
    tree.delete(*tree.get_children())
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"
    for c in df.columns:
        tree.heading(c, text=str(c))
        tree.column(c, width=90, anchor="center")
    for row in df.head(max_rows).itertuples(index=False):
        tree.insert("", "end", values=[f"{v:.4f}" if isinstance(v, float) else v for v in row])


def scrolled_tree(parent, height=8):
    f = tk.Frame(parent)
    tree = ttk.Treeview(f, height=height)
    vs = ttk.Scrollbar(f, orient="vertical", command=tree.yview)
    hs = ttk.Scrollbar(f, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vs.grid(row=0, column=1, sticky="ns")
    hs.grid(row=1, column=0, sticky="ew")
    f.rowconfigure(0, weight=1)
    f.columnconfigure(0, weight=1)
    return f, tree


class Chart:
    def __init__(self, parent, size=(4.6, 3.0)):
        self.fig = Figure(figsize=size, dpi=90)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.widget = self.canvas.get_tk_widget()

    def draw(self, fn):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        fn(ax)
        self.fig.tight_layout()
        self.canvas.draw()


# ------------------------------------------------------------------ friendly input names
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ROOMS = {1: "Kitchen", 2: "Living room", 3: "Laundry room", 4: "Office room", 5: "Bathroom",
         6: "Outside (north wall)", 7: "Ironing room", 8: "Teenager's room", 9: "Parents' room"}
WEATHER = {"T_out": "Temperature (\u00b0C)", "RH_out": "Humidity (%)", "Press_mm_hg": "Pressure (mm Hg)",
           "Windspeed": "Wind speed (m/s)", "Visibility": "Visibility (km)", "Tdewpoint": "Dew point (\u00b0C)"}
GROUP_ORDER = ["Rooms - temperature and humidity", "Weather outside", "Usage and time",
               "Reaction times", "Power (+ produces, - consumes)", "Price elasticity", "Other inputs"]


def describe(name):
    """dataset column (T1, RH_2, ...) -> (friendly title, subtitle) shown on the form."""
    m = re.fullmatch(r"T(\d)", name)
    if m:
        return ROOMS.get(int(m.group(1)), f"Room {m.group(1)}"), "Temperature (\u00b0C)"
    m = re.fullmatch(r"RH_(\d)", name)
    if m:
        return ROOMS.get(int(m.group(1)), f"Room {m.group(1)}"), "Humidity (%)"
    if name in WEATHER:
        return "Weather station", WEATHER[name]
    fixed = {"lights": ("Lights", "Energy used (Wh)"), "hour": ("Time of day", "HH:MM, 24-hour"),
             "weekday": ("Day of week", "Pick a day")}
    if name in fixed:
        return fixed[name]
    for prefix, sub in (("tau", "Reaction time (s)"), ("p", "Power (+ produce / - consume)"), ("g", "Price elasticity")):
        m = re.fullmatch(prefix + r"(\d)", name)
        if m:
            return f"Participant {m.group(1)}", sub
    return name, ""


def group_of(name):
    if re.fullmatch(r"T\d|RH_\d", name):
        return GROUP_ORDER[0]
    if name in WEATHER:
        return GROUP_ORDER[1]
    if name in ("lights", "hour", "weekday"):
        return GROUP_ORDER[2]
    for i, pat in ((3, r"tau\d"), (4, r"p\d"), (5, r"g\d")):
        if re.fullmatch(pat, name):
            return GROUP_ORDER[i]
    return GROUP_ORDER[6]


def layout(names):
    """[(group title, [columns in a nice order]), ...]"""
    def key(item):
        idx, n = item
        m = re.fullmatch(r"T(\d)", n)
        if m:
            return (int(m.group(1)), 0)
        m = re.fullmatch(r"RH_(\d)", n)
        return (int(m.group(1)), 1) if m else (idx, 0)
    out = []
    for g in GROUP_ORDER:
        items = [(i, n) for i, n in enumerate(names) if group_of(n) == g]
        if items:
            out.append((g, [n for _, n in sorted(items, key=key)]))
    return out


def float_to_time(v):
    h, m = int(v) % 24, int(round((v - int(v)) * 60))
    if m == 60:
        h, m = (h + 1) % 24, 0
    return f"{h:02d}:{m:02d}"


def hint_for(df, name):
    """small grey 'typical range' text taken from the loaded dataset."""
    if df is None or name not in df.columns or name == "weekday":
        return ""
    lo, hi = float(df[name].min()), float(df[name].max())
    return f"{float_to_time(lo)} to {float_to_time(hi)}" if name == "hour" else f"typical: {lo:.4g} to {hi:.4g}"


class TimeBox(ttk.Entry):
    """Type 15:40 (or 15.67); the model still receives the decimal hour."""
    def get(self):
        t = super().get().strip()
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", t)
        return str(int(m.group(1)) + int(m.group(2)) / 60) if m else t

    def insert(self, index, text):
        try:
            text = float_to_time(float(text))
        except ValueError:
            pass
        super().insert(index, text)


class DayBox(ttk.Combobox):
    """Pick Monday..Sunday; the model still receives 0..6."""
    def __init__(self, parent):
        super().__init__(parent, values=DAYS, state="readonly", width=11)

    def get(self):
        t = super().get()
        return str(DAYS.index(t)) if t in DAYS else ""

    def delete(self, first, last=None):
        self.set("")

    def insert(self, index, text):
        try:
            self.set(DAYS[int(float(text)) % 7])
        except (ValueError, IndexError):
            pass


# ------------------------------------------------------------------ app
class App(tk.Tk):
    FRIENDLY_NATIVE = True      # tells optional theme files the friendly form is already built in

    def __init__(self):
        super().__init__()
        self.title("Smart Grid Stability Analyzer")
        self.geometry("1150x740")
        self.configure(bg=BG)

        self.train_df = None
        self.test_df = None
        self.bundle = None      # dict: model, scaler, features, classes, target
        self.entries = {}
        self.batch_out = None

        tk.Label(self, text="⚡ Smart Grid Stability Analyzer", bg=BG, fg=BLUE,
                 font=("Segoe UI", 20, "bold")).pack(pady=(10, 0))
        tk.Label(self, text="AI built from scratch with NumPy to predict grid stress and energy demand — "
                            "fewer blackouts, less wasted energy, lower emissions",
                 bg=BG, font=("Segoe UI", 10)).pack()

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=12, pady=8)
        self.t1, self.t2, self.t3, self.t4 = (tk.Frame(self.nb, bg=BG) for _ in range(4))
        for t, name in [(self.t1, "1. Data"), (self.t2, "2. Train Model"),
                        (self.t3, "3. Predict One"), (self.t4, "4. Predict Many")]:
            self.nb.add(t, text=f"   {name}   ")

        self.build_data()
        self.build_train()
        self.build_predict()
        self.build_batch()

    # ============================================================ TAB 1 DATA
    def build_data(self):
        t = self.t1
        bar = tk.Frame(t, bg=BG)
        bar.pack(fill="x", pady=6)
        ttk.Button(bar, text="Load TRAIN csv", command=lambda: self.load_csv("train")).pack(side="left", padx=4)
        ttk.Button(bar, text="Load TEST csv (optional)", command=lambda: self.load_csv("test")).pack(side="left", padx=4)
        tk.Label(bar, text="Target column:", bg=BG).pack(side="left", padx=(14, 2))
        self.target_box = ttk.Combobox(bar, state="readonly", width=14)
        self.target_box.pack(side="left")
        self.target_box.bind("<<ComboboxSelected>>", self.on_target_change)
        tk.Label(bar, text="Ignore columns (comma separated):", bg=BG).pack(side="left", padx=(14, 2))
        self.ignore_var = tk.StringVar(value="stab")
        ttk.Entry(bar, textvariable=self.ignore_var, width=14).pack(side="left")
        ttk.Button(bar, text="Apply", command=self.refresh_data).pack(side="left", padx=6)
        ttk.Button(bar, text="Correlation heatmap", command=self.show_heatmap).pack(side="left", padx=4)

        bar2 = tk.Frame(t, bg=BG)
        bar2.pack(fill="x", pady=2)
        tk.Label(bar2, text="If target is a NUMBER -> High demand when value >", bg=BG).pack(side="left", padx=(6, 2))
        self.thr_var = tk.StringVar(value="")
        ttk.Entry(bar2, textvariable=self.thr_var, width=10).pack(side="left")
        tk.Label(bar2, text="(auto = median)", bg=BG, fg="#64748b").pack(side="left", padx=4)
        self.shuffle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar2, text="Shuffle rows before train/test split "
                                   "(untick for time-ordered data)", variable=self.shuffle_var).pack(side="left", padx=20)

        self.data_info = tk.Label(t, text="Load your CSV to begin", bg=BG, fg=BLUE, font=("Segoe UI", 10, "bold"),
                                  wraplength=1100, justify="left")
        self.data_info.pack(anchor="w", padx=6)

        body = tk.Frame(t, bg=BG)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="Statistical summary (df.describe)", bg=BG, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        f, self.stats_tree = scrolled_tree(left, 7)
        f.pack(fill="both", expand=True)
        tk.Label(left, text="Correlation of each feature with the target (SciPy Pearson r)",
                 bg=BG, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 0))
        self.corr_text = tk.Text(left, height=7, font=("Consolas", 10))
        self.corr_text.pack(fill="x")
        self.class_chart = Chart(body)
        self.class_chart.widget.pack(side="right", padx=8)

    def load_csv(self, which):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not p:
            return
        try:
            df = pd.read_csv(p)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        has_date = "date" in df.columns
        df = add_time_features(df)
        if which == "train":
            self.train_df = df
            cols = list(df.columns)
            self.target_box["values"] = cols
            if "stabf" in cols:
                self.target_box.set("stabf")
            elif "Appliances" in cols:
                self.target_box.set("Appliances")
            else:
                self.target_box.set(cols[-1])
            auto_ignore = [c for c in ("stab", "rv1", "rv2") if c in cols]
            self.ignore_var.set(", ".join(auto_ignore))
            self.shuffle_var.set(not has_date)       # time data: keep order
            self.thr_var.set("")
        else:
            self.test_df = df
        self.refresh_data()

    def on_target_change(self, event=None):
        self.thr_var.set("")
        self.refresh_data()

    def setup_columns(self):
        """Work out target, features and class names from the UI choices."""
        df = self.train_df
        self.target = self.target_box.get()
        ignore = [c.strip() for c in self.ignore_var.get().split(",") if c.strip()]
        self.features = [c for c in df.columns
                         if c != self.target and c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
        col = df[self.target]
        self.threshold_mode = bool(pd.api.types.is_numeric_dtype(col) and col.nunique() > 2)
        if self.threshold_mode:
            try:
                self.threshold = float(self.thr_var.get())
            except ValueError:
                self.threshold = float(col.median())
                self.thr_var.set(f"{self.threshold:g}")
            self.classes = ["Normal", "High"]
        else:
            self.classes = sorted(col.astype(str).unique())

    def labels_01(self, df, target, classes, threshold_mode, threshold):
        """Turn the target column of df into 0/1 (1 = classes[1])."""
        if threshold_mode:
            return (df[target] > threshold).astype(int).to_numpy()
        return (df[target].astype(str) == classes[1]).astype(int).to_numpy()

    def xy(self, df):
        X = df[self.features].to_numpy(dtype=float)
        y = self.labels_01(df, self.target, self.classes, self.threshold_mode,
                           getattr(self, "threshold", None))
        return X, y

    def refresh_data(self):
        if self.train_df is None:
            return
        self.setup_columns()
        df = self.train_df
        if len(self.classes) != 2:
            messagebox.showerror("Target problem",
                                 f"Target '{self.target}' has {len(self.classes)} text classes. Pick a column with "
                                 f"exactly 2 classes (e.g. stable/unstable) or a numeric column.")
            return
        test_n = len(self.test_df) if self.test_df is not None else 0
        mode = (f"NUMBER target -> classes: Normal (<= {self.threshold:g}) vs High (> {self.threshold:g})"
                if self.threshold_mode else f"CLASS target: {self.classes[0]} vs {self.classes[1]}")
        self.data_info.config(text=f"{mode}\nTrain rows: {len(df)} | Test rows: {test_n or 'auto 80/20 split'} | "
                                   f"Missing: {int(df.isna().sum().sum())}\n"
                                   f"Features ({len(self.features)}): {', '.join(self.features)}")
        desc = df[self.features].describe().T.round(3).reset_index().rename(columns={"index": "feature"})
        fill_tree(self.stats_tree, desc)

        X, y = self.xy(df)
        r = feature_correlations(X, y)
        order = np.argsort(-np.abs(r))
        lines = [f"{self.features[i]:<10} r = {r[i]:+.3f}" for i in order]
        self.corr_text.delete("1.0", tk.END)
        self.corr_text.insert(tk.END, f"(positive r => higher value pushes towards '{self.classes[1]}')\n"
                              + "\n".join(lines))

        counts = [int((y == 0).sum()), int((y == 1).sum())]
        self.class_chart.draw(lambda ax: (ax.bar(self.classes, counts, color=[BLUE, "#f59e0b"]),
                                          ax.set_title(f"Class distribution of '{self.target}'"),
                                          ax.set_ylabel("count")))

    def show_heatmap(self):
        if self.train_df is None:
            messagebox.showwarning("No data", "Load the train CSV first.")
            return
        self.setup_columns()
        corr = self.train_df[self.features].corr().to_numpy()
        win = tk.Toplevel(self)
        win.title("Correlation heatmap")
        ch = Chart(win, size=(7, 6))
        ch.widget.pack()

        def plot(ax):
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(self.features)))
            ax.set_yticks(range(len(self.features)))
            ax.set_xticklabels(self.features, rotation=90, fontsize=6)
            ax.set_yticklabels(self.features, fontsize=6)
            ch.fig.colorbar(im, ax=ax)
            ax.set_title("Feature correlation matrix")
        ch.draw(plot)

    # ============================================================ TAB 2 TRAIN
    def build_train(self):
        t = self.t2
        bar = tk.Frame(t, bg=BG)
        bar.pack(fill="x", pady=6)
        tk.Label(bar, text="Model:", bg=BG).pack(side="left", padx=(6, 2))
        self.model_box = ttk.Combobox(bar, state="readonly", width=22,
                                      values=["Neural Network", "Logistic Regression", "k-Nearest Neighbours"])
        self.model_box.set("Logistic Regression")
        self.model_box.pack(side="left")
        self.p = {}
        for label, key, default in [("Learning rate", "lr", "0.3"), ("Epochs", "epochs", "800"),
                                    ("Hidden neurons", "hidden", "32"), ("k (kNN)", "k", "5")]:
            tk.Label(bar, text=label, bg=BG).pack(side="left", padx=(12, 2))
            v = tk.StringVar(value=default)
            ttk.Entry(bar, textvariable=v, width=7).pack(side="left")
            self.p[key] = v
        ttk.Button(bar, text="TRAIN", command=self.train).pack(side="left", padx=12)
        ttk.Button(bar, text="Save model", command=self.save_model).pack(side="left", padx=3)
        ttk.Button(bar, text="Load model", command=self.load_model).pack(side="left", padx=3)

        self.train_status = tk.Label(t, text="", bg=BG, fg=BLUE, font=("Segoe UI", 10, "bold"))
        self.train_status.pack(anchor="w", padx=6)

        body = tk.Frame(t, bg=BG)
        body.pack(fill="both", expand=True)
        self.result_text = tk.Text(body, width=44, font=("Consolas", 10))
        self.result_text.pack(side="left", fill="y", padx=6)
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        self.chart_a = Chart(right, (5.2, 3.0))
        self.chart_a.widget.pack()
        self.chart_b = Chart(right, (5.2, 3.2))
        self.chart_b.widget.pack()

    def train(self):
        if self.train_df is None:
            messagebox.showwarning("No data", "Load the train CSV in tab 1 first.")
            return
        self.refresh_data()
        if len(self.classes) != 2:
            return
        try:
            lr, epochs = float(self.p["lr"].get()), int(self.p["epochs"].get())
            hidden, k = int(self.p["hidden"].get()), int(self.p["k"].get())
        except ValueError:
            messagebox.showerror("Bad settings", "Check the numbers in the settings boxes.")
            return

        Xtr, ytr = self.xy(self.train_df)
        if self.test_df is not None:
            Xte, yte = self.xy(self.test_df)
        else:
            Xtr, Xte, ytr, yte = train_test_split(Xtr, ytr, 0.2, shuffle=self.shuffle_var.get())

        scaler = Scaler().fit(Xtr)
        A, B = scaler.transform(Xtr), scaler.transform(Xte)

        name = self.model_box.get()
        self.train_status.config(text="Training... please wait")
        self.update()
        if name == "Neural Network":
            model = NeuralNetScratch(hidden, lr, epochs)
        elif name == "Logistic Regression":
            model = LogisticRegressionScratch(lr, epochs)
        else:
            model = KNNScratch(k)
        model.fit(A, ytr)

        sub = np.random.default_rng(0).permutation(len(A))[:3000]      # sample keeps big data fast
        m_tr = metrics(ytr[sub], model.predict(A[sub]))
        m_te = metrics(yte, model.predict(B))
        self.bundle = dict(model=model, scaler=scaler, features=self.features,
                           classes=self.classes, target=self.target, name=name,
                           test_accuracy=m_te["accuracy"], threshold_mode=self.threshold_mode,
                           threshold=getattr(self, "threshold", None))
        self.train_status.config(text=f"✔ {name} trained | test accuracy = {m_te['accuracy'] * 100:.2f}%")
        self.show_results(name, m_tr, m_te, model, A, ytr, B, yte)
        self.build_predict_form()

    def show_results(self, name, m_tr, m_te, model, A, ytr, B, yte):
        cm = m_te["cm"]
        c0, c1 = self.classes
        txt = (f"Model: {name}\n"
               f"{'-' * 40}\n"
               f"Train accuracy : {m_tr['accuracy'] * 100:6.2f}%\n"
               f"Test accuracy  : {m_te['accuracy'] * 100:6.2f}%\n"
               f"Precision      : {m_te['precision']:.3f}\n"
               f"Recall         : {m_te['recall']:.3f}\n"
               f"F1-score       : {m_te['f1']:.3f}\n\n"
               f"Confusion matrix (test)\n"
               f"                 pred {c0[:6]:<6} pred {c1[:6]:<6}\n"
               f"actual {c0[:6]:<6}   {cm[0, 0]:<11} {cm[0, 1]:<6}\n"
               f"actual {c1[:6]:<6}   {cm[1, 0]:<11} {cm[1, 1]:<6}\n\n"
               f"(precision/recall are for '{c1}')\n"
               f"Guessing the bigger class would score {max(yte.mean(), 1 - yte.mean()) * 100:.1f}%,\n"
               f"so a good model must beat that.\n")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, txt)

        # chart A: loss curve, or accuracy vs k for kNN
        if isinstance(model, KNNScratch):
            ks = list(range(1, 26, 4))
            accs = []
            for kk in ks:
                accs.append(np.mean(KNNScratch(kk).fit(A, ytr).predict(B[:800]) == yte[:800]) * 100)
            self.chart_a.draw(lambda ax: (ax.plot(ks, accs, "o-", color=BLUE), ax.set_xlabel("k"),
                                          ax.set_ylabel("test accuracy %"), ax.set_title("Accuracy vs k")))
        else:
            self.chart_a.draw(lambda ax: (ax.plot(model.history, color=RED), ax.set_xlabel("epoch"),
                                          ax.set_ylabel("loss"), ax.set_title("Training loss curve")))

        # chart B: permutation importance
        imp = permutation_importance(model, B, yte)
        order = np.argsort(imp)
        names = [self.features[i] for i in order]
        self.chart_b.draw(lambda ax: (ax.barh(names, imp[order], color=BLUE),
                                      ax.set_title("Feature importance (accuracy drop when shuffled)"),
                                      ax.tick_params(labelsize=8)))

    def save_model(self):
        if not self.bundle:
            messagebox.showinfo("Nothing to save", "Train a model first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Model", "*.pkl")])
        if p:
            with open(p, "wb") as f:
                pickle.dump(self.bundle, f)
            messagebox.showinfo("Saved", "Model saved.")

    def load_model(self):
        p = filedialog.askopenfilename(filetypes=[("Model", "*.pkl")])
        if not p:
            return
        with open(p, "rb") as f:
            self.bundle = pickle.load(f)
        self.features = self.bundle["features"]
        self.classes = self.bundle["classes"]
        self.target = self.bundle["target"]
        self.threshold_mode = self.bundle.get("threshold_mode", False)
        self.threshold = self.bundle.get("threshold")
        self.train_status.config(text=f"✔ Loaded {self.bundle['name']} "
                                      f"(test accuracy {self.bundle['test_accuracy'] * 100:.2f}%)")
        self.build_predict_form()

    # ============================================================ TAB 3 PREDICT ONE
    def build_predict(self):
        t = self.t3
        tk.Label(t, text="Enter grid parameters, then click Predict", bg=BG,
                 font=("Segoe UI", 11, "bold")).pack(pady=8)
        self.form = tk.Frame(t, bg=BG)
        self.form.pack()
        row = tk.Frame(t, bg=BG)
        row.pack(pady=10)
        ttk.Button(row, text="Predict", command=self.predict_one).pack(side="left", padx=5)
        ttk.Button(row, text="Fill random row from data", command=self.fill_random).pack(side="left", padx=5)
        ttk.Button(row, text="Clear", command=self.clear_form).pack(side="left", padx=5)
        self.one_result = tk.Label(t, text="Train or load a model first", bg="white",
                                   font=("Segoe UI", 18, "bold"), height=2)
        self.one_result.pack(fill="x", padx=30, pady=6)
        self.one_detail = tk.Label(t, text="", bg=BG, wraplength=1000, justify="left", font=("Segoe UI", 10))
        self.one_detail.pack(padx=30)

    def build_predict_form(self):
        for w in self.form.winfo_children():
            w.destroy()
        self.entries.clear()
        row = 0
        for group, items in layout(self.bundle["features"]):
            tk.Label(self.form, text=group.upper(), bg=BG, fg="#64748b", font=("Segoe UI", 8, "bold")).grid(
                row=row, column=0, columnspan=6, sticky="w", padx=14, pady=(12, 0))
            row += 1
            for i, name in enumerate(items):
                cell = tk.Frame(self.form, bg=BG)
                cell.grid(row=row + i // 6, column=i % 6, padx=12, pady=4, sticky="w")
                title, sub = describe(name)
                tk.Label(cell, text=title, bg=BG, fg=BLUE, font=("Segoe UI", 10, "bold")).pack(anchor="w")
                tk.Label(cell, text=sub, bg=BG, fg="#64748b", font=("Segoe UI", 8)).pack(anchor="w")
                if name == "weekday":
                    e = DayBox(cell)
                elif name == "hour":
                    e = TimeBox(cell, width=11)
                else:
                    e = ttk.Entry(cell, width=11)
                e.pack(anchor="w", pady=(2, 0))
                hint = hint_for(self.train_df, name)
                if hint:
                    tk.Label(cell, text=hint, bg=BG, fg="#94a3b8", font=("Segoe UI", 8)).pack(anchor="w")
                self.entries[name] = e
            row += (len(items) + 5) // 6
        self.one_result.config(text="Enter values and click Predict", fg="black")

    def fill_random(self):
        src = self.test_df if self.test_df is not None else self.train_df
        if src is None or not self.entries:
            messagebox.showwarning("Missing", "Need a trained model and a loaded dataset.")
            return
        row = src.sample(1).iloc[0]
        for name, e in self.entries.items():
            e.delete(0, tk.END)
            if name in row:
                e.insert(0, f"{float(row[name]):.4f}")

    def clear_form(self):
        for e in self.entries.values():
            e.delete(0, tk.END)
        self.one_result.config(text="Enter values and click Predict", fg="black")
        self.one_detail.config(text="")

    def predict_raw(self, X):
        b = self.bundle
        p1 = b["model"].predict_proba(b["scaler"].transform(X))   # probability of classes[1]
        idx = (p1 >= 0.5).astype(int)
        labels = [b["classes"][i] for i in idx]
        conf = np.where(idx == 1, p1, 1 - p1) * 100
        return labels, conf, p1

    def predict_one(self):
        if not self.bundle:
            messagebox.showwarning("No model", "Train or load a model first.")
            return
        try:
            vals = [float(self.entries[f].get()) for f in self.bundle["features"]]
        except ValueError:
            messagebox.showerror("Invalid input", "Fill every box with a number.")
            return
        labels, conf, p1 = self.predict_raw(np.array([vals]))
        good = labels[0] == self.bundle["classes"][0]
        if self.bundle.get("threshold_mode"):
            msg = f"Energy demand: {labels[0].upper()}  ({conf[0]:.1f}% confidence)"
        else:
            msg = f"Grid is {labels[0].upper()}  ({conf[0]:.1f}% confidence)"
        self.one_result.config(text=msg, fg=GREEN if good else RED)
        self.one_detail.config(text=self.advice(good, self.bundle.get("threshold_mode")))

    @staticmethod
    def advice(stable, energy_mode=False):
        if energy_mode:
            if stable:
                return ("✔ Normal demand. The grid can meet this load with its regular, efficient supply "
                        "mix - no extra backup generation needed.")
            return ("⚠ High demand expected (peak load). Peaks force utilities to switch on costly, "
                    "high-emission backup plants and can overload the grid. Suggested actions: shift "
                    "flexible loads (washing machines, EV charging) to off-peak hours, use smart "
                    "meters and time-of-use pricing, and pre-store renewable energy in batteries.")
        if stable:
            return ("✔ The grid is operating safely. Stable grids need less emergency backup generation, "
                    "so energy losses and carbon emissions stay low.")
        return ("⚠ Instability risk! Suggested actions: increase participant reaction speed (lower tau), "
                "use dynamic pricing to shift demand (adjust g), and balance power generation and "
                "consumption (p) to prevent blackouts and wasteful emergency generation.")

    # ============================================================ TAB 4 BATCH
    def build_batch(self):
        t = self.t4
        bar = tk.Frame(t, bg=BG)
        bar.pack(fill="x", pady=6)
        ttk.Button(bar, text="Load CSV & Predict All Rows", command=self.run_batch).pack(side="left", padx=6)
        ttk.Button(bar, text="Export Results", command=self.export_batch).pack(side="left", padx=6)
        self.batch_info = tk.Label(bar, text="", bg=BG, fg=BLUE, font=("Segoe UI", 10, "bold"))
        self.batch_info.pack(side="left", padx=10)
        body = tk.Frame(t, bg=BG)
        body.pack(fill="both", expand=True)
        f, self.batch_tree = scrolled_tree(body, 18)
        f.pack(side="left", fill="both", expand=True)
        self.pie = Chart(body)
        self.pie.widget.pack(side="right", padx=8)

    def run_batch(self):
        if not self.bundle:
            messagebox.showwarning("No model", "Train or load a model first.")
            return
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not p:
            return
        df = add_time_features(pd.read_csv(p))
        feats = self.bundle["features"]
        missing = [f for f in feats if f not in df.columns]
        if missing:
            messagebox.showerror("Column mismatch", f"CSV is missing: {missing}")
            return
        labels, conf, _ = self.predict_raw(df[feats].to_numpy(dtype=float))
        out = df.copy()
        out["Predicted"] = labels
        out["Confidence_%"] = np.round(conf, 2)
        self.batch_out = out
        fill_tree(self.batch_tree, out)

        info = f"{len(out)} rows predicted"
        tgt = self.bundle["target"]
        if tgt in df.columns:
            b = self.bundle
            truth = self.labels_01(df, tgt, b["classes"], b.get("threshold_mode", False), b.get("threshold"))
            pred01 = np.array([b["classes"].index(l) for l in labels])
            acc = np.mean(truth == pred01) * 100
            info += f" | accuracy on this file: {acc:.2f}%"
        self.batch_info.config(text=info)

        counts = pd.Series(labels).value_counts()
        cols = [GREEN if i == self.bundle["classes"][0] else RED for i in counts.index]
        self.pie.draw(lambda ax: (ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=cols),
                                  ax.set_title("Predicted grid state")))

    def export_batch(self):
        if self.batch_out is None:
            messagebox.showinfo("Nothing to export", "Run a batch prediction first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            self.batch_out.to_csv(p, index=False)
            messagebox.showinfo("Saved", f"Saved to {p}")


if __name__ == "__main__":
    #App().mainloop()
    # OPTIONAL dark theme: comment out the line above and un-comment the next line
    import theme_ui; theme_ui.run(App())
