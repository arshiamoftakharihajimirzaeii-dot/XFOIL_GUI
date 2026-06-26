
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import os
import sys
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

# ── Colours ────────────────────────────────────────────────────────────────────
BG        = "#ffffff"
BG2       = "#ffffff"
BG3       = "#f3f6fb"
ACCENT    = "#2563eb"
ACCENT2   = "#0891b2"
FG        = "#111827"
FG2       = "#4b5563"
GREEN     = "#15803d"
RED       = "#dc2626"
BORDER    = "#d8dee9"
ENTRY_BG  = "#ffffff"
BTN_BG    = "#e5e7eb"
BTN_HOV   = "#d1d5db"

FONT_H1   = ("Segoe UI", 14, "bold")
FONT_H2   = ("Segoe UI", 11, "bold")
FONT_BODY = ("Segoe UI", 10)
FONT_MONO = ("Consolas", 9)
FONT_TINY = ("Segoe UI", 9)

# ── XFOIL path ─────────────────────────────────────────────────────────────────
DEFAULT_XFOIL = "xfoil.exe"


# ══════════════════════════════════════════════════════════════════════════════
#  Helper widgets
# ══════════════════════════════════════════════════════════════════════════════

class StyledLabel(tk.Label):
    def __init__(self, parent, text, font=None, fg=FG, **kw):
        super().__init__(parent, text=text, font=font or FONT_BODY,
                         fg=fg, bg=kw.pop("bg", BG2), **kw)


class StyledEntry(tk.Entry):
    def __init__(self, parent, width=14, **kw):
        kw.pop("bg", None)  # always use ENTRY_BG; silently drop any caller-supplied bg
        super().__init__(parent, width=width,
                         bg=ENTRY_BG, fg=FG, insertbackground=FG,
                         relief="flat", font=FONT_BODY,
                         highlightthickness=1,
                         highlightbackground=BORDER,
                         highlightcolor=ACCENT, **kw)

    def set(self, val):
        self.delete(0, tk.END)
        self.insert(0, str(val))


class StyledButton(tk.Button):
    def __init__(self, parent, text, command=None, color=BTN_BG, **kw):
        kw.setdefault("fg", FG)
        kw.setdefault("activebackground", BTN_HOV)
        kw.setdefault("activeforeground", kw["fg"])
        kw.setdefault("relief", "flat")
        kw.setdefault("font", FONT_BODY)
        kw.setdefault("padx", 12)
        kw.setdefault("pady", 5)
        kw.setdefault("cursor", "hand2")
        super().__init__(parent, text=text, command=command,
                         bg=color, **kw)
        self.bind("<Enter>", lambda e: self.config(bg=BTN_HOV))
        self.bind("<Leave>", lambda e: self.config(bg=color))


class Separator(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BORDER, height=1, **kw)


def section_header(parent, text, bg=BG2):
    f = tk.Frame(parent, bg=bg)
    tk.Label(f, text=text, font=FONT_H2, fg=ACCENT, bg=bg).pack(side="left")
    tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x", expand=True, padx=(8, 0))
    return f


# ══════════════════════════════════════════════════════════════════════════════
#  Airfoil geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def naca4_coords(code: str, n=100):
    """Return upper/lower surface x,y for a NACA 4-digit airfoil."""
    code = code.strip().upper().replace("NACA", "").strip()
    if len(code) != 4 or not code.isdigit():
        return None, None, None, None
    m = int(code[0]) / 100
    p = int(code[1]) / 10
    t = int(code[2:]) / 100

    x = np.linspace(0, 1, n)
    # thickness
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1015 * x**4)
    # camber line
    yc = np.where(x < p,
                  m / (p**2) * (2 * p * x - x**2),
                  m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2)) if m > 0 else np.zeros_like(x)
    dyc = np.where(x < p,
                   2 * m / (p**2) * (p - x),
                   2 * m / (1 - p)**2 * (p - x)) if m > 0 else np.zeros_like(x)
    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    return xu, yu, xl, yl


def load_dat_coords(filepath: str):
    """Parse a Selig-format .dat file, return arrays x, y (full contour)."""
    xs, ys = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except ValueError:
                    pass  # header row
    return np.array(xs), np.array(ys)


# ══════════════════════════════════════════════════════════════════════════════
#  XFOIL runner
# ══════════════════════════════════════════════════════════════════════════════

def build_xfoil_input(airfoil_type, airfoil_value, re, mach,
                       alpha_start, alpha_end, alpha_step,
                       n_panels, polar_file, dat_file=None):
    """Build the XFOIL command string."""
    lines = []

    if airfoil_type == "dat" and dat_file:
        lines.append(f"LOAD {dat_file}")
    else:
        lines.append(f"NACA {airfoil_value}")

    lines += [
        f"PANE {n_panels}" if n_panels else "PANE",
        "OPER",
        f"VISC {re:.0f}",
        f"MACH {mach}",
        "ITER 100",
        "PACC",
        polar_file,
        "",                 # no dump file
        f"ASEQ {alpha_start} {alpha_end} {alpha_step}",
        "PACC",
        "",
        "QUIT",
        "",
    ]
    return "\n".join(lines)


def run_xfoil(xfoil_path, input_text, timeout=60):
    """Run XFOIL and return (stdout, stderr, returncode)."""
    proc = subprocess.run(
        xfoil_path,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
    )
    return proc.stdout, proc.stderr, proc.returncode


def parse_polar(filepath):
    """Parse XFOIL polar file.  Returns a DataFrame or None."""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(
            filepath,
            skiprows=12,
            sep=r"\s+",
            names=["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"],
            engine="python",
        )
        return df.dropna()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Main application
# ══════════════════════════════════════════════════════════════════════════════

class XFoilApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XFOIL GUI")
        self.configure(bg=BG)
        self.minsize(900, 560)
        self.resizable(True, True)

        # State
        self.dat_file_path  = tk.StringVar(value="")
        self.airfoil_source = tk.StringVar(value="naca")   # "naca" | "dat"
        self.re_mode        = tk.StringVar(value="direct") # "direct" | "calc"
        self._airfoil_dat_xy = (None, None)
        self._polar_df = None

        self._build_ui()
        self._update_airfoil_preview()

    # ── Layout ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        topbar = tk.Frame(self, bg=BG, pady=4)
        topbar.pack(fill="x", padx=10)
        tk.Label(topbar, text="XFOIL GUI", font=FONT_H1,
                 fg=ACCENT, bg=BG).pack(side="left")
        tk.Label(topbar, text="Aerodynamic polar analysis",
                 font=FONT_TINY, fg=FG2, bg=BG).pack(side="left", padx=12)

        # Main area: left panel + right charts
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._build_left_panel(main)
        self._build_right_panel(main)

    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=BG2, bd=0, relief="flat", width=340)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        canvas = tk.Canvas(left, bg=BG2, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        self._left_inner = tk.Frame(canvas, bg=BG2)

        self._left_inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        left_window = canvas.create_window((0, 0), window=self._left_inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(left_window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        self._left_footer = tk.Frame(left, bg=BG2)
        self._left_footer.pack(side="bottom", fill="x")

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        p = self._left_inner
        pad = dict(padx=12, pady=4)

        # ── Airfoil source ──────────────────────────────────────────────────
        section_header(p, "  Airfoil definition").pack(fill="x", padx=12, pady=(12, 4))

        src_f = tk.Frame(p, bg=BG2)
        src_f.pack(fill="x", **pad)
        tk.Radiobutton(src_f, text="NACA 4/5-digit", variable=self.airfoil_source,
                       value="naca", command=self._on_source_change,
                       bg=BG2, fg=FG, selectcolor=BG3, activebackground=BG2,
                       font=FONT_BODY).pack(side="left")
        tk.Radiobutton(src_f, text=".dat file", variable=self.airfoil_source,
                       value="dat", command=self._on_source_change,
                       bg=BG2, fg=FG, selectcolor=BG3, activebackground=BG2,
                       font=FONT_BODY).pack(side="left", padx=12)

        # NACA row
        self._naca_frame = tk.Frame(p, bg=BG2)
        self._naca_frame.pack(fill="x", **pad)
        tk.Label(self._naca_frame, text="NACA code", font=FONT_TINY,
                 fg=FG2, bg=BG2).pack(anchor="w")
        naca_row = tk.Frame(self._naca_frame, bg=BG2)
        naca_row.pack(fill="x")
        self.entry_naca = StyledEntry(naca_row, width=10)
        self.entry_naca.set("2412")
        self.entry_naca.pack(side="left")
        self.entry_naca.bind("<Return>", lambda e: self._update_airfoil_preview())
        StyledButton(naca_row, "Preview", self._update_airfoil_preview,
                     width=8).pack(side="left", padx=6)

        # DAT file row
        self._dat_frame = tk.Frame(p, bg=BG2)
        tk.Label(self._dat_frame, text=".dat file path", font=FONT_TINY,
                 fg=FG2, bg=BG2).pack(anchor="w")
        dat_row = tk.Frame(self._dat_frame, bg=BG2)
        dat_row.pack(fill="x")
        self._dat_label = tk.Label(dat_row, text="No file selected",
                                   font=FONT_TINY, fg=FG2, bg=ENTRY_BG,
                                   anchor="w", padx=6, width=20, relief="flat",
                                   highlightthickness=1, highlightbackground=BORDER)
        self._dat_label.pack(side="left", fill="x", expand=True)
        StyledButton(dat_row, "Browse…", self._browse_dat, width=8).pack(side="left", padx=6)

        # ── Airfoil preview ─────────────────────────────────────────────────
        section_header(p, "  Airfoil shape").pack(fill="x", padx=12, pady=(10, 4))

        preview_frame = tk.Frame(p, bg=BG3, bd=0)
        preview_frame.pack(fill="x", padx=12, pady=2)
        fig_prev = Figure(figsize=(2.8, 1.1), facecolor=BG3)
        self._ax_prev = fig_prev.add_axes([0.02, 0.08, 0.96, 0.84],
                                           facecolor=BG3)
        self._ax_prev.set_aspect("equal")
        self._ax_prev.axis("off")
        self._canvas_prev = FigureCanvasTkAgg(fig_prev, preview_frame)
        self._canvas_prev.get_tk_widget().pack(fill="x")

        # ── Flow conditions ─────────────────────────────────────────────────
        section_header(p, "  Flow conditions").pack(fill="x", padx=12, pady=(12, 4))

        # Mach
        mach_f = tk.Frame(p, bg=BG2)
        mach_f.pack(fill="x", **pad)
        tk.Label(mach_f, text="Mach number", font=FONT_TINY, fg=FG2, bg=BG2).pack(anchor="w")
        self.entry_mach = StyledEntry(mach_f)
        self.entry_mach.set("0.05")
        self.entry_mach.pack(fill="x")

        # Re mode toggle
        re_hdr = tk.Frame(p, bg=BG2)
        re_hdr.pack(fill="x", padx=12, pady=(6, 2))
        tk.Label(re_hdr, text="Reynolds number", font=FONT_TINY,
                 fg=FG2, bg=BG2).pack(side="left")
        for val, lbl in (("direct", "Direct"), ("calc", "Computed")):
            tk.Radiobutton(re_hdr, text=lbl, variable=self.re_mode,
                           value=val, command=self._on_re_mode_change,
                           bg=BG2, fg=FG, selectcolor=BG3, activebackground=BG2,
                           font=FONT_TINY).pack(side="left", padx=6)

        # Solver Re value
        self._re_direct_frame = tk.Frame(p, bg=BG2)
        self._re_direct_frame.pack(fill="x", **pad)
        tk.Label(self._re_direct_frame, text="Re used by solver", font=FONT_TINY,
                 fg=FG2, bg=BG2).pack(anchor="w")
        self.entry_re = StyledEntry(self._re_direct_frame)
        self.entry_re.set("1000000")
        self.entry_re.pack(fill="x")

        # Calculated Re  ─  Re = V * L / nu
        self._re_calc_frame = tk.Frame(p, bg=BG3, padx=8, pady=6)
        fields = [
            ("Velocity V (m/s)", "entry_vel", "30"),
            ("Chord length L (m)", "entry_chord", "0.5"),
            ("Kinematic viscosity ν (m²/s)", "entry_nu", "1.5e-5"),
        ]
        for lbl, attr, default in fields:
            tk.Label(self._re_calc_frame, text=lbl, font=FONT_TINY,
                     fg=FG2, bg=BG3).pack(anchor="w")
            e = StyledEntry(self._re_calc_frame)
            e.set(default)
            e.pack(fill="x", pady=(0, 4))
            setattr(self, attr, e)

        re_calc_row = tk.Frame(self._re_calc_frame, bg=BG3)
        re_calc_row.pack(fill="x")
        StyledButton(re_calc_row, "Calculate Re →", self._calc_re).pack(side="left")
        self._re_result_label = tk.Label(re_calc_row, text="", font=FONT_TINY,
                                          fg=GREEN, bg=BG3)
        self._re_result_label.pack(side="left", padx=8)

        # ── Solver settings ─────────────────────────────────────────────────
        self._solver_header = section_header(p, "  Solver settings")
        self._solver_header.pack(fill="x", padx=12, pady=(12, 4))

        solver_grid = tk.Frame(p, bg=BG2)
        solver_grid.pack(fill="x", **pad)
        solver_grid.columnconfigure((0, 1), weight=1)

        labels_entries = [
            ("α start (°)", "entry_a1", "-5",   0, 0),
            ("α end (°)",   "entry_a2", "15",    0, 1),
            ("α step (°)",  "entry_step","1",    1, 0),
            ("Panel count", "entry_npan","160",  1, 1),
        ]
        for lbl, attr, default, row, col in labels_entries:
            f = tk.Frame(solver_grid, bg=BG2)
            f.grid(row=row, column=col, sticky="ew", padx=4, pady=3)
            tk.Label(f, text=lbl, font=FONT_TINY, fg=FG2, bg=BG2).pack(anchor="w")
            e = StyledEntry(f, width=10)
            e.set(default)
            e.pack(fill="x")
            setattr(self, attr, e)

        # ── XFOIL executable path ────────────────────────────────────────────
        section_header(p, "  XFOIL executable").pack(fill="x", padx=12, pady=(12, 4))
        exe_f = tk.Frame(p, bg=BG2)
        exe_f.pack(fill="x", **pad)
        self.entry_xfoil = StyledEntry(exe_f, width=18)
        self.entry_xfoil.set(DEFAULT_XFOIL)
        self.entry_xfoil.pack(side="left", fill="x", expand=True)
        StyledButton(exe_f, "Browse…", self._browse_xfoil).pack(side="left", padx=4)

        # ── Run button ───────────────────────────────────────────────────────
        footer = self._left_footer
        Separator(footer).pack(fill="x", padx=12, pady=(8, 6))
        self._run_btn = StyledButton(footer, "Run XFOIL Analysis",
                                     self._run_analysis, color=ACCENT,
                                     fg="#ffffff", font=FONT_H2)
        self._run_btn.pack(fill="x", padx=12, pady=4)
        self._run_btn.bind("<Enter>", lambda e: self._run_btn.config(bg="#1d4ed8"))
        self._run_btn.bind("<Leave>", lambda e: self._run_btn.config(bg=ACCENT))

        self._status_label = tk.Label(footer, text="Ready", font=FONT_TINY,
                                      fg=GREEN, bg=BG2)
        self._status_label.pack(pady=(2, 8))

        # show initial state
        self._on_source_change()
        self._on_re_mode_change()

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        nb = ttk.Notebook(right)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG3, foreground=FG2,
                         padding=[12, 5], font=FONT_BODY)
        style.map("TNotebook.Tab",
                  background=[("selected", BG2)],
                  foreground=[("selected", ACCENT)])
        nb.pack(fill="both", expand=True)

        # Tab 1: CL vs alpha
        self._tab_cl = tk.Frame(nb, bg=BG)
        nb.add(self._tab_cl, text="  CL vs α  ")
        self._fig_cl, self._ax_cl = self._make_fig()
        self._canvas_cl = self._embed_fig(self._fig_cl, self._tab_cl)

        # Tab 2: Drag polar
        self._tab_drag = tk.Frame(nb, bg=BG)
        nb.add(self._tab_drag, text="  Drag polar  ")
        self._fig_drag, self._ax_drag = self._make_fig()
        self._canvas_drag = self._embed_fig(self._fig_drag, self._tab_drag)

        # Tab 3: CM vs alpha
        self._tab_cm = tk.Frame(nb, bg=BG)
        nb.add(self._tab_cm, text="  CM vs α  ")
        self._fig_cm, self._ax_cm = self._make_fig()
        self._canvas_cm = self._embed_fig(self._fig_cm, self._tab_cm)

        # Tab 4: L/D vs alpha
        self._tab_ld = tk.Frame(nb, bg=BG)
        nb.add(self._tab_ld, text="  L/D vs α  ")
        self._fig_ld, self._ax_ld = self._make_fig()
        self._canvas_ld = self._embed_fig(self._fig_ld, self._tab_ld)

        # Tab 5: Console
        self._tab_con = tk.Frame(nb, bg=BG)
        nb.add(self._tab_con, text="  Console  ")
        self._console = tk.Text(self._tab_con, bg=BG2, fg=FG,
                                 font=FONT_MONO, relief="flat",
                                 state="disabled", wrap="word")
        con_scroll = ttk.Scrollbar(self._tab_con, command=self._console.yview)
        self._console.configure(yscrollcommand=con_scroll.set)
        self._console.pack(side="left", fill="both", expand=True)
        con_scroll.pack(side="right", fill="y")

    # ── UI helpers ───────────────────────────────────────────────────────────

    def _make_fig(self):
        fig = Figure(figsize=(6, 4.5), facecolor=BG)
        ax  = fig.add_subplot(111, facecolor=BG2)
        ax.tick_params(colors=FG2, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.xaxis.label.set_color(FG2)
        ax.yaxis.label.set_color(FG2)
        ax.grid(True, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.6)
        fig.tight_layout(pad=1.5)
        return fig, ax

    def _embed_fig(self, fig, frame):
        canvas = FigureCanvasTkAgg(fig, frame)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.config(bg=BG3)
        toolbar.update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return canvas

    def _set_status(self, msg, color=FG2):
        self._status_label.config(text=msg, fg=color)
        self.update_idletasks()

    def _log(self, text):
        self._console.config(state="normal")
        self._console.insert(tk.END, text)
        self._console.see(tk.END)
        self._console.config(state="disabled")
        self.update_idletasks()

    # ── Source / mode switching ───────────────────────────────────────────────

    def _on_source_change(self):
        if self.airfoil_source.get() == "naca":
            self._naca_frame.pack(fill="x", padx=12, pady=4)
            self._dat_frame.pack_forget()
        else:
            self._naca_frame.pack_forget()
            self._dat_frame.pack(fill="x", padx=12, pady=4)
        self._update_airfoil_preview()

    def _on_re_mode_change(self):
        self._re_direct_frame.pack(fill="x", padx=12, pady=2)
        if self.re_mode.get() == "direct":
            self._re_calc_frame.pack_forget()
        else:
            self._re_calc_frame.pack(
                fill="x",
                padx=12,
                pady=4,
                before=self._solver_header,
            )

    # ── Airfoil preview ───────────────────────────────────────────────────────

    def _update_airfoil_preview(self, *_):
        ax = self._ax_prev
        ax.cla()
        ax.set_facecolor(BG3)
        ax.axis("off")

        if self.airfoil_source.get() == "dat" and self.dat_file_path.get():
            try:
                xs, ys = load_dat_coords(self.dat_file_path.get())
                self._airfoil_dat_xy = (xs, ys)
                ax.plot(xs, ys, color=ACCENT, linewidth=1.2)
                ax.fill(xs, ys, alpha=0.18, color=ACCENT)
                ax.set_aspect("equal")
                name = os.path.basename(self.dat_file_path.get())
                ax.set_title(name, color=FG2, fontsize=8, pad=2)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                        ha="center", va="center", color=RED, fontsize=8)
        else:
            code = self.entry_naca.get().strip()
            xu, yu, xl, yl = naca4_coords(code)
            if xu is not None:
                ax.plot(np.concatenate([xu, xl[::-1]]),
                        np.concatenate([yu, yl[::-1]]),
                        color=ACCENT, linewidth=1.2)
                ax.fill(np.concatenate([xu, xl[::-1]]),
                        np.concatenate([yu, yl[::-1]]),
                        alpha=0.18, color=ACCENT)
                ax.set_aspect("equal")
                ax.set_title(f"NACA {code}", color=FG2, fontsize=8, pad=2)
            else:
                ax.text(0.5, 0.5, "Invalid NACA code", transform=ax.transAxes,
                        ha="center", va="center", color=FG2, fontsize=8)

        self._canvas_prev.draw()

    # ── File browsers ─────────────────────────────────────────────────────────

    def _browse_dat(self):
        path = filedialog.askopenfilename(
            title="Select airfoil .dat file",
            filetypes=[("Airfoil dat", "*.dat"), ("All files", "*.*")]
        )
        if path:
            self.dat_file_path.set(path)
            self._dat_label.config(text=os.path.basename(path))
            self._update_airfoil_preview()

    def _browse_xfoil(self):
        path = filedialog.askopenfilename(
            title="Select XFOIL executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")]
        )
        if path:
            self.entry_xfoil.set(path)

    # ── Reynolds calculator ───────────────────────────────────────────────────

    def _calc_re(self):
        try:
            V  = float(self.entry_vel.get())
            L  = float(self.entry_chord.get())
            nu = float(self.entry_nu.get())
            Re = V * L / nu
            self._re_result_label.config(text=f"Re = {Re:,.0f}", fg=GREEN)
            
            self.entry_re.set(f"{Re:.0f}")
        except ValueError:
            self._re_result_label.config(text="Invalid input", fg=RED)

    # ── Run analysis ──────────────────────────────────────────────────────────

    def _run_analysis(self):
        self._run_btn.config(state="disabled")
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()

    def _run_analysis_thread(self):
        try:
            self._set_status("Running XFOIL…", ACCENT)
            self._console.config(state="normal")
            self._console.delete("1.0", tk.END)
            self._console.config(state="disabled")

            xfoil_exe = self.entry_xfoil.get().strip()
            if not os.path.isfile(xfoil_exe):
                # Try PATH
                import shutil
                found = shutil.which(xfoil_exe)
                if not found:
                    self._set_status("XFOIL executable not found!", RED)
                    messagebox.showerror("XFOIL not found",
                        f"Cannot find '{xfoil_exe}'.\n"
                        "Please set the correct path to xfoil.exe.")
                    return
                xfoil_exe = found

            re_val = float(self.entry_re.get().replace(",", ""))
            mach   = float(self.entry_mach.get())
            a1     = float(self.entry_a1.get())
            a2     = float(self.entry_a2.get())
            step   = float(self.entry_step.get())
            npan_raw = self.entry_npan.get().strip()
            npan   = int(npan_raw) if npan_raw.isdigit() else None

            polar_file = os.path.join(os.path.dirname(xfoil_exe) or ".", "polar_out.txt")
            if os.path.exists(polar_file):
                os.remove(polar_file)

            src = self.airfoil_source.get()
            dat_path = self.dat_file_path.get() if src == "dat" else None
            naca_code = self.entry_naca.get().strip()

            inp = build_xfoil_input(
                airfoil_type=src,
                airfoil_value=naca_code,
                re=re_val, mach=mach,
                alpha_start=a1, alpha_end=a2, alpha_step=step,
                n_panels=npan,
                polar_file=polar_file,
                dat_file=dat_path,
            )

            self._log("═══ XFOIL INPUT ═══\n" + inp + "\n═══════════════════\n")

            stdout, stderr, rc = run_xfoil(xfoil_exe, inp, timeout=90)
            self._log(stdout or "")
            if stderr:
                self._log("\n[STDERR]\n" + stderr)

            df = parse_polar(polar_file)
            if df is None or df.empty:
                self._set_status("No polar data — check console", RED)
                self._log("\n[ERROR] Polar file empty or missing.\n")
                return

            self._polar_df = df
            self.after(0, self._plot_results, df, naca_code if src == "naca" else os.path.basename(dat_path or ""))
            self._set_status(f"Done — {len(df)} α points", GREEN)

        except Exception as e:
            self._set_status(f"Error: {e}", RED)
            self._log(f"\n[EXCEPTION] {e}\n")
        finally:
            self.after(0, lambda: self._run_btn.config(state="normal"))

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _style_ax(self, ax, xlabel, ylabel, title):
        ax.cla()
        ax.set_facecolor(BG2)
        ax.set_xlabel(xlabel, color=FG2, fontsize=10)
        ax.set_ylabel(ylabel, color=FG2, fontsize=10)
        ax.set_title(title, color=FG, fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors=FG2, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.6)

    def _plot_results(self, df, name):
        kw = dict(color=ACCENT, linewidth=2, marker="o",
                  markersize=4, markerfacecolor=ACCENT2)

        # CL vs alpha
        self._style_ax(self._ax_cl, "Angle of attack α (°)",
                       "Lift coefficient CL", f"CL vs α — {name}")
        self._ax_cl.plot(df["alpha"], df["CL"], **kw)
        # mark stall
        idx_max = df["CL"].idxmax()
        self._ax_cl.axvline(df.loc[idx_max, "alpha"], color=RED,
                             linewidth=1, linestyle="--", alpha=0.7,
                             label=f"Stall @ α={df.loc[idx_max,'alpha']:.1f}°")
        self._ax_cl.legend(facecolor=BG3, edgecolor=BORDER,
                           labelcolor=FG2, fontsize=9)
        self._fig_cl.tight_layout(pad=1.5)
        self._canvas_cl.draw()

        # Drag polar
        self._style_ax(self._ax_drag, "Drag coefficient CD",
                       "Lift coefficient CL", f"Drag polar — {name}")
        self._ax_drag.plot(df["CD"], df["CL"], **kw)
        self._fig_drag.tight_layout(pad=1.5)
        self._canvas_drag.draw()

        # CM vs alpha
        self._style_ax(self._ax_cm, "Angle of attack α (°)",
                       "Pitching moment CM", f"CM vs α — {name}")
        self._ax_cm.plot(df["alpha"], df["CM"],
                         color="#ff9f43", linewidth=2,
                         marker="s", markersize=3)
        self._fig_cm.tight_layout(pad=1.5)
        self._canvas_cm.draw()

        # L/D vs alpha
        ld = df["CL"] / df["CD"]
        self._style_ax(self._ax_ld, "Angle of attack α (°)",
                       "Lift-to-drag ratio L/D", f"L/D vs α — {name}")
        self._ax_ld.plot(df["alpha"], ld,
                         color=GREEN, linewidth=2,
                         marker="^", markersize=3)
        idx_ld = ld.idxmax()
        self._ax_ld.axvline(df.loc[idx_ld, "alpha"], color=ACCENT2,
                             linewidth=1, linestyle="--", alpha=0.7,
                             label=f"Max L/D={ld.max():.1f} @ α={df.loc[idx_ld,'alpha']:.1f}°")
        self._ax_ld.legend(facecolor=BG3, edgecolor=BORDER,
                           labelcolor=FG2, fontsize=9)
        self._fig_ld.tight_layout(pad=1.5)
        self._canvas_ld.draw()


if __name__ == "__main__":
    app = XFoilApp()
    app.mainloop()
