import sys
import json
import pandas as pd
from PySide6.QtWidgets import *
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm

# ----------------- Subplot Model -----------------
class SubplotConfig:
    def __init__(self, dataset="", x="", left=None, right=None, colors=None):
        self.dataset = dataset
        self.x = x
        self.left = left or []
        self.right = right or []
        self.colors = colors or {}

# ----------------- Plot Tab -----------------
class PlotTab(QWidget):
    MAX_SUBPLOTS = 5
    MIN_SUBPLOT_HEIGHT = 250

    def __init__(self, data):
        super().__init__()
        self.dataframes = data
        self.subplots = [None]*self.MAX_SUBPLOTS
        self.current_edit_index = 0  # default to first subplot
        self.all_columns = []

        self._build_ui()
        self._preallocate_axes()

    # ---------------- UI -----------------
    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # Controls layout
        ctrl_layout = QVBoxLayout()
        self.dataset = QComboBox()
        ctrl_layout.addWidget(QLabel("Dataset"))
        ctrl_layout.addWidget(self.dataset)

        self.x = QComboBox()
        ctrl_layout.addWidget(QLabel("X"))
        ctrl_layout.addWidget(self.x)

        # Left Y
        self.left_search = QLineEdit()
        self.left_list = QListWidget()
        self.left_list.setSelectionMode(QListWidget.MultiSelection)
        ctrl_layout.addWidget(QLabel("Left Y"))
        ctrl_layout.addWidget(self.left_search)
        ctrl_layout.addWidget(self.left_list)

        # Right Y
        self.right_search = QLineEdit()
        self.right_list = QListWidget()
        self.right_list.setSelectionMode(QListWidget.MultiSelection)
        ctrl_layout.addWidget(QLabel("Right Y"))
        ctrl_layout.addWidget(self.right_search)
        ctrl_layout.addWidget(self.right_list)

        # Subplot selection
        self.subplot_select = QComboBox()
        self.subplot_select.addItems([f"Subplot {i+1}" for i in range(self.MAX_SUBPLOTS)])
        self.subplot_select.currentIndexChanged.connect(self.change_edit_subplot)
        ctrl_layout.addWidget(QLabel("Select Subplot to Edit"))
        ctrl_layout.addWidget(self.subplot_select)

        # Buttons
        self.save_subplot_btn = QPushButton("Save Subplot")
        self.remove_subplot_btn = QPushButton("Clear Subplot")
        self.color_btn = QPushButton("Pick Color for Selected Y")
        ctrl_layout.addWidget(self.save_subplot_btn)
        ctrl_layout.addWidget(self.remove_subplot_btn)
        ctrl_layout.addWidget(self.color_btn)

        # Axis limits
        self.ymin = QLineEdit(); self.ymax = QLineEdit()
        self.y2min = QLineEdit(); self.y2max = QLineEdit()
        self.xmin = QLineEdit(); self.xmax = QLineEdit()
        ctrl_layout.addWidget(QLabel("Y1 limits")); ctrl_layout.addWidget(self.ymin); ctrl_layout.addWidget(self.ymax)
        ctrl_layout.addWidget(QLabel("Y2 limits")); ctrl_layout.addWidget(self.y2min); ctrl_layout.addWidget(self.y2max)
        ctrl_layout.addWidget(QLabel("X limits")); ctrl_layout.addWidget(self.xmin); ctrl_layout.addWidget(self.xmax)

        # Save/load session
        self.save_session_btn = QPushButton("Save Session")
        self.load_session_btn = QPushButton("Load Session")
        ctrl_layout.addWidget(self.save_session_btn)
        ctrl_layout.addWidget(self.load_session_btn)

        # Export
        self.export_btn = QPushButton("Export Plot")
        ctrl_layout.addWidget(self.export_btn)

        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl_layout)

        # Plot area
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.addWidget(self.canvas)
        plot_layout.setContentsMargins(0,0,0,0)
        plot_container.setLayout(plot_layout)
        self.scroll.setWidget(plot_container)

        main_layout.addWidget(ctrl_widget, 1)
        main_layout.addWidget(self.scroll, 3)

        # ---------- Connections ----------
        self.dataset.currentIndexChanged.connect(self.refresh_columns)
        self.left_search.textChanged.connect(self.filter_columns)
        self.right_search.textChanged.connect(self.filter_columns)
        self.save_subplot_btn.clicked.connect(self.save_current_subplot)
        self.remove_subplot_btn.clicked.connect(self.clear_current_subplot)
        self.color_btn.clicked.connect(self.pick_color)
        self.save_session_btn.clicked.connect(self.save_session)
        self.load_session_btn.clicked.connect(self.load_session)
        self.export_btn.clicked.connect(self.export_plot)

    # ---------------- Preallocate Axes -----------------
    def _preallocate_axes(self):
        self.axes = [self.fig.add_subplot(self.MAX_SUBPLOTS, 1, i+1) for i in range(self.MAX_SUBPLOTS)]
        for ax in self.axes:
            ax.clear()
        self.fig.set_size_inches(8, self.MAX_SUBPLOTS*self.MIN_SUBPLOT_HEIGHT/100)
        self.canvas.draw()

    # ---------------- Data Handling -----------------
    def refresh_columns(self):
        name = self.dataset.currentText()
        if name in self.dataframes:
            cols = list(self.dataframes[name].columns)
            self.all_columns = cols
            self.x.clear()
            self.x.addItems(cols)
            self.filter_columns()

    def filter_columns(self):
        ltxt = self.left_search.text().lower()
        rtxt = self.right_search.text().lower()
        self.left_list.clear()
        self.right_list.clear()
        for col in self.all_columns:
            if ltxt in col.lower():
                self.left_list.addItem(col)
            if rtxt in col.lower():
                self.right_list.addItem(col)

    # ---------------- Subplot Management -----------------
    def change_edit_subplot(self, idx):
        self.current_edit_index = idx
        sp = self.subplots[idx]
        if sp:
            self.dataset.setCurrentText(sp.dataset)
            self.refresh_columns()
            self.x.setCurrentText(sp.x)
            self._select_items(self.left_list, sp.left)
            self._select_items(self.right_list, sp.right)
        else:
            self.left_list.clearSelection()
            self.right_list.clearSelection()

    def _select_items(self, widget, items):
        for i in range(widget.count()):
            widget.item(i).setSelected(widget.item(i).text() in items)

    def save_current_subplot(self):
        sp = SubplotConfig(
            dataset=self.dataset.currentText(),
            x=self.x.currentText(),
            left=[i.text() for i in self.left_list.selectedItems()],
            right=[i.text() for i in self.right_list.selectedItems()],
            colors=self.subplots[self.current_edit_index].colors if self.subplots[self.current_edit_index] else {}
        )
        self.subplots[self.current_edit_index] = sp
        self.plot()

    def clear_current_subplot(self):
        self.subplots[self.current_edit_index] = None
        self.axes[self.current_edit_index].clear()
        self.canvas.draw()

    # ---------------- Color Picker -----------------
    def pick_color(self):
        color = QColorDialog.getColor()
        if not color.isValid(): return
        hex_color = color.name()
        sp = self.subplots[self.current_edit_index]
        if not sp: return
        selected = [i.text() for i in self.left_list.selectedItems()] + [i.text() for i in self.right_list.selectedItems()]
        for s in selected:
            sp.colors[s] = hex_color
        self.plot()

    # ---------------- Plotting -----------------
    def plot(self):
        for idx, sp in enumerate(self.subplots):
            ax = self.axes[idx]
            ax.clear()
            if not sp: continue
            df = self.dataframes[sp.dataset]
            ax2 = None
            lines = []

            # Left
            for i, y in enumerate(sp.left):
                color = sp.colors.get(y) or cm.tab10(i%10)
                l, = ax.plot(df[sp.x], df[y], color=color, label=y)
                lines.append(l)

            # Right
            if sp.right:
                ax2 = ax.twinx()
                for i, y in enumerate(sp.right):
                    color = sp.colors.get(y) or cm.tab10((i+len(sp.left))%10)
                    l, = ax2.plot(df[sp.x], df[y], color=color, label=y)
                    lines.append(l)

            # Labels
            ax.set_ylabel(", ".join(sp.left))
            if ax2:
                ax2.set_ylabel(", ".join(sp.right))

            # Axis limits
            self._set_limits(ax, ax2)

            ax.legend(lines, [l.get_label() for l in lines])
            ax.grid(True)

        self.fig.tight_layout()
        self.canvas.draw()

    def _set_limits(self, ax, ax2=None):
        ymin = self._to_float(self.ymin.text())
        ymax = self._to_float(self.ymax.text())
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)
        if ax2:
            ymin2 = self._to_float(self.y2min.text())
            ymax2 = self._to_float(self.y2max.text())
            if ymin2 is not None or ymax2 is not None:
                ax2.set_ylim(bottom=ymin2, top=ymax2)
        xmin = self._to_float(self.xmin.text())
        xmax = self._to_float(self.xmax.text())
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)

    def _to_float(self, val):
        try: return float(val)
        except: return None

    # ---------------- Session -----------------
    def save_session(self):
        file, _ = QFileDialog.getSaveFileName(self, "", "", "JSON (*.json)")
        if not file: return
        data = []
        for sp in self.subplots:
            if sp:
                data.append({
                    "dataset": sp.dataset,
                    "x": sp.x,
                    "left": sp.left,
                    "right": sp.right,
                    "colors": sp.colors
                })
            else:
                data.append(None)
        with open(file,"w") as f:
            json.dump(data,f)

    def load_session(self):
        file, _ = QFileDialog.getOpenFileName(self, "", "", "JSON (*.json)")
        if not file: return
        with open(file) as f:
            data = json.load(f)
        self.subplots = []
        for d in data:
            if d:
                sp = SubplotConfig(d["dataset"], d["x"], d["left"], d["right"], d["colors"])
                self.subplots.append(sp)
            else:
                self.subplots.append(None)
        self.plot()

    # ---------------- Export -----------------
    def export_plot(self):
        file, _ = QFileDialog.getSaveFileName(self, "", "", "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)")
        if not file: return
        self.fig.savefig(file)


# ----------------- Main Window -----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Preallocated Plot Tool")
        self.resize(1200, 800)

        self.dataframes = {}  # Shared datasets
        self.tabs = QTabWidget()

        # --- Top toolbar with buttons ---
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        toolbar.setLayout(toolbar_layout)

        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_csv)
        toolbar_layout.addWidget(self.load_btn)

        self.new_tab_btn = QPushButton("+")
        self.new_tab_btn.clicked.connect(self.add_tab)
        toolbar_layout.addWidget(self.new_tab_btn)

        # --- Main layout ---
        main_layout = QVBoxLayout()
        main_layout.addWidget(toolbar)
        main_layout.addWidget(self.tabs)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Add first tab
        self.add_tab()

    # ----------------- Tabs -----------------
    def add_tab(self):
        tab = PlotTab(self.dataframes)
        self.tabs.addTab(tab, f"Tab {self.tabs.count()+1}")

    # ----------------- Load CSV -----------------
    def load_csv(self):
        file, _ = QFileDialog.getOpenFileName(self, "", "Select CSV", "CSV (*.csv)")
        if not file:
            return

        df = pd.read_csv(file)
        name = file.split("/")[-1]
        self.dataframes[name] = df

        # Update dataset dropdowns in all tabs
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            tab.dataset.clear()
            tab.dataset.addItems(self.dataframes.keys())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())