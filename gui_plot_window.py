# ===============================================================
# gui_plot_window.py — Janela para exibir gráficos Matplotlib
# ===============================================================

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QMainWindow

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)


class PlotWindow(QMainWindow):
    """
    Janela independente para exibição de gráficos Matplotlib.
    """

    def __init__(self, fig, title="Gráfico"):
        super().__init__()

        self.setWindowTitle(title)
        self.setMinimumSize(800, 600)

        # Widget container
        container = QWidget()
        layout = QVBoxLayout(container)

        # Canvas do gráfico
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        # Toolbar Matplotlib
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        self.setCentralWidget(container)

        # Atualiza a renderização
        self.canvas.draw()
