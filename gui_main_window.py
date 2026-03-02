# ======================================================================
# gui_main_window.py — Janela principal da aplicação econométrica
# (MQO-only + Menus + Cancelar + robustez + Splitters + 6 painéis)
#
#  - Split principal (Vertical): Tabs (Resultados/Tabela) 40% | Figuras 60%
#  - Área de figuras:
#       Split vertical (2 linhas iguais)
#         - Linha 1: Split horizontal (3 colunas iguais)  -> Boxplot | Gráficos | Resíduos
#         - Linha 2: Split horizontal (3 colunas iguais)  -> Corr    | Aderência | Histograma
#  - Ao carregar arquivo: abre automaticamente o diálogo da variável dependente
#  - Painéis SEM título (somente borda + figura). Duplo clique abre PlotWindow.
#  - Pós-fit: executa Resultados e foca na aba "Resultados" (não muda para Tabela)
#  - Seleção de modelo: NO-OP se idx já está selecionado (sem log) + imprime descrição das transformações
#
#  Requisito (atual):
#    - "Modelo | Usar o modelo sem outliers?" deve ficar DESABILITADO ao escolher novo modelo
#      e só ser HABILITADO após "Modelo | Limpar Outliers".
#    - Implementado via flag interno: self._limpo_ready / self._limpo_ready_idx.
#
#  Mudanças nesta versão:
#    - Entrada aceita: CSV/TXT/TSV/TAB com delimitadores ";", ",", TAB (autodetect + fallback)
#    - Entrada aceita: Excel (.xls/.xlsx/.xlsm) e ODS (.ods)
#    - "Calcular (Fit MQO)" e "Limpar Outliers" passam a ser operações BLOQUEANTES
#      (menus desabilitam durante a execução e reabilitam automaticamente ao terminar)
# ======================================================================

import os
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QPlainTextEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QMessageBox, QDialog, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QFrame, QSizePolicy, QSplitter,
    QScrollArea, QToolBar, QStyle
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QAction, QKeySequence, QIcon

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from gui_worker import start_worker
from gui_plot_window import PlotWindow
from gui_style import load_dark_style

from model import MQO


class ClickableCanvas(FigureCanvas):
    def __init__(self, fig, on_dblclick=None, parent=None):
        super().__init__(fig)
        self._on_dblclick = on_dblclick
        if parent is not None:
            self.setParent(parent)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    def mouseDoubleClickEvent(self, event):
        if callable(self._on_dblclick):
            self._on_dblclick()
        return super().mouseDoubleClickEvent(event)


# ============================================================
# Painel de figura (sem título visível) + duplo-clique para abrir janela
# ============================================================
class FigurePanel(QFrame):
    def __init__(self, key: str, window_title: str, open_request=None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._key = str(key)
        self._window_title = str(window_title)
        self._open_request = open_request
        self._fig = None

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(0)

        self.placeholder = QLabel("Sem figura")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #888;")
        self.placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self.placeholder, 1)

        self._canvas: FigureCanvas | None = None

        self.setToolTip(self._window_title)

    def set_figure(self, fig):
        self._fig = fig

        if self._canvas is not None:
            self.layout().removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None

        if fig is None:
            self.placeholder.show()
            return

        def _open():
            if callable(self._open_request):
                self._open_request(self._key, self._window_title)

        self._canvas = ClickableCanvas(fig, on_dblclick=_open, parent=self)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.placeholder.hide()
        self.layout().addWidget(self._canvas, 1)
        self._canvas.draw_idle()

    def mouseDoubleClickEvent(self, event):
        try:
            if callable(self._open_request):
                self._open_request(self._key, self._window_title)
        finally:
            super().mouseDoubleClickEvent(event)


# ============================================================
# Diálogos
# ============================================================
class DependentVarDialog(QDialog):
    def __init__(self, parent: QMainWindow, columns: list[str], current: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Definir variável dependente")
        self.setModal(True)
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Selecione a variável dependente (alvo):"))

        self.combo = QComboBox()
        self.combo.addItems([str(c) for c in columns])
        if current:
            i = self.combo.findText(str(current))
            if i >= 0:
                self.combo.setCurrentIndex(i)
        root.addWidget(self.combo)

        row = QHBoxLayout()
        row.addStretch()

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_cancel)

        btn_ok = QPushButton("Definir")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        row.addWidget(btn_ok)

        root.addLayout(row)

    def selected(self) -> str:
        return self.combo.currentText().strip()


class SelectModelDialog(QDialog):
    def __init__(self, parent: QMainWindow, max_idx: int, current: int | None = None):
        super().__init__(parent)
        self.setWindowTitle("Selecionar modelo (idx)")
        self.setModal(True)
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Selecione o índice do modelo:"))

        self.spin = QSpinBox()
        self.spin.setMinimum(0)
        self.spin.setMaximum(max(0, int(max_idx)))
        if current is not None:
            self.spin.setValue(max(0, min(int(current), int(max_idx))))
        root.addWidget(self.spin)

        row = QHBoxLayout()
        row.addStretch()

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_cancel)

        btn_ok = QPushButton("Selecionar")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        row.addWidget(btn_ok)

        root.addLayout(row)

    def selected(self) -> int:
        return int(self.spin.value())


class CleanOutliersDialog(QDialog):
    def __init__(self, parent: QMainWindow, r2_alvo: float, lim_sigma: float):
        super().__init__(parent)
        self.setWindowTitle("Limpar Outliers")
        self.setModal(True)
        self.setMinimumWidth(460)

        root = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("R² alvo:"))
        self.spin_r2 = QDoubleSpinBox()
        self.spin_r2.setDecimals(3)
        self.spin_r2.setSingleStep(0.01)
        self.spin_r2.setRange(0.0, 0.999)
        self.spin_r2.setValue(float(r2_alvo))
        row1.addWidget(self.spin_r2)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("lim σ:"))
        self.spin_sigma = QSpinBox()
        self.spin_sigma.setRange(1, 10)
        self.spin_sigma.setValue(int(round(float(lim_sigma))))
        row2.addWidget(self.spin_sigma)
        root.addLayout(row2)

        rowb = QHBoxLayout()
        rowb.addStretch()

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        rowb.addWidget(btn_cancel)

        btn_ok = QPushButton("Limpar")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        rowb.addWidget(btn_ok)

        root.addLayout(rowb)

    def values(self) -> tuple[float, float]:
        return float(self.spin_r2.value()), float(self.spin_sigma.value())

class PredictionDialog(QDialog):
    def __init__(self, parent, x_columns, current_y):
        super().__init__(parent)
        self.setWindowTitle("Predição de Valor de Mercado")
        self.setMinimumWidth(450)
        
        layout = QVBoxLayout(self)
        self.inputs = {}

        layout.addWidget(QLabel("<b>Características do Imóvel:</b>"))
        
        # Grid para os inputs
        for col in x_columns:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{col}:"))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 999999999.0) 
            spin.setDecimals(4)
            spin.setValue(0.0)
            row.addWidget(spin)
            self.inputs[col] = spin
            layout.addLayout(row)

        layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))

        # Novo seletor para a variável multiplicadora
        layout.addWidget(QLabel("<b>Cálculo do Valor Total:</b>"))
        row_multi = QHBoxLayout()
        row_multi.addWidget(QLabel("Multiplicar valor unitário por:"))
        self.combo_multi = QComboBox()
        
        # Adiciona as colunas X (já exclui o Y por definição do x_columns)
        self.combo_multi.addItems(x_columns)
        
        # Tenta selecionar "Área" automaticamente se existir
        idx_area = self.combo_multi.findText("Área", Qt.MatchFlag.MatchContains)
        if idx_area >= 0:
            self.combo_multi.setCurrentIndex(idx_area)
            
        row_multi.addWidget(self.combo_multi)
        layout.addLayout(row_multi)

        btns = QHBoxLayout()
        btn_calc = QPushButton("Calcular")
        btn_calc.clicked.connect(self.accept)
        btns.addWidget(btn_calc)
        layout.addLayout(btns)

    def get_data(self):
        return {
            "valores": {col: spin.value() for col, spin in self.inputs.items()},
            "multiplicador_col": self.combo_multi.currentText()
        }

class DropTableWidget(QTableWidget):
    """Uma Tabela que aceita arrastar e soltar arquivos."""
    fileDropped = pyqtSignal(str) # Sinal que avisa quando um arquivo caiu aqui

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True) # Habilita o recebimento de 'drops'

    def dragEnterEvent(self, event):
        # Verifica se o que está sendo arrastado é um arquivo (URL)
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        # Captura o caminho do primeiro arquivo solto
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = str(url.toLocalFile())
            self.fileDropped.emit(file_path) # Dispara o sinal com o caminho do arquivo
            event.accept()
        else:
            event.ignore()

# ============================================================
# MainWindow
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Avaliação Imobiliária — AvalIm (MQO)")
        self.setMinimumSize(1100, 780)
        self.setStyleSheet(load_dark_style())

        # --- Atributos de Estado ---
        self.df = None
        self.model = None
        self.preco = None
        self.csv_path = None
        self.outliers_lim = 2.0
        self.R2_alvo = 0.75
        self._usar_limpo_flag = False
        self._limpo_ready = False
        self._limpo_ready_idx = None
        self.threads = []
        self.plot_windows = []
        self._current_fit_thread = None
        self._updating_model_spin = False
        self._plots_job_id = 0
        self._split_sizes_applied = False
        self._blocking_threads = set()
        self._ultima_amplitude = None

        self._build_menus()
        self._build_toolbar()

        # Widget Central e Layout Raiz
        container = QWidget()
        layout = QVBoxLayout(container)
        self.setCentralWidget(container)

        # --- LINHA 1: Controles Superiores (Arquivo e Seletor de Modelo) ---
        linha1 = QHBoxLayout()
        self.lbl_arquivo = QLabel("Arquivo: (nenhum)")
        self.lbl_arquivo.setToolTip("Arquivo de dados carregado.")
        linha1.addWidget(self.lbl_arquivo)

        linha1.addStretch()

        linha1.addWidget(QLabel("Modelo:"))
        self.spin_modelo = QSpinBox()
        self.spin_modelo.setMinimum(0)
        self.spin_modelo.setMaximum(0)
        self.spin_modelo.setEnabled(False)
        self.spin_modelo.setKeyboardTracking(False)
        self.spin_modelo.valueChanged.connect(self._on_model_spin_value_changed)
        linha1.addWidget(self.spin_modelo)
        layout.addLayout(linha1)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.lbl_status)

        # --- SPLITTER PRINCIPAL (DIVISÃO VERTICAL: ESQUERDA | DIREITA) ---
        self.split_main = QSplitter(Qt.Orientation.Horizontal)
        self.split_main.setHandleWidth(10)

        # ---------------------------------------------------------
        # 1. LADO ESQUERDO: Abas (Dados, Resultados, Tabela)
        # ---------------------------------------------------------
        self.scroll_left = QScrollArea()
        self.scroll_left.setWidgetResizable(True)
        self.scroll_left.setFrameShape(QFrame.Shape.NoFrame)
        
        self.tabs = QTabWidget()
        
        # Aba 1: Dados Brutos
        self.table_dados = DropTableWidget()
        self.table_dados.setSortingEnabled(True)
        
        # Conecta o sinal do 'drop' à função de carregar dados
        self.table_dados.fileDropped.connect(self.load_data_from_path)
        
        self.tabs.addTab(self.table_dados, "Dados")

        # Aba 2: Resultados (ESTILIZADA COM FONTE MONOESPAÇADA)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        # IMPORTANTE: Desativa quebra de linha para não bagunçar as tabelas
        self.log_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        # Configuração rigorosa da fonte para alinhamento de colunas
        mono = QFont("DejaVu Sans Mono", 10)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.log_box.setFont(mono)
        self.log_box.document().setDefaultFont(mono)
        self.log_box.setStyleSheet("""
            QPlainTextEdit {
                font-family: "DejaVu Sans Mono", "Courier New", monospace;
                font-size: 10pt;
            }
        """)
        self.tabs.addTab(self.log_box, "Resultados")

        # Aba 3: Tabela de Resultados
        self.table = QTableWidget()
        self.table.setSortingEnabled(True)
        self.tabs.addTab(self.table, "Tabela (Resultados)")

        self.scroll_left.setWidget(self.tabs)
        self.split_main.addWidget(self.scroll_left)

        # ---------------------------------------------------------
        # 2. LADO DIREITO: Pilha de Gráficos com Scroll
        # ---------------------------------------------------------
        self.scroll_right = QScrollArea()
        self.scroll_right.setWidgetResizable(True)
        self.scroll_right.setFrameShape(QFrame.Shape.NoFrame)
        
        self.plots_container = QWidget()
        self.plots_layout = QVBoxLayout(self.plots_container)
        self.plots_layout.setContentsMargins(5, 5, 5, 5)
        self.plots_layout.setSpacing(20)

        # Instanciação dos Painéis na ordem solicitada:
        self.panel_box = FigurePanel("box", "Boxplot", self._open_plot_from_panel)
        self.panel_graficos = FigurePanel("graficos", "Gráficos (Modelo)", self._open_plot_from_panel)
        self.panel_residuos = FigurePanel("residuos", "Resíduos Padronizados", self._open_plot_from_panel)
        # NOVO: Distância de Cook inserida após os Resíduos
        self.panel_cooks = FigurePanel("cooks", "Distância de Cook", self._open_plot_from_panel)
        self.panel_corr = FigurePanel("corr", "Matriz de Correlação", self._open_plot_from_panel)
        self.panel_aderencia = FigurePanel("aderencia", "Aderência", self._open_plot_from_panel)
        self.panel_hist = FigurePanel("hist", "Histograma", self._open_plot_from_panel)

        # Adiciona os painéis ao layout vertical da direita (Pilha)
        self.plots_layout.addWidget(self.panel_box)
        self.plots_layout.addWidget(self.panel_graficos)
        self.plots_layout.addWidget(self.panel_residuos)
        self.plots_layout.addWidget(self.panel_cooks)
        self.plots_layout.addWidget(self.panel_corr)
        self.plots_layout.addWidget(self.panel_aderencia)
        self.plots_layout.addWidget(self.panel_hist)

        self.scroll_right.setWidget(self.plots_container)
        self.split_main.addWidget(self.scroll_right)

        # Adiciona o Splitter ao layout principal
        layout.addWidget(self.split_main, 1)

        self._update_action_states()

    def showEvent(self, event):
        super().showEvent(event)
        if self._split_sizes_applied:
            return
        self._split_sizes_applied = True

        h = max(600, self.height())
        self.split_main.setSizes([int(h * 0.40), int(h * 0.60)])

        h2 = max(300, int(h * 0.60))
        self.split_rows.setSizes([int(h2 * 0.50), int(h2 * 0.50)])

        w = max(900, self.width())
        self.split_row_1.setSizes([w // 3, w // 3, w - 2 * (w // 3)])
        self.split_row_2.setSizes([w // 3, w // 3, w - 2 * (w // 3)])

    # ============================================================
    # MENU
    # ============================================================
    def _make_action(self, text, slot=None, shortcut=None, tip=None, checkable=False):
        act = QAction(text, self)
        if shortcut is not None:
            act.setShortcut(shortcut)
        if tip:
            act.setStatusTip(tip)
            act.setToolTip(tip)
        act.setCheckable(bool(checkable))
        if slot is not None:
            act.triggered.connect(slot)
        return act

    def _build_menus(self):
        mb = self.menuBar()

        m_file = mb.addMenu("&Arquivo")
        # Mantive o nome do método (load_csv) por compatibilidade, mas agora carrega vários formatos.
        self.act_load_csv = self._make_action("Carregar &Dados...", slot=self.load_csv, shortcut=QKeySequence.StandardKey.Open)
        self.act_export = self._make_action("Exportar Laudo PDF...", slot=self.exportar_laudo_pdf, shortcut=QKeySequence("Ctrl+Shift+E"))
        self.act_exit = self._make_action("Sa&ir", slot=self.close, shortcut=QKeySequence.StandardKey.Quit)
        m_file.addAction(self.act_load_csv)
        m_file.addAction(self.act_export)
        m_file.addSeparator()
        m_file.addAction(self.act_exit)

        m_model = mb.addMenu("&Modelo")
        self.act_set_preco = self._make_action("&Definir variável dependente...", slot=self.set_preco, shortcut=QKeySequence("Ctrl+D"))
        self.act_fit = self._make_action("&Calcular (Fit MQO)", slot=self.fit_model, shortcut=QKeySequence("F5"))
        self.act_cancel = self._make_action("&Cancelar execução", slot=self.cancel_current, shortcut=QKeySequence("Esc"))
        self.act_resultados = self._make_action("&Resultados", slot=self.resultados, shortcut=QKeySequence("Ctrl+T"))
        self.act_select_model = self._make_action("&Selecionar modelo (idx)...", slot=self.selecionar_modelo, shortcut=QKeySequence("Ctrl+Enter"))
        self.act_clean_outliers = self._make_action("Limpar &Outliers...", slot=self.menu_limpar_outliers, shortcut=QKeySequence("Ctrl+L"))
        self.act_use_clean = self._make_action(
            "Usar o modelo sem outliers?",
            slot=self.toggle_usar_limpo,
            shortcut=QKeySequence("Ctrl+U"),
            checkable=True
        )
        self.act_use_clean.setChecked(False)
        self.act_predict = self._make_action("Predizer Valor...", slot=self.run_predicao, shortcut=QKeySequence("Ctrl+P"))
        self.act_enquadrar = self._make_action("Enquadramento NBR...", slot=self.run_enquadramento)
        self.act_cooks = self._make_action("Distância de &Cook", slot=self.run_cooks, shortcut=QKeySequence("Ctrl+Shift+C"))
        
        m_model.addAction(self.act_set_preco)
        m_model.addSeparator()
        m_model.addAction(self.act_fit)
        m_model.addAction(self.act_cancel)
        m_model.addSeparator()
        m_model.addAction(self.act_resultados)
        m_model.addAction(self.act_select_model)
        m_model.addSeparator()
        m_model.addAction(self.act_clean_outliers)
        m_model.addAction(self.act_use_clean)
        m_model.addAction(self.act_predict)
        m_model.addAction(self.act_enquadrar)
        
        m_rep = mb.addMenu("&Relatórios")
        self.act_summary = self._make_action("&Resumo (summary)", slot=self.run_summary, shortcut=QKeySequence("Ctrl+M"))
        self.act_elast = self._make_action("&Elasticidades", slot=self.run_elasticidades, shortcut=QKeySequence("Ctrl+E"))
        self.act_dist_res = self._make_action("Dist. &Resíduos", slot=self.run_distribuicao_residuos, shortcut=QKeySequence("Ctrl+Shift+R"))
        m_rep.addAction(self.act_summary)
        m_rep.addAction(self.act_elast)
        m_rep.addSeparator()
        m_rep.addAction(self.act_dist_res)

        # m_plot = mb.addMenu("&Gráficos")
        # self.act_graficos = self._make_action("&Gráficos", slot=self.run_graficos, shortcut=QKeySequence("Ctrl+G"))
        # self.act_boxplot = self._make_action("&Boxplot", slot=self.run_boxplot, shortcut=QKeySequence("Ctrl+B"))
        # self.act_corr = self._make_action("Matriz de &Correlação", slot=self.run_matrix_corr, shortcut=QKeySequence("Ctrl+K"))
        # self.act_aderencia = self._make_action("&Aderência", slot=self.run_aderencia, shortcut=QKeySequence("Ctrl+A"))
        # self.act_residuos = self._make_action("Resíduos &Padronizados", slot=self.run_residuos, shortcut=QKeySequence("Ctrl+P"))
        # self.act_hist = self._make_action("&Histograma", slot=self.run_histograma, shortcut=QKeySequence("Ctrl+H"))

        # m_plot.addAction(self.act_boxplot)
        # m_plot.addAction(self.act_graficos)
        # m_plot.addAction(self.act_residuos)
        # m_plot.addSeparator()
        # m_plot.addAction(self.act_corr)
        # m_plot.addAction(self.act_aderencia)
        # m_plot.addSeparator()
        # m_plot.addAction(self.act_hist)
        # m_plot.addAction(self.act_cooks)

        m_tests = mb.addMenu("&Testes")
        self.act_shapiro = self._make_action("Normalidade &SW (Shapiro)", slot=self.run_shapiro, shortcut=QKeySequence("Ctrl+1"))
        self.act_kstest = self._make_action("Normalidade &KS", slot=self.run_kstest, shortcut=QKeySequence("Ctrl+2"))
        self.act_bp = self._make_action("&Heterocedasticidade (BP)", slot=self.run_bp, shortcut=QKeySequence("Ctrl+3"))
        self.act_dw = self._make_action("&Autocorrelação (DW)", slot=self.run_dw, shortcut=QKeySequence("Ctrl+4"))
        self.act_vif = self._make_action("&Multicolinearidade (VIF)", slot=self.run_vif, shortcut=QKeySequence("Ctrl+5"))
        m_tests.addAction(self.act_shapiro)
        m_tests.addAction(self.act_kstest)
        m_tests.addSeparator()
        m_tests.addAction(self.act_bp)
        m_tests.addAction(self.act_dw)
        m_tests.addSeparator()
        m_tests.addAction(self.act_vif)

        m_help = mb.addMenu("A&juda")

        def _about():
            QMessageBox.information(
                self,
                "AvalIm (MQO)",
                "AvalIm — MQO\n\n"
                "Split principal: Resultados/Tabela (40%) | Figuras (60%).\n"
                "Figuras: 2 linhas iguais; cada linha tem 3 colunas iguais (1,1,1).\n"
                "Duplo-clique em um painel abre a janela do gráfico (PlotWindow).\n\n"
                "Entrada aceita: CSV/TXT/TSV/TAB (delimitadores ';', ',', TAB) + Excel/ODS."
            )

        self.act_about = self._make_action("&Sobre", slot=_about, shortcut=QKeySequence("F1"))
        m_help.addAction(self.act_about)

    # ============================================================
    # Controle de threads bloqueantes (fit/outliers)
    # ============================================================
    def _register_blocking_thread(self, th):
        if th is None:
            return
        self._blocking_threads.add(th)
        try:
            th.finished.connect(lambda: self._on_blocking_thread_finished(th))
        except Exception:
            pass
        self._update_action_states()

    def _on_blocking_thread_finished(self, th):
        try:
            self._blocking_threads.discard(th)
        except Exception:
            pass

        if self._current_fit_thread is th:
            self._current_fit_thread = None

        self._update_action_states()

    # ============================================================
    # Estado de ações
    # ============================================================
    def _update_action_states(self):
        has_df = self.df is not None
        has_preco = bool(self.preco)
        has_model = self.model is not None
        has_any_fit = has_model and (getattr(self.model, "r2s", None) is not None) and (len(getattr(self.model, "r2s", [])) > 0)

        is_running = (self._current_fit_thread is not None) or (len(self._blocking_threads) > 0)

        # habilita "usar_limpo" somente se:
        # - já houve fit
        # - e o usuário rodou Limpar Outliers no idx atual
        cur_idx = self._current_idx()
        has_clean_ready = bool(
            has_any_fit and
            self._limpo_ready and
            (self._limpo_ready_idx is not None) and
            (cur_idx is not None) and
            int(self._limpo_ready_idx) == int(cur_idx)
        )

        self.act_load_csv.setEnabled(not is_running)
        self.act_set_preco.setEnabled(has_df and (not is_running))
        self.act_fit.setEnabled(has_df and has_preco and (not is_running))
        self.act_cancel.setEnabled(is_running)

        self.act_resultados.setEnabled(has_any_fit and (not is_running))
        self.act_select_model.setEnabled(has_any_fit and (not is_running))
        self.act_clean_outliers.setEnabled(has_any_fit and (not is_running))

        # act_use_clean: só depois do Limpar Outliers
        self.act_use_clean.setEnabled(has_clean_ready and (not is_running))
        if not has_clean_ready:
            self._usar_limpo_flag = False
            try:
                self.act_use_clean.blockSignals(True)
                self.act_use_clean.setChecked(False)
            finally:
                self.act_use_clean.blockSignals(False)

        for act in [
            self.act_summary, self.act_elast, self.act_dist_res,
            # self.act_boxplot, self.act_corr, self.act_graficos, self.act_aderencia,
            # self.act_residuos, self.act_hist,
            self.act_shapiro, self.act_kstest, self.act_bp, self.act_dw, self.act_vif
        ]:
            act.setEnabled(has_any_fit and (not is_running))

        if has_any_fit and (not is_running):
            n = self._num_modelos()
            cur = self._current_idx()
            if cur is None:
                cur = 0
            self._updating_model_spin = True
            try:
                self.spin_modelo.setEnabled(True)
                self.spin_modelo.setMaximum(max(0, n - 1))
                self.spin_modelo.setValue(max(0, min(int(cur), max(0, n - 1))))
            finally:
                self._updating_model_spin = False
        else:
            self._updating_model_spin = True
            try:
                self.spin_modelo.setEnabled(False)
                self.spin_modelo.setMinimum(0)
                self.spin_modelo.setMaximum(0)
                self.spin_modelo.setValue(0)
            finally:
                self._updating_model_spin = False

        is_running = (self._current_fit_thread is not None) or (len(self._blocking_threads) > 0)
        has_df = self.df is not None
        has_any_fit = self.model is not None and len(getattr(self.model, "r2s", [])) > 0

        if hasattr(self, 'btn_calc_tool'):
            if is_running:
                # MODO STOP: Execução ativa, habilitamos para cancelar
                icon_stop = self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserStop)
                self.btn_calc_tool.setIcon(icon_stop)
                self.btn_calc_tool.setText("Parar Execução")
                self.btn_calc_tool.setToolTip("Cancelar execução ativa (Esc)")
                self.btn_calc_tool.setEnabled(True) # Precisa estar habilitado para clicar no STOP
            else:
                # MODO PLAY: Nada rodando, pronto para calcular
                icon_play = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
                self.btn_calc_tool.setIcon(icon_play)
                self.btn_calc_tool.setText("Calcular (Fit MQO)")
                self.btn_calc_tool.setToolTip("Executar Cálculo MQO (F5)")
                # Habilitado apenas se houver dados carregados
                self.btn_calc_tool.setEnabled(has_df)

        if hasattr(self, 'btn_predict_tool'):
            # Lupa desabilitada enquanto calcula ou se não houver modelo
            self.btn_predict_tool.setEnabled(not is_running and has_any_fit)
        
        # Controle do botão de PDF na Toolbar
        if hasattr(self, 'btn_pdf_tool'):
            # Habilitado apenas se NÃO estiver calculando e se HOUVER um modelo pronto
            self.btn_pdf_tool.setEnabled(not is_running and has_any_fit)

    # ============================================================
    # Helpers gerais
    # ============================================================
    def log(self, msg):
        self.log_box.appendPlainText(str(msg))

    def log_sep(self, title=None, width=110, ch="─"):
        if title:
            s = f" {title} "
            n = max(0, width - len(s))
            left = n // 2
            right = n - left
            self.log(f"{ch * left}{s}{ch * right}")
        else:
            self.log(ch * width)

    def log_action(self, title):
        self.log_sep(title=title)

    def usar_limpo(self) -> bool:
        return bool(self._usar_limpo_flag)

    def progress_slot(self, atual, total, prefixo):
        if total <= 0:
            total = 1
        pct = int((atual / total) * 100)
        self.progress.setValue(pct)
        self.lbl_status.setText(f"{prefixo}: {pct}%")

    def _num_modelos(self) -> int:
        if not self.model:
            return 0
        modelos = getattr(self.model, "modelos", None)
        return int(len(modelos)) if modelos else 0

    def _current_idx(self) -> int | None:
        if not self.model:
            return None
        v = getattr(self.model, "_modelo_idx", None)
        return int(v) if v is not None else None

    def _transform_desc(self, idx: int) -> str:
        try:
            comb = self.model.combinacoes[int(idx)]
            cols = list(self.model.colunas)
            parts = []
            for t, col in zip(comb, cols):
                parts.append(self.model.transformada_print(int(t), col))
            return " | ".join(parts)
        except Exception:
            return ""

    def _run_if_exists(self, method_name: str, action_title: str, kwargs=None, callback=None):
        self.log_action(action_title)

        if not self.model:
            self.log("Nenhum modelo ajustado.")
            return

        fn = getattr(self.model, method_name, None)
        if fn is None or not callable(fn):
            self.log(f"Método '{method_name}' não disponível.")
            return

        self.threads.append(
            start_worker(fn, self.log, self.progress_slot, callback=callback, kwargs=kwargs, owner=self)
        )

    # ============================================================
    # PlotWindow por duplo clique / menu
    # ============================================================
    def _open_plot_from_panel(self, key: str, title: str):
        if not self.model:
            return
        try:
            usar_limpo = self.usar_limpo()
            if key == "box":
                fig = self.model.boxplot(usar_limpo=usar_limpo, show=False)
            elif key == "graficos":
                fig = self.model.graficos(usar_limpo=usar_limpo, show=False)
            elif key == "residuos":
                fig = self.model.residuos_grafico(usar_limpo=usar_limpo, show=False)
            elif key == "cooks":
                fig = self.model.cooks_distance_grafico(usar_limpo=usar_limpo, show=False)
            elif key == "corr":
                fig = self.model.matrix_corr(usar_limpo=usar_limpo, show=False)
            elif key == "aderencia":
                fig = self.model.aderencia(usar_limpo=usar_limpo, show=False)
            elif key == "hist":
                fig = self.model.histograma(usar_limpo=usar_limpo, show=False)
            else:
                return

            win = PlotWindow(fig, title=title)
            win.show()
            self.plot_windows.append(win)
        except Exception as e:
            self.log_action("Abrir gráfico")
            self.log(f"Erro ao abrir '{title}': {e}")

    def _open_plot_window(self, fig, title: str):
        if fig is None:
            self.log_action("Gráficos")
            self.log(f"Nenhuma figura para '{title}'.")
            return
        win = PlotWindow(fig, title=title)
        win.show()
        self.plot_windows.append(win)

    # ============================================================
    # Atualização dos 6 painéis (NO THREAD PRINCIPAL)
    # ============================================================
    def _refresh_plot_panels(self):
        if not self.model:
            return

        self._plots_job_id += 1
        job_id = self._plots_job_id
        self.lbl_status.setText("Atualizando pilha de gráficos...")

        QTimer.singleShot(0, lambda: self._make_all_plots_main(job_id))

    def _make_all_plots_main(self, job_id: int):
        if job_id != self._plots_job_id or not self.model:
            return

        usar_limpo = self.usar_limpo()
        figs = {}
        errs = {}

        def safe(name: str, call):
            try:
                figs[name] = call()
            except Exception as e:
                figs[name] = None
                errs[name] = str(e)

        # Geração de todos os gráficos, incluindo o novo de Cook
        safe("box", lambda: self.model.boxplot(usar_limpo=usar_limpo, show=False))
        safe("graficos", lambda: self.model.graficos(usar_limpo=usar_limpo, show=False))
        safe("residuos", lambda: self.model.residuos_grafico(usar_limpo=usar_limpo, show=False))
        safe("cooks", lambda: self.model.cooks_distance_grafico(usar_limpo=usar_limpo, show=False))
        safe("corr", lambda: self.model.matrix_corr(usar_limpo=usar_limpo, show=False))
        safe("aderencia", lambda: self.model.aderencia(usar_limpo=usar_limpo, show=False))
        safe("hist", lambda: self.model.histograma(usar_limpo=usar_limpo, show=False))

        # Atribuição aos painéis
        self.panel_box.set_figure(figs.get("box"))
        self.panel_graficos.set_figure(figs.get("graficos"))
        self.panel_residuos.set_figure(figs.get("residuos"))
        self.panel_cooks.set_figure(figs.get("cooks"))
        self.panel_corr.set_figure(figs.get("corr"))
        self.panel_aderencia.set_figure(figs.get("aderencia"))
        self.panel_hist.set_figure(figs.get("hist"))

        if errs:
            for k, msg in errs.items():
                self.log(f"⚠️ Erro no gráfico '{k}': {msg}")

        self.lbl_status.setText("Pilha de gráficos atualizada.")

    # ============================================================
    # SPINBOX "Modelo"
    # ============================================================
    def _on_model_spin_value_changed(self, value: int):
        if self._updating_model_spin:
            return
        if not self.spin_modelo.isEnabled():
            return
        if not self.model:
            return
        self._apply_model_idx(int(value), log_source="(spin: setas)")

    def _on_model_spin_commit(self):
        if self._updating_model_spin:
            return
        if not self.spin_modelo.isEnabled():
            return
        if not self.model:
            return
        self._apply_model_idx(int(self.spin_modelo.value()), log_source="(spin: commit)")

    def _apply_model_idx(self, idx: int, log_source: str = ""):
        if not self.model:
            return

        cur = self._current_idx()
        # Permite a execução se o índice mudar ou se for forçado (fim do Fit)
        if cur is not None and int(idx) == int(cur) and log_source != "(force)":
            return

        n = self._num_modelos()
        if n <= 0:
            return

        idx = int(idx)
        if idx < 0 or idx >= n:
            self.log_action("Selecionar modelo")
            self.log(f"Índice inválido: {idx} (faixa: 0..{n-1})")
            cur2 = cur if cur is not None else 0
            self._updating_model_spin = True
            try:
                self.spin_modelo.setValue(int(cur2))
            finally:
                self._updating_model_spin = False
            return

        try:
            self.model.modelo = idx

            # Reset de flags e interface
            self._usar_limpo_flag = False
            self._limpo_ready = False
            self._limpo_ready_idx = None
            try:
                self.act_use_clean.blockSignals(True)
                self.act_use_clean.setChecked(False)
            finally:
                self.act_use_clean.blockSignals(False)

            self.log_box.clear() # Opcional: limpa o log para focar no novo modelo
            self.log_action("Selecionar modelo")
            self.log(f"Modelo selecionado: {idx} {log_source if log_source != '(force)' else ''}".rstrip())

            # --- 1. GERAÇÃO DA EQUAÇÃO DE REGRESSÃO ---
            try:
                res = self.model.modelos[idx]
                params = res.params
                comb = self.model.combinacoes[idx]
                colunas = list(self.model.colunas)
                y_name = self.model.preco
                
                y_idx = colunas.index(y_name)
                y_form = self.model.transformada_print(int(comb[y_idx]), y_name)
                
                b0 = params.get('const', 0)
                equacao = f"{y_form} = {b0:.6f}"
                
                for i, col in enumerate(colunas):
                    if col == y_name: continue
                    if col in params:
                        val = params[col]
                        transf_idx = int(comb[i])
                        x_form = self.model.transformada_print(transf_idx, col)
                        sinal = " + " if val >= 0 else " - "
                        equacao += f"{sinal}{abs(val):.6f} * {x_form}"
                
                self.log_sep("EQUAÇÃO DE REGRESSÃO")
                self.log(equacao)
                self.log_sep()
            except Exception as e_eq:
                self.log(f"Aviso: Falha na equação: {e_eq}")

            # --- 2. GERAÇÃO DO RESUMO (SUMMARY) AUTOMÁTICO ---
            try:
                # Chamamos o método resumo do modelo diretamente
                # O parâmetro usar_limpo define se usamos a amostra com ou sem outliers
                summary_text = self.model.resumo(usar_limpo=self.usar_limpo())
                if summary_text:
                    self.log(summary_text)
                    self.log_sep()
            except Exception as e_sum:
                self.log(f"Aviso: Falha ao gerar o resumo estatístico: {e_sum}")

            self._update_action_states()
            self._refresh_plot_panels()

        except Exception as e:
            self.log_action("Selecionar modelo")
            self.log(f"Erro ao selecionar modelo {idx}: {e}")
            # ... (seu código de restauração do spinbox em caso de erro)

    def selecionar_modelo(self):
        self.log_action("Selecionar modelo")

        if not self.model:
            self.log("Nenhum modelo ajustado.")
            return

        n = self._num_modelos()
        if n <= 0:
            self.log("Nenhum modelo disponível.")
            return

        dlg = SelectModelDialog(self, max_idx=n - 1, current=(self._current_idx() or 0))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            idx = dlg.selected()
            cur = self._current_idx()
            if cur is not None and int(idx) == int(cur):
                return

            self._updating_model_spin = True
            try:
                self.spin_modelo.setValue(int(idx))
            finally:
                self._updating_model_spin = False

            self._apply_model_idx(int(idx), log_source="(menu)")

    # ============================================================
    # Toggle "usar modelo limpo"
    # ============================================================
    def toggle_usar_limpo(self, checked: bool):
        cur_idx = self._current_idx()
        ok_ready = bool(
            self._limpo_ready and
            (self._limpo_ready_idx is not None) and
            (cur_idx is not None) and
            int(self._limpo_ready_idx) == int(cur_idx)
        )
        if checked and not ok_ready:
            self.log_action("Usar modelo sem outliers?")
            self.log("Ação bloqueada: rode 'Modelo → Limpar Outliers' antes para este modelo.")
            self._usar_limpo_flag = False
            try:
                self.act_use_clean.blockSignals(True)
                self.act_use_clean.setChecked(False)
            finally:
                self.act_use_clean.blockSignals(False)
            self._update_action_states()
            return

        self._usar_limpo_flag = bool(checked)
        self.log_action("Usar modelo sem outliers?")
        self.log(f"usar_limpo = {self._usar_limpo_flag}")
        self._update_action_states()

        if self.model is not None and self._current_idx() is not None:
            self._refresh_plot_panels()

    # ============================================================
    # CANCELAR
    # ============================================================
    def _finish_fit_ui(self, status_text="Concluído."):
        self._current_fit_thread = None
        self.lbl_status.setText(status_text)
        self._update_action_states()

    def _on_fit_thread_finished(self):
        th = self._current_fit_thread
        if th is None:
            # mesmo assim garante reabilitar
            self._update_action_states()
            return

        try:
            cancelled = bool(th.isInterruptionRequested())
        except Exception:
            cancelled = False

        # sempre limpa referência e reabilita
        self._finish_fit_ui("Cancelado." if cancelled else "Concluído.")

    def cancel_current(self):
        self.log_action("Cancelar execução")

        active = set(self._blocking_threads)
        if self._current_fit_thread is not None:
            active.add(self._current_fit_thread)

        if not active:
            self.log("Nenhuma execução ativa para cancelar.")
            self._update_action_states()
            return

        try:
            for th in list(active):
                try:
                    th.requestInterruption()
                except Exception:
                    pass
            self.lbl_status.setText("Cancelando...")
            self.log("Sinal de cancelamento enviado (requestInterruption).")
        except Exception as e:
            self.log(f"Falha ao cancelar: {e}")

        self._update_action_states()

    # ============================================================
    # Leitura de arquivos (CSV/TXT/TSV + Excel/ODS)
    # ============================================================
    def _sanitize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpezas simples para evitar colunas lixo (Excel) e linhas/colunas vazias."""
        if df is None:
            return df

        # normaliza nomes
        try:
            df = df.copy()
            df.columns = [str(c).strip() for c in df.columns]
        except Exception:
            pass

        # remove colunas "Unnamed: ..." típicas do Excel
        try:
            drop_unnamed = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]
            if drop_unnamed:
                df = df.drop(columns=drop_unnamed)
        except Exception:
            pass

        # remove colunas totalmente vazias
        try:
            df = df.dropna(axis=1, how="all")
        except Exception:
            pass

        # remove linhas totalmente vazias
        try:
            df = df.dropna(axis=0, how="all")
        except Exception:
            pass

        return df

    def _read_excel_like(self, path: str) -> tuple[pd.DataFrame, str]:
        ext = os.path.splitext(path)[1].lower()

        if ext == ".ods":
            engine = "odf"
        elif ext in {".xlsx", ".xlsm"}:
            engine = "openpyxl"
        else:  # ".xls"
            engine = "xlrd"

        try:
            xls = pd.ExcelFile(path, engine=engine)
            sheet = xls.sheet_names[0] if xls.sheet_names else 0
            df = pd.read_excel(xls, sheet_name=sheet)
            return df, f"Planilha: {sheet} | formato: {ext} | engine={engine}"
        except ImportError as e:
            if ext in {".xlsx", ".xlsm"}:
                raise ImportError("Falha ao ler XLSX/XLSM (engine ausente). Instale: pip install openpyxl") from e
            if ext == ".xls":
                raise ImportError("Falha ao ler XLS (engine ausente). Instale: pip install xlrd") from e
            if ext == ".ods":
                raise ImportError("Falha ao ler ODS (engine ausente). Instale: pip install odfpy") from e
            raise
        except Exception as e:
            raise RuntimeError(f"Falha ao ler planilha: {e}") from e

    def _read_text_table(self, path: str) -> tuple[pd.DataFrame, str]:
        """
        Leitura robusta para CSV/TXT/TSV/TAB:
          - tenta autodetect do separador (engine python)
          - faz fallback em lista de separadores comuns
          - tenta decimal ',' e '.' (em ordens razoáveis)
        """
        # Primeiro: autodetect de separador
        attempts = []

        # (sep=None) tenta sniff automaticamente (engine python)
        attempts.append(dict(sep=None, engine="python", decimal=","))
        attempts.append(dict(sep=None, engine="python", decimal="."))

        # fallback explícito
        for sep in [";", "\t", ",", "|"]:
            # para ';' e '\t', decimal ',' é muito comum no Brasil
            attempts.append(dict(sep=sep, engine="python", decimal="," if sep in [";", "\t"] else "."))
            attempts.append(dict(sep=sep, engine="python", decimal="."))

        last_err = None
        for kw in attempts:
            try:
                df = pd.read_csv(
                    path,
                    encoding="utf-8-sig",
                    on_bad_lines="skip",
                    **kw
                )
                # Heurística: se caiu em 1 coluna só (e arquivo não é mesmo 1 coluna), tenta outra
                if df is not None and hasattr(df, "shape") and df.shape[1] == 1:
                    # se o conteúdo parece conter separadores (muitos ';' ou ','), provavelmente falhou
                    # => continua tentando
                    last_err = RuntimeError(f"Leitura gerou apenas 1 coluna (provável separador/decimal errado): {kw}")
                    continue

                info = f"Texto: sep={repr(kw.get('sep'))} | decimal={kw.get('decimal')} | engine={kw.get('engine')}"
                return df, info
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"Falha ao ler arquivo texto. Último erro: {last_err}")

    def _read_table_file(self, path: str) -> tuple[pd.DataFrame, str]:
        ext = os.path.splitext(path)[1].lower()

        if ext in {".xlsx", ".xls", ".xlsm", ".ods"}:
            df, info = self._read_excel_like(path)
        else:
            # trata como texto/CSV/TSV
            df, info = self._read_text_table(path)

        df = self._sanitize_df(df)
        return df, info

    # ============================================================
    # CARREGAR (mantive o nome load_csv por compatibilidade)
    # ============================================================
    def load_csv(self):
        filtros = (
            "Dados (*.csv *.txt *.tsv *.tab *.xls *.xlsx *.xlsm *.ods);;"
            "CSV/TXT/TSV (*.csv *.txt *.tsv *.tab);;"
            "Excel (*.xls *.xlsx *.xlsm);;"
            "ODS (*.ods);;"
            "Todos (*.*)"
        )
        path, _ = QFileDialog.getOpenFileName(self, "Carregar dados", "", filtros)
        if not path:
            return

        self.log_action("Carregar dados")

        ok = False
        try:
            df, info = self._read_table_file(path)
            self.df = df
            self.csv_path = path
            
            # Preenche a aba "Dados" imediatamente
            self._fill_table_from_df(self.table_dados, self.df)
            
            # Foca na aba de Dados para o usuário ver o que carregou
            self.tabs.setCurrentWidget(self.table_dados)
            
            self.preco = None
            self.model = None

            # reseta flags de limpo
            self._usar_limpo_flag = False
            self._limpo_ready = False
            self._limpo_ready_idx = None
            try:
                self.act_use_clean.blockSignals(True)
                self.act_use_clean.setChecked(False)
            finally:
                self.act_use_clean.blockSignals(False)

            base = os.path.basename(path)
            self.lbl_arquivo.setText(f"Arquivo: {base}")
            self.lbl_arquivo.setToolTip(path)

            # limpa painéis
            self.panel_box.set_figure(None)
            self.panel_graficos.set_figure(None)
            self.panel_residuos.set_figure(None)
            self.panel_corr.set_figure(None)
            self.panel_aderencia.set_figure(None)
            self.panel_hist.set_figure(None)

            # logs úteis
            self.log(f"Arquivo carregado: {path}")
            self.log(f"Leitura: {info}")
            try:
                self.log(f"Dimensões: {int(self.df.shape[0])} linhas × {int(self.df.shape[1])} colunas")
            except Exception:
                pass

            # alerta prático: Fit explode com muitas colunas
            try:
                ncols = int(self.df.shape[1])
                if ncols >= 10:
                    self.log("⚠️ Aviso: muitas colunas detectadas. O Fit testa 7^N combinações e pode ficar muito demorado.")
            except Exception:
                pass

            ok = True
        except Exception as e:
            self.log(f"Erro ao carregar arquivo: {e}")

        self._update_action_states()

        if ok:
            self.set_preco()

    # ============================================================
    # DEFINIR VARIÁVEL DEPENDENTE
    # ============================================================
    def _apply_preco(self, preco: str):
        self.log_action("Definir variável dependente")

        if self.df is None:
            self.log("Carregue o arquivo primeiro.")
            self._update_action_states()
            return

        preco = (preco or "").strip()
        if not preco:
            self.log("Seleção inválida.")
            self._update_action_states()
            return

        self.preco = preco
        self.log(f"Variável dependente: {self.preco}")
        self._update_action_states()

    def set_preco(self):
        if self.df is None:
            self.log_action("Definir variável dependente")
            self.log("Carregue o arquivo primeiro.")
            self._update_action_states()
            return

        dlg = DependentVarDialog(self, list(self.df.columns), current=self.preco)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._apply_preco(dlg.selected())

    # ============================================================
    # LIMPAR OUTLIERS (menu) — agora BLOQUEANTE
    # ============================================================
    def menu_limpar_outliers(self):
        self.log_action("Limpar outliers")

        if not self.model:
            self.log("Nenhum modelo ajustado.")
            return

        fn = getattr(self.model, "outliers_exc", None)
        if fn is None or not callable(fn):
            self.log("Método outliers_exc() não disponível.")
            return

        dlg = CleanOutliersDialog(self, r2_alvo=float(self.R2_alvo), lim_sigma=float(self.outliers_lim))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        r2_alvo, lim_sigma = dlg.values()
        self.R2_alvo = float(r2_alvo)
        self.outliers_lim = float(lim_sigma)

        try:
            setattr(self.model, "outliers_lim", float(self.outliers_lim))
        except Exception as e:
            self.log(f"Aviso: não consegui definir outliers_lim no modelo: {e}")

        self.log(f"Parâmetros: R2_alvo={self.R2_alvo:.3f} | outliers_lim={self.outliers_lim:.0f}σ")

        idx_at_start = self._current_idx()

        def after_outliers(_result):
            ok_same_idx = (
                self._current_idx() is not None and
                idx_at_start is not None and
                int(self._current_idx()) == int(idx_at_start)
            )
            has_clean = bool(
                ok_same_idx and
                self.model is not None and
                getattr(self.model, "amostra_limpa", None) is not None and
                getattr(self.model, "modelo_limpo", None) is not None
            )

            self._limpo_ready = bool(has_clean)
            self._limpo_ready_idx = int(idx_at_start) if has_clean else None

            # não marca automaticamente; só habilita o item
            self._usar_limpo_flag = False
            try:
                self.act_use_clean.blockSignals(True)
                self.act_use_clean.setChecked(False)
            finally:
                self.act_use_clean.blockSignals(False)

            self._update_action_states()
            if self.model is not None and self._current_idx() is not None:
                self._refresh_plot_panels()

        th = start_worker(
            fn,
            self.log,
            self.progress_slot,
            kwargs={"R2_alvo": float(self.R2_alvo)},
            callback=after_outliers,
            owner=self
        )
        self.threads.append(th)
        self._register_blocking_thread(th)

    # ============================================================
    # CALLBACK: pós-fit
    # ============================================================
    def _after_fit(self, _result):
        th = self._current_fit_thread
        if th is not None:
            try:
                if th.isInterruptionRequested():
                    self.log_action("Pós-fit")
                    self.log("Fit cancelado.")
                    self._update_action_states()
                    return
            except Exception:
                pass

        # Reset de flags e botões de interface
        self._usar_limpo_flag = False
        self._limpo_ready = False
        self._limpo_ready_idx = None
        try:
            self.act_use_clean.blockSignals(True)
            self.act_use_clean.setChecked(False)
        finally:
            self.act_use_clean.blockSignals(False)

        try:
            # 1. Foca na aba de Resultados
            self.tabs.setCurrentWidget(self.log_box)
            
            # 2. Preenche a tabela de resultados (aba Tabela)
            self.resultados()

            n = self._num_modelos()
            if n <= 0:
                self.log("Fit concluído, mas não há modelos disponíveis.")
                self._update_action_states()
                return

            # 3. Identifica o melhor modelo
            best_idx = self._current_idx() if self._current_idx() is not None else 0
            best_idx = max(0, min(int(best_idx), n - 1))

            # 4. EXECUTAR EQUAÇÃO E RESUMO (SUMMARY)
            # Esta função agora limpa o log e escreve a Equação + Summary automaticamente
            self._apply_model_idx(best_idx, log_source="(force)")

            # 5. ATUALIZAÇÃO APENAS VISUAL (Barra de Status e Progresso)
            # Removemos os logs de texto aqui para manter o log limpo
            self.lbl_status.setText(f"Concluído (Melhor modelo: {best_idx})")
            self.progress.setValue(100)

        except Exception as e:
            self.log(f"Erro no processamento pós-fit: {e}")

        self._update_action_states()

        # Atualiza a pilha de gráficos lateral
        if self.model is not None:
            self._refresh_plot_panels()

    # ============================================================
    # FIT — agora BLOQUEANTE
    # ============================================================
    def fit_model(self):
        self.log_box.clear()
        self.progress.setValue(0)
        self.lbl_status.setText("")

        self.log_action("Calcular (Fit MQO)")

        if self.df is None or self.preco is None:
            self.log("Carregue o arquivo e selecione a variável dependente (Menu → Modelo → Definir).")
            self._update_action_states()
            return

        # reseta flags de limpo antes do fit
        self._usar_limpo_flag = False
        self._limpo_ready = False
        self._limpo_ready_idx = None
        try:
            self.act_use_clean.blockSignals(True)
            self.act_use_clean.setChecked(False)
        finally:
            self.act_use_clean.blockSignals(False)

        # log preventivo: estimativa de combinações
        try:
            ncols = len(self.df.columns)
            total_est = 7 ** int(ncols)
            self.log(f"Colunas: {ncols} | combinações estimadas: 7^{ncols} = {total_est:,}".replace(",", "."))
        except Exception:
            pass

        try:
            # Pegamos apenas os dados que o usuário deixou "Ativos" na aba Dados
            df_selecionado = self._get_active_df()

            if df_selecionado.empty:
                self.log("❌ Erro: Nenhuma linha ativa selecionada na aba Dados.")
                self._update_action_states()
                return

            self.model = MQO(
                df_selecionado,
                preco=self.preco,
                gui_log=self.log,
                gui_progress=self.progress_slot,
                outliers_lim=float(self.outliers_lim),
            )
        except Exception as e:
            self.model = None
            self.log(f"Erro ao instanciar MQO: {e}")
            self._update_action_states()
            return

        self.log("Executando Fit MQO...")

        th = start_worker(self.model.fit, self.log, self.progress_slot, callback=self._after_fit, owner=self)
        self._current_fit_thread = th
        self.threads.append(th)

        # garante reabilitar menus ao terminar
        th.finished.connect(self._on_fit_thread_finished)

        # marca como bloqueante
        self._register_blocking_thread(th)

        self._update_action_states()

    # ============================================================
    # RESULTADOS
    # ============================================================
    def resultados(self):
        self.log_action("Resultados")

        if not self.model:
            self.log("Nenhum modelo ajustado.")
            return

        # if hasattr(self.model, "resultados") and callable(getattr(self.model, "resultados")):
        #     self.threads.append(start_worker(self.model.resultados, self.log, self.progress_slot, owner=self))
        # else:
        #     self.log("Método resultados() não disponível.")

        if hasattr(self.model, "resultados_tabela") and callable(getattr(self.model, "resultados_tabela")):
            self.threads.append(
                start_worker(
                    self.model.resultados_tabela,
                    self.log,
                    self.progress_slot,
                    callback=self._show_table_df,
                    kwargs={"qtd": 500},
                    owner=self
                )
            )
        else:
            self.log("Método resultados_tabela() não disponível.")

        self.tabs.setCurrentWidget(self.log_box)

    # ============================================================
    # MENU “Gráficos” — abre PlotWindow
    # ============================================================
    def run_graficos(self):
        self.log_action("Gráficos")
        if not self.model:
            self.log("Nenhum modelo ajustado.")
            return
        try:
            fig = self.model.graficos(usar_limpo=self.usar_limpo(), show=False)
            self._open_plot_window(fig, "Gráficos (Modelo)")
        except Exception as e:
            self.log(f"Erro em gráficos: {e}")

    def run_boxplot(self):
        self.log_action("Boxplot")
        if not self.model:
            self.log("Nenhum modelo ajustado.")
            return
        try:
            fig = self.model.boxplot(usar_limpo=self.usar_limpo(), show=False)
            self._open_plot_window(fig, "Boxplot")
        except Exception as e:
            self.log(f"Erro em boxplot: {e}")

    def run_matrix_corr(self):
        self.log_action("Matriz de Correlação")
        if not self.model:
            self.log("Nenhum modelo ajustado.")
            return
        try:
            fig = self.model.matrix_corr(usar_limpo=self.usar_limpo(), show=False)
            self._open_plot_window(fig, "Matriz de Correlação")
        except Exception as e:
            self.log(f"Erro em matriz de correlação: {e}")

    def run_aderencia(self):
        self.log_action("Aderência")
        if not self.model:
            self.log("Nenhum modelo ajustado.")
            return
        try:
            fig = self.model.aderencia(usar_limpo=self.usar_limpo(), show=False)
            self._open_plot_window(fig, "Aderência - MQO")
        except Exception as e:
            self.log(f"Erro em aderência: {e}")

    def run_residuos(self):
        self.log_action("Resíduos Padronizados")
        if not self.model:
            self.log("Nenhum modelo ajustado.")
            return
        try:
            fig = self.model.residuos_grafico(usar_limpo=self.usar_limpo(), show=False)
            self._open_plot_window(fig, "Resíduos Padronizados (MQO)")
        except Exception as e:
            self.log(f"Erro em resíduos: {e}")

    def run_histograma(self):
        self.log_action("Histograma")
        if not self.model:
            self.log("Nenhum modelo ajustado.")
            return
        try:
            fig = self.model.histograma(usar_limpo=self.usar_limpo(), show=False)
            self._open_plot_window(fig, "Histograma dos Resíduos")
        except Exception as e:
            self.log(f"Erro em histograma: {e}")

    def run_cooks(self):
        self.log_action("Distância de Cook")
        if not self.model:
            self.log("Nenhum modelo ajustado.")
            return
        try:
            fig = self.model.cooks_distance_grafico(usar_limpo=self.usar_limpo(), show=False)
            self._open_plot_window(fig, "Distância de Cook (Influência)")
        except Exception as e:
            self.log(f"Erro ao gerar Distância de Cook: {e}")
    # ============================================================
    # TESTES / TEXTOS — worker
    # ============================================================
    def run_shapiro(self):
        self._run_if_exists("teste_shapiro", "Normalidade (Shapiro-Wilk)", kwargs={"usar_limpo": self.usar_limpo()})

    def run_kstest(self):
        self._run_if_exists("teste_kstest", "Normalidade (Kolmogorov-Smirnov)", kwargs={"usar_limpo": self.usar_limpo()})

    def run_bp(self):
        self._run_if_exists("heterocedasticidade", "Heterocedasticidade (Breusch-Pagan)", kwargs={"usar_limpo": self.usar_limpo()})

    def run_dw(self):
        self._run_if_exists("autocorrelacao", "Autocorrelação (Durbin-Watson)", kwargs={"usar_limpo": self.usar_limpo()})

    def run_vif(self):
        self._run_if_exists("multicolinearidade", "Multicolinearidade (VIF)", kwargs={"usar_limpo": self.usar_limpo()})

    def run_distribuicao_residuos(self):
        self._run_if_exists("distribuicao_residuos", "Distribuição dos resíduos", kwargs={"usar_limpo": self.usar_limpo()})

    def _log_elasticidades(self, elast):
        self.log_action("Elasticidades")
        if elast is None:
            self.log("Elasticidades: nenhum resultado.")
            return
        try:
            for k, v in elast.items():
                self.log(f"- {k}: {v:.6f}")
        except Exception:
            self.log(str(elast))

    def run_elasticidades(self):
        if not self.model:
            self.log_action("Elasticidades")
            self.log("Nenhum modelo ajustado.")
            return
        fn = getattr(self.model, "elasticidades", None)
        if fn is None or not callable(fn):
            self.log_action("Elasticidades")
            self.log("Método elasticidades() não disponível.")
            return
        self.threads.append(
            start_worker(
                fn,
                self.log,
                self.progress_slot,
                callback=self._log_elasticidades,
                kwargs={"usar_limpo": self.usar_limpo()},
                owner=self
            )
        )

    def run_summary(self):
        self.log_action("Resumo")
        if not self.model:
            self.log("Nenhum modelo ajustado.")
            return
        fn = getattr(self.model, "resumo", None)
        if fn is None or not callable(fn):
            self.log("Método resumo() não disponível.")
            return

        def show_text(txt):
            if txt is None:
                self.log("Resumo vazio.")
                return
            self.log(txt)

        self.threads.append(
            start_worker(
                fn,
                self.log,
                self.progress_slot,
                callback=show_text,
                kwargs={"usar_limpo": self.usar_limpo()},
                owner=self
            )
        )

    # ============================================================
    # TABELA
    # ============================================================
    def _show_table_df(self, df: pd.DataFrame):
        if df is None or df.empty:
            self.log("Tabela vazia.")
            return

        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for r in range(len(df)):
            for c in range(len(df.columns)):
                self.table.setItem(r, c, QTableWidgetItem(str(df.iat[r, c])))

        self.table.resizeColumnsToContents()

    def run_predicao(self):
        if not self.model or self._current_idx() is None:
            QMessageBox.warning(self, "Aviso", "Ajuste o modelo primeiro!")
            return

        x_cols = [c for c in self.model.colunas if c != self.model.preco]
        
        # Passamos as colunas X e o nome do Y para o diálogo
        dlg = PredictionDialog(self, x_cols, self.model.preco)
        
        if dlg.exec() == QDialog.DialogCode.Accepted:
            try:
                data = dlg.get_data()
                valores_dict = data["valores"]
                col_multi = data["multiplicador_col"]
                
                # Fator multiplicador (ex: o valor da Área digitado)
                fator = valores_dict[col_multi]
                
                if fator <= 0:
                    QMessageBox.warning(self, "Aviso", f"O valor de '{col_multi}' deve ser maior que zero.")
                    return

                # Chama o modelo (que retorna valor UNITÁRIO)
                res = self.model.predicao_completa(valores_dict, usar_limpo=self.usar_limpo())
                
                # Cálculo da Amplitude (usando os valores unitários)
                amplitude = (res['ic_superior'] - res['ic_inferior']) / res['valor_pontual']
                self._ultima_amplitude = amplitude

                # Log de Resultados
                self.log_action("Predição de Valor")
                self.log(f"--- VALORES UNITÁRIOS ({self.model.preco}) ---")
                self.log(f"Unitário Estimado: R$ {res['valor_pontual']:,.2f}")
                self.log(f"I.C. Unitário: R$ {res['ic_inferior']:,.2f} a R$ {res['ic_superior']:,.2f}")

                self.log(f"\n--- VALORES TOTAIS (x {col_multi}: {fator:,.2f}) ---")
                val_total = res['valor_pontual'] * fator
                ic_inf_total = res['ic_inferior'] * fator
                ic_sup_total = res['ic_superior'] * fator
                arb_inf_total = res['arbitrio_inferior'] * fator
                arb_sup_total = res['arbitrio_superior'] * fator

                self.log(f"VALOR TOTAL ESTIMADO: R$ {val_total:,.2f}")
                self.log(f"I.C. Total: R$ {ic_inf_total:,.2f} a R$ {ic_sup_total:,.2f}")
                self.log(f"Campo de Arbítrio Total (±15%): R$ {arb_inf_total:,.2f} a R$ {arb_sup_total:,.2f}")

                # Enquadramento de Precisão
                info_p = self.model.enquadramento_nbr(usar_limpo=self.usar_limpo(), amplitude_percentual=amplitude)
                graus = ["Inidôneo", "I", "II", "III"]
                self.log(f"\nAmplitude: {amplitude*100:.2f}% (Grau de Precisão {graus[info_p['precisao']]})")

                # Alerta Visual com Valor Total
                QMessageBox.information(self, "Resultado Final", 
                    f"VALOR TOTAL: R$ {val_total:,.2f}\n"
                    f"Limite Inferior: R$ {ic_inf_total:,.2f}\n"
                    f"Limite Superior: R$ {ic_sup_total:,.2f}\n"
                    f"Precisão: Grau {graus[info_p['precisao']]}")
                
            except Exception as e:
                self.log(f"Erro no cálculo: {e}")

    def run_enquadramento(self):
        if not self.model or self._current_idx() is None:
            self.log("Ajuste o modelo primeiro.")
            return

        try:
            # Tenta usar a última amplitude se ela existir
            info = self.model.enquadramento_nbr(
                usar_limpo=self.usar_limpo(), 
                amplitude_percentual=self._ultima_amplitude
            )
            
            self.log_sep("ENQUADRAMENTO NBR 14653-2")
            self.log(f"1. Quantidade de Dados: Grau {info['grau_n']}")
            self.log(f"2. Significância Global (Modelo): Grau {info['grau_f']}")
            self.log(f"3. Significância dos Regressores: Grau {info['grau_p']}")
            
            status = ["Inidôneo", "I", "II", "III"]
            self.log_sep()
            self.log(f"GRAU DE FUNDAMENTAÇÃO: {status[info['fundamentacao']]}")
            
            # Se houver predição, mostra a precisão
            if "precisao" in info:
                self.log(f"GRAU DE PRECISÃO: {status[info['precisao']]} (Amplitude: {info['amplitude']:.2f}%)")
            else:
                self.log("GRAU DE PRECISÃO: Não calculado (realize uma Predição primeiro)")
            
            self.log_sep()

        except Exception as e:
            self.log(f"Erro ao enquadrar: {e}")

    def exportar_laudo_pdf(self):
        # Validação inicial
        if not self.model or self._current_idx() is None:
            QMessageBox.warning(self, "Erro", "Ajuste o modelo antes de exportar o laudo.")
            return

        # Diálogo para salvar o arquivo
        path, _ = QFileDialog.getSaveFileName(self, "Salvar Laudo", "Laudo_AvalIm.pdf", "PDF (*.pdf)")
        if not path:
            return

        try:
            from fpdf import FPDF
            import tempfile
            import os
            import platform
            import subprocess

            # --- FUNÇÃO AUXILIAR PARA ABRIR O PDF (Multiplataforma) ---
            def abrir_documento(caminho):
                sistema = platform.system()
                try:
                    if sistema == "Windows":
                        os.startfile(caminho)
                    elif sistema == "Darwin":  # macOS
                        subprocess.run(["open", caminho], check=True)
                    else:  # Linux
                        subprocess.run(["xdg-open", caminho], check=True)
                except Exception as e:
                    self.log(f"Laudo salvo, mas não foi possível abrir automaticamente: {e}")

            # 1. Preparar os dados para o laudo
            usar_limpo = self.usar_limpo()
            info_nbr = self.model.enquadramento_nbr(
                usar_limpo=usar_limpo, 
                amplitude_percentual=getattr(self, "_ultima_amplitude", None)
            )
            graus = ["Inidôneo", "I", "II", "III"]
            
            # 2. Gerar gráficos em pasta temporária
            with tempfile.TemporaryDirectory() as tmpdir:
                self.lbl_status.setText("Gerando gráficos para o PDF...")
                graficos_paths = self.model.salvar_todos_graficos(tmpdir, usar_limpo=usar_limpo)

                # 3. Construir o PDF
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                
                # Título Principal
                pdf.set_font("Helvetica", "B", 18)
                pdf.cell(0, 15, "RELATÓRIO DE AVALIAÇÃO IMOBILIÁRIA", ln=True, align="C")
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 5, "Gerado pelo Sistema AvalIm - MQO", ln=True, align="C")
                pdf.ln(10)

                # Seção 1: Identificação
                pdf.set_font("Helvetica", "B", 12)
                pdf.set_fill_color(240, 240, 240)
                pdf.cell(0, 10, " 1. INFORMAÇÕES DO MODELO", ln=True, fill=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.ln(2)
                pdf.cell(0, 7, f"Arquivo de Dados: {os.path.basename(self.csv_path)}", ln=True)
                pdf.cell(0, 7, f"Variável Dependente (Y): {self.model.preco}", ln=True)
                pdf.cell(0, 7, f"Tamanho da Amostra: {info_nbr['n']} dados", ln=True)
                pdf.cell(0, 7, f"Número de Variáveis Independentes: {info_nbr['k']}", ln=True)
                pdf.ln(5)

                # Seção 2: Enquadramento NBR 14653-2
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 10, " 2. ENQUADRAMENTO DA FUNDAMENTAÇÃO E PRECISÃO", ln=True, fill=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.ln(2)
                pdf.cell(0, 7, f"- Grau de Fundamentação: {graus[info_nbr['fundamentacao']]}", ln=True)
                
                if "precisao" in info_nbr:
                    pdf.cell(0, 7, f"- Grau de Precisão: {graus[info_nbr['precisao']]} (Amplitude: {info_nbr['amplitude']:.2f}%)", ln=True)
                else:
                    pdf.cell(0, 7, "- Grau de Precisão: Não calculado (requer predição)", ln=True)
                pdf.ln(5)

                # Seção 3: Sumário Estatístico
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 10, " 3. RESULTADOS DO MODELO (Summary)", ln=True, fill=True)
                pdf.ln(2)
                pdf.set_font("Courier", "", 8)
                texto_resumo = self.model.resumo(usar_limpo=usar_limpo)
                pdf.multi_cell(0, 4, texto_resumo)
                pdf.ln(5)

                # Seção 4: Anexos Gráficos (em novas páginas)
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 14)
                pdf.cell(0, 10, "4. ANEXOS GRÁFICOS", ln=True, align="C")
                pdf.ln(5)

                # Organiza 2 gráficos por página para melhor visualização
                y_pos = 30
                for i, (nome_grafico, g_path) in enumerate(graficos_paths.items()):
                    if i > 0 and i % 2 == 0:
                        pdf.add_page()
                        y_pos = 20
                    
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.cell(0, 10, f"Gráfico: {nome_grafico.upper()}", ln=True)
                    pdf.image(g_path, x=15, y=y_pos + 10, w=180)
                    y_pos += 125  # Pula para a metade de baixo ou próxima página

                # Gerar o arquivo final
                pdf.output(path)
                self.lbl_status.setText(f"Laudo exportado: {os.path.basename(path)}")
                
                QMessageBox.information(self, "Sucesso", f"O laudo foi gerado com sucesso em:\n{path}")
                
                # Abre o documento usando a lógica multiplataforma
                abrir_documento(path)

        except ImportError:
            QMessageBox.critical(self, "Erro", "Biblioteca 'fpdf2' não encontrada. Instale com: pip install fpdf2")
        except Exception as e:
            self.log(f"Erro ao exportar PDF: {e}")
            QMessageBox.critical(self, "Erro", f"Falha na exportação: {e}")

    def _fill_table_from_df(self, table_widget: QTableWidget, df: pd.DataFrame):
        if df is None:
            table_widget.setRowCount(0)
            table_widget.setColumnCount(0)
            return

        table_widget.setRowCount(len(df))
        # Adicionamos +1 coluna para o Checkbox de "Ativo"
        table_widget.setColumnCount(len(df.columns) + 1)
        
        headers = ["Ativo"] + [str(c) for c in df.columns]
        table_widget.setHorizontalHeaderLabels(headers)

        table_widget.blockSignals(True)
        for r in range(len(df)):
            # --- Inserir Checkbox na Coluna 0 ---
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            chk_item.setCheckState(Qt.CheckState.Checked) # Começa como ativo
            chk_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table_widget.setItem(r, 0, chk_item)

            # --- Preencher os dados das outras colunas ---
            for c in range(len(df.columns)):
                val = df.iat[r, c]
                text = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
                table_widget.setItem(r, c + 1, QTableWidgetItem(text))
        
        table_widget.blockSignals(False)
        table_widget.resizeColumnsToContents()

        # Conectar sinal para mudar a cor da linha quando desmarcar
        table_widget.itemChanged.connect(self._on_data_item_changed)

    def _on_data_item_changed(self, item):
        # Só nos interessa se a mudança foi na coluna 0 (Ativo)
        if item.column() != 0:
            return

        row = item.row()
        table = self.table_dados
        is_checked = (item.checkState() == Qt.CheckState.Checked)

        # Se desmarcado, pintamos a linha de cinza escuro
        color = Qt.GlobalColor.white if is_checked else Qt.GlobalColor.darkGray
        
        for col in range(table.columnCount()):
            cell = table.item(row, col)
            if cell:
                cell.setForeground(color)
                
    def _get_active_df(self) -> pd.DataFrame:
        """Varre a tabela de dados e retorna um DataFrame apenas com as linhas marcadas."""
        if self.df is None:
            return None

        active_indices = []
        for r in range(self.table_dados.rowCount()):
            item = self.table_dados.item(r, 0)
            # Verifica se o checkbox da primeira coluna está marcado
            if item and item.checkState() == Qt.CheckState.Checked:
                active_indices.append(r)

        if not active_indices:
            return pd.DataFrame() # Retorna vazio se nada estiver marcado

        # Filtra o DataFrame original usando os índices da tabela
        return self.df.iloc[active_indices].copy()
        
    def _on_data_item_changed(self, item):
        """Muda a cor da linha para indicar se o dado está ativo ou não."""
        if item.column() != 0: # Só nos interessa a coluna do Checkbox
            return

        row = item.row()
        is_active = (item.checkState() == Qt.CheckState.Checked)
        
        # Define a cor: Branco para ativo, Cinza para inativo
        color = Qt.GlobalColor.white if is_active else Qt.GlobalColor.gray
        
        for col in range(self.table_dados.columnCount()):
            cell = self.table_dados.item(row, col)
            if cell:
                cell.setForeground(color)

    def resizeEvent(self, event):
        """Força os gráficos a terem 50% da altura da janela para garantir o scroll."""
        super().resizeEvent(event)
        
        # Calcula a altura alvo (metade da área visível da janela)
        altura_alvo = self.height() // 2
        
        # Aplica a altura mínima a todos os painéis da pilha
        paineis = [
            self.panel_box, self.panel_graficos, self.panel_residuos,
            self.panel_cooks, self.panel_corr, self.panel_aderencia, self.panel_hist
        ]
        
        for p in paineis:
            p.setMinimumHeight(altura_alvo)

    def showEvent(self, event):
        """Define as proporções iniciais do splitter horizontal."""
        super().showEvent(event)
        if self._split_sizes_applied:
            return
        self._split_sizes_applied = True

        # Divide a tela ao meio verticalmente (50% Abas | 50% Gráficos)
        largura = self.width()
        self.split_main.setSizes([largura // 2, largura // 2])

    def load_data_from_path(self, path: str):
        """Função centralizada para processar o arquivo vindo de qualquer fonte."""
        if not path:
            return

        self.log_action("Carregar dados (Drag & Drop)")
        
        ok = False
        try:
            # Reutiliza sua lógica de leitura robusta
            df, info = self._read_table_file(path)
            self.df = df
            self.csv_path = path

            # Reseta os estados e limpa a interface (conforme seu código original)
            self.preco = None
            self.model = None
            self._usar_limpo_flag = False
            self._limpo_ready = False
            self._limpo_ready_idx = None
            
            # Preenche a tabela
            self._fill_table_from_df(self.table_dados, self.df)
            
            # Logs e UI
            self.lbl_arquivo.setText(f"Arquivo: {os.path.basename(path)}")
            self.log(f"Arquivo carregado via Drag & Drop: {path}")
            self.log(f"Leitura: {info}")
            
            ok = True
        except Exception as e:
            self.log(f"Erro ao processar arquivo: {e}")
            QMessageBox.critical(self, "Erro", f"Não foi possível ler o arquivo:\n{e}")

        self._update_action_states()
        if ok:
            self.set_preco() # Abre o diálogo da variável dependente

    def _build_toolbar(self):
        """Constrói a barra de ferramentas com ícones de acesso rápido."""
        toolbar = QToolBar("Barra de Ferramentas Principal")
        toolbar.setIconSize(QSize(32, 32)) # Ícones grandes e visíveis
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # 1. CARREGAR DADOS (+)
        # Usamos o ícone de 'Adicionar' ou 'Abrir' do sistema
        icon_add = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder)
        # Se você tiver um arquivo: icon_add = QIcon("caminho/para/mais.png")
        act_toolbar_load = toolbar.addAction(icon_add, "Carregar Dados")
        act_toolbar_load.triggered.connect(self.load_csv)
        act_toolbar_load.setToolTip("Carregar Dados (Arquivo ou Arrastar)")

        toolbar.addSeparator()

        # 2. CALCULAR (RAIO)
        # Usamos o ícone de 'Play' do sistema como padrão para 'Calcular'
        icon_calc = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        # Se você tiver um arquivo: icon_calc = QIcon("caminho/para/raio.png")
        self.btn_calc_tool = toolbar.addAction(icon_calc, "Calcular (Fit MQO)")
        self.btn_calc_tool.triggered.connect(self._handle_calc_action)
        self.btn_calc_tool.setToolTip("Executar Cálculo MQO (F5)")

        # 3. PREDIZER VALOR (LUPA)
        # Usamos o ícone de 'Busca/Lupa' do sistema
        icon_search = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView)
        # Se você tiver um arquivo: icon_search = QIcon("caminho/para/lupa.png")
        self.btn_predict_tool = toolbar.addAction(icon_search, "Predizer Valor")
        self.btn_predict_tool.triggered.connect(self.run_predicao)
        self.btn_predict_tool.setToolTip("Predição de Valor de Mercado (Ctrl+P)")

        # Adicionamos as referências à lista de ações para desabilitar durante o Fit
        # (Isso garante que o usuário não clique em calcular enquanto já está calculando)
        self._toolbar_actions = [self.btn_calc_tool, self.btn_predict_tool]
        
        # 4. EXPORTAR PDF (ÍCONE DE DISQUETE/PDF)
        # Ícone padrão de 'Salvar' do sistema. 
        # DICA: Se tiver um ícone próprio, use: QIcon("caminho/pdf_icon.png")
        icon_pdf = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        self.btn_pdf_tool = toolbar.addAction(icon_pdf, "Exportar Laudo PDF")
        self.btn_pdf_tool.triggered.connect(self.exportar_laudo_pdf)
        self.btn_pdf_tool.setToolTip("Exportar Laudo Completo em PDF (Ctrl+Shift+E)")

    def _handle_calc_action(self):
        """Decide se inicia o Fit ou se interrompe a execução atual."""
        is_running = (self._current_fit_thread is not None) or (len(self._blocking_threads) > 0)
        
        if is_running:
            self.cancel_current()
        else:
            self.fit_model()

