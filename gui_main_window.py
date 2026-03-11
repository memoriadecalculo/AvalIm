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
import locale
import matplotlib

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QPlainTextEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QMessageBox, QDialog, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QFrame, QSizePolicy, QSplitter,
    QScrollArea, QToolBar, QStyle, QCheckBox, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QAction, QKeySequence, QIcon, QColor, QValidator

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from gui_worker import start_worker
from gui_plot_window import PlotWindow
from gui_style import load_dark_style, load_light_style

from model import MQO

# ======================================================================
# FORÇA O PYTHON E O MATPLOTLIB A USAREM O PADRÃO DO SISTEMA OPERACIONAL
# ======================================================================
try:
    # Tenta definir a formatação numérica para o padrão do sistema (Brasil: vírgula decimal)
    locale.setlocale(locale.LC_NUMERIC, '')
    # Informa ao Matplotlib para usar essa configuração nos eixos dos gráficos
    matplotlib.rcParams['axes.formatter.use_locale'] = True
except Exception as e:
    print(f"Aviso: Não foi possível definir o locale numérico automaticamente: {e}")

class RankingSpinBox(QSpinBox):
    """SpinBox que navega no ranking R2 internamente usando valores negativos."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ranking = []

    def set_ranking(self, ranking_list):
        self._ranking = ranking_list

    def validate(self, text, pos):
        # O SEGREDO ESTÁ AQUI: Dizemos ao PyQt que o texto é válido, 
        # liberando o disparo automático do sinal 'valueChanged'.
        return QValidator.State.Acceptable, text, pos

    def textFromValue(self, val):
        rank_idx = -val 
        if not self._ranking or rank_idx < 0 or rank_idx >= len(self._ranking):
            return ""
        # Mostra APENAS o número bruto do modelo
        return str(self._ranking[rank_idx])

    def valueFromText(self, text):
        try:
            raw_idx = int(text)
            if raw_idx in self._ranking:
                rank_idx = self._ranking.index(raw_idx)
                return -rank_idx
        except ValueError:
            pass
        return self.value()

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
        import matplotlib.pyplot as plt # Adicione este import local
        
        old_fig = self._fig
        self._fig = fig

        if self._canvas is not None:
            self.layout().removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None
            
        # O SEGREDO CONTRA O SEGFAULT: Matar a figura antiga no C++
        if old_fig is not None:
            try:
                old_fig.clf()
                plt.close(old_fig)
            except Exception:
                pass

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

class SettingsDialog(QDialog):
    def __init__(self, parent, current_config):
        super().__init__(parent)
        self.setWindowTitle("Configurações do Sistema")
        self.setModal(True)
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        self.spins = {}

        # 1. Limite de Outliers (Desvios)
        row = QHBoxLayout()
        row.addWidget(QLabel("Limite de Outliers (Desvios Padrão):"))
        spin_lim = QDoubleSpinBox()
        spin_lim.setRange(1.0, 5.0)
        spin_lim.setSingleStep(0.1)
        spin_lim.setValue(current_config.get('outliers_lim', 2.0))
        self.spins['outliers_lim'] = spin_lim
        row.addWidget(spin_lim)
        layout.addLayout(row)

        # 2. R² Alvo (Para o Saneamento Automático)
        row = QHBoxLayout()
        row.addWidget(QLabel("R² Alvo (Saneamento Automático):"))
        spin_r2 = QDoubleSpinBox()
        spin_r2.setRange(0.0, 0.999)
        spin_r2.setSingleStep(0.01)
        spin_r2.setValue(current_config.get('r2_alvo', 0.75))
        self.spins['r2_alvo'] = spin_r2
        row.addWidget(spin_r2)
        layout.addLayout(row)

        # 3. Máximo de Outliers Removíveis (% da Amostra)
        row = QHBoxLayout()
        row.addWidget(QLabel("Máx. de Outliers Removíveis (%):"))
        spin_max_out = QDoubleSpinBox()
        spin_max_out.setRange(0.01, 0.50)
        spin_max_out.setSingleStep(0.01)
        spin_max_out.setValue(current_config.get('max_outliers_pct', 0.20))
        self.spins['max_outliers_pct'] = spin_max_out
        row.addWidget(spin_max_out)
        layout.addLayout(row)

        # 4. Máximo de Modelos na Tabela
        row = QHBoxLayout()
        row.addWidget(QLabel("Modelos gerados na Tabela (Resultados):"))
        spin_modelos = QSpinBox()
        spin_modelos.setRange(50, 5000)
        spin_modelos.setSingleStep(50)
        spin_modelos.setValue(current_config.get('max_modelos_tabela', 500))
        self.spins['max_modelos_tabela'] = spin_modelos
        row.addWidget(spin_modelos)
        layout.addLayout(row)

        layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))

        btns = QHBoxLayout()
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_cancel)

        btn_ok = QPushButton("Salvar Configurações")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btns.addWidget(btn_ok)

        layout.addLayout(btns)

    def get_config(self):
        return {k: v.value() for k, v in self.spins.items()}

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

class MetricCard(QFrame):
    """Card visual que agora funciona como um botão para ver detalhes."""
    clicked = pyqtSignal() # Novo sinal de clique

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor) # Vira a "mãozinha"
        self.setStyleSheet("""
            MetricCard {
                background-color: #f9f9f9;
                border: 1px solid #dddddd;
                border-radius: 6px;
                min-width: 125px;
            }
            MetricCard:hover { background-color: #f0f0f0; border-color: #bbb; }
        """)
        
        layout = QVBoxLayout(self)
        self.lbl_title = QLabel(label.upper())
        self.lbl_title.setStyleSheet("color: #666; font-size: 9px; font-weight: bold;")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_value = QLabel("-")
        self.lbl_value.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_value)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def set_value(self, text, color="#000"):
        self.lbl_value.setText(text)
        self.lbl_value.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")

class NumericItem(QTableWidgetItem):
    """Garante que a tabela formate o número visualmente, mas ordene matematicamente."""
    def __init__(self, val_num, val_str):
        super().__init__(val_str)
        self.val_num = float(val_num)
        self.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other):
        if hasattr(other, 'val_num'):
            return self.val_num < other.val_num
        return super().__lt__(other)

# ============================================================
# MainWindow
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Avaliação Imobiliária — AvalIm (MQO)")
        self.setMinimumSize(1100, 780)
        self.setStyleSheet(load_light_style())

        # --- Atributos de Estado ---
        self.df = None
        self.model = None
        self.preco = None
        self.csv_path = None
        
        # DICIONÁRIO CENTRAL DE CONFIGURAÇÕES
        self.config = {
            'outliers_lim': 2.0,
            'r2_alvo': 0.75,
            'max_outliers_pct': 0.20,
            'conv_lim': 0.5,
            'max_modelos_tabela': 500
        }
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

        # --- LINHA 1: Controles Superiores ---
        linha1 = QHBoxLayout()
        self.lbl_arquivo = QLabel("Arquivo: (nenhum)")
        self.lbl_arquivo.setToolTip("Arquivo de dados carregado.")
        linha1.addWidget(self.lbl_arquivo)

        linha1.addStretch()

        linha1.addWidget(QLabel("Modelo:"))
        self.spin_modelo = RankingSpinBox() # <-- USA A CLASSE NOVA
        self.spin_modelo.setMinimum(0)
        self.spin_modelo.setMaximum(0)
        self.spin_modelo.setEnabled(False)
        self.spin_modelo.setKeyboardTracking(False)
        self.spin_modelo.valueChanged.connect(self._on_model_spin_value_changed)
        linha1.addWidget(self.spin_modelo)
        
        self.chk_sem_outliers = QCheckBox("Sem outliers")
        self.chk_sem_outliers.setEnabled(False)
        self.chk_sem_outliers.setToolTip("Alternar visualização entre modelo bruto e saneado")
        self.chk_sem_outliers.toggled.connect(self._on_chk_outliers_toggled)
        linha1.addWidget(self.chk_sem_outliers)
        
        layout.addLayout(linha1)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.lbl_status)
        
        # --- DASHBOARD DE MÉTRICAS ---
        self.dash_layout = QHBoxLayout()
        self.dash_layout.setSpacing(10)
        self.dash_layout.setContentsMargins(0, 5, 0, 10)

        self.card_r2 = MetricCard("R²")
        self.card_r2_adj = MetricCard("R² Ajustado")
        self.card_fund = MetricCard("Fundamentação")
        self.card_outliers = MetricCard("Outliers")
        self.card_resid = MetricCard("Resíduos (68% 90% 95%)")
        self.card_norm = MetricCard("Normalidade (SW)")
        self.card_norm_ks = MetricCard("Normalidade (KS)")
        self.card_homoc = MetricCard("Homocedasticidade (BP)")
        self.card_auto = MetricCard("Autocorrelação (DW)")
        self.card_vif = MetricCard("Multicolinearidade")
        
        self.dash_layout.addWidget(self.card_r2)
        self.dash_layout.addWidget(self.card_r2_adj)
        self.dash_layout.addWidget(self.card_fund)
        self.dash_layout.addWidget(self.card_outliers)
        self.dash_layout.addWidget(self.card_resid)
        self.dash_layout.addWidget(self.card_norm)
        self.dash_layout.addWidget(self.card_norm_ks)
        self.dash_layout.addWidget(self.card_homoc)
        self.dash_layout.addWidget(self.card_auto)
        self.dash_layout.addWidget(self.card_vif)
        self.dash_layout.addStretch()

        layout.addLayout(self.dash_layout)
        
        # ==========================================================
        # CONEXÃO DOS CLIQUES NOS CARDS DO DASHBOARD
        # ==========================================================
        # 1. Cards básicos: Apenas levam o usuário para a aba de resultados
        self.card_r2.clicked.connect(self.run_summary)
        self.card_r2.clicked.connect(self.focus_results_tab)
        
        self.card_r2_adj.clicked.connect(self.run_summary)
        self.card_r2_adj.clicked.connect(self.focus_results_tab)
        
        self.card_fund.clicked.connect(self.run_enquadramento)
        self.card_fund.clicked.connect(self.focus_results_tab)
        
        self.card_outliers.clicked.connect(self.focus_results_tab)
        
        self.card_resid.clicked.connect(self.run_distribuicao_residuos)
        self.card_resid.clicked.connect(self.focus_results_tab)

        # 2. Cards de Testes Estatísticos: Executam o teste correspondente 
        # (e também garantem que a tela mude para a aba Resultados para ver o log)
        
        # Normalidade SW (Shapiro)
        self.card_norm.clicked.connect(self.run_shapiro)
        self.card_norm.clicked.connect(self.focus_results_tab)
        
        # Normalidade KS
        self.card_norm_ks.clicked.connect(self.run_kstest)
        self.card_norm_ks.clicked.connect(self.focus_results_tab)
        
        # Homocedasticidade (BP)
        self.card_homoc.clicked.connect(self.run_bp)
        self.card_homoc.clicked.connect(self.focus_results_tab)
        
        # Autocorrelação (DW)
        self.card_auto.clicked.connect(self.run_dw)
        self.card_auto.clicked.connect(self.focus_results_tab)
        
        # Multicolinearidade (VIF)
        self.card_vif.clicked.connect(self.run_vif)
        self.card_vif.clicked.connect(self.focus_results_tab)
        
        # --- SPLITTER PRINCIPAL ---
        self.split_main = QSplitter(Qt.Orientation.Horizontal)
        self.split_main.setHandleWidth(10)

        self.scroll_left = QScrollArea()
        self.scroll_left.setWidgetResizable(True)
        self.scroll_left.setFrameShape(QFrame.Shape.NoFrame)
        
        self.tabs = QTabWidget()
        
        self.table_dados = DropTableWidget()
        self.table_dados.setSortingEnabled(True)
        self.table_dados.fileDropped.connect(self.load_data_from_path)
        self.tabs.addTab(self.table_dados, "Dados")

        # Aba 2: Resultados (ESTILIZADA COM FONTE MONOESPAÇADA - PRESERVADO)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        mono = QFont("DejaVu Sans Mono", 10)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.log_box.setFont(mono)
        self.log_box.document().setDefaultFont(mono)
        self.log_box.setStyleSheet("""
            QPlainTextEdit {
                background-color: #ffffff;
                color: #000000;
                font-family: "DejaVu Sans Mono", "Courier New", monospace;
                font-size: 10pt;
                border: 1px solid #cccccc;
            }
        """)
        self.tabs.addTab(self.log_box, "Resultados")

        self.table = QTableWidget()
        self.table.setSortingEnabled(True)
        self.tabs.addTab(self.table, "Tabela (Resultados)")
        
        self.table_avaliandos = DropTableWidget()
        self.tabs.addTab(self.table_avaliandos, "Avaliandos")
        self.table_avaliandos.fileDropped.connect(self.load_avaliandos_from_path)
        
        self.scroll_left.setWidget(self.tabs)
        self.split_main.addWidget(self.scroll_left)

        self.scroll_right = QScrollArea()
        self.scroll_right.setWidgetResizable(True)
        self.scroll_right.setFrameShape(QFrame.Shape.NoFrame)
        
        self.plots_container = QWidget()
        self.plots_layout = QVBoxLayout(self.plots_container)
        self.plots_layout.setContentsMargins(5, 5, 5, 5)
        self.plots_layout.setSpacing(20)

        self.panel_box = FigurePanel("box", "Boxplot", self._open_plot_from_panel)
        self.panel_graficos = FigurePanel("graficos", "Gráficos (Modelo)", self._open_plot_from_panel)
        self.panel_residuos = FigurePanel("residuos", "Resíduos Padronizados", self._open_plot_from_panel)
        self.panel_cooks = FigurePanel("cooks", "Distância de Cook", self._open_plot_from_panel)
        self.panel_corr = FigurePanel("corr", "Matriz de Correlação", self._open_plot_from_panel)
        self.panel_aderencia = FigurePanel("aderencia", "Aderência", self._open_plot_from_panel)
        self.panel_hist = FigurePanel("hist", "Histograma", self._open_plot_from_panel)

        self.plots_layout.addWidget(self.panel_box)
        self.plots_layout.addWidget(self.panel_graficos)
        self.plots_layout.addWidget(self.panel_residuos)
        self.plots_layout.addWidget(self.panel_cooks)
        self.plots_layout.addWidget(self.panel_corr)
        self.plots_layout.addWidget(self.panel_aderencia)
        self.plots_layout.addWidget(self.panel_hist)

        self.scroll_right.setWidget(self.plots_container)
        self.split_main.addWidget(self.scroll_right)

        layout.addWidget(self.split_main, 1)

        self._update_action_states()

    def fmt_num(self, val, decimals=4):
        """Formata o número garantindo os separadores de milhar e decimal do Sistema."""
        from PyQt6.QtCore import QLocale
        try:
            loc = QLocale.system()
            dec_sep = loc.decimalPoint()
            
            # Gera a formatação americana padrão com milhar (ex: 1,234,567.89)
            txt = f"{float(val):,.{decimals}f}"
            
            # Se o sistema usar vírgula como decimal (Padrão BR), nós invertemos os símbolos
            if dec_sep == ',':
                txt = txt.replace(',', 'X').replace('.', ',').replace('X', '.')
                
            return txt
        except (ValueError, TypeError):
            return str(val)
            
    def fmt_summary(self, text):
        """Substitui os pontos por vírgulas nos relatórios do Statsmodels de forma segura."""
        from PyQt6.QtCore import QLocale
        if QLocale.system().decimalPoint() == ',':
            import re
            # Substitui o ponto por vírgula APENAS se estiver entre dois números
            text = re.sub(r'(\d)\.(\d)', r'\1,\2', str(text))
        return text
    
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

        # --- MENU ARQUIVO ---
        m_file = mb.addMenu("&Arquivo")
        
        # ---> ADICIONE ESTE BLOCO AQUI (NOVO PROJETO)
        self.act_novo = self._make_action(
            "&Novo Projeto", 
            slot=self.novo_projeto, 
            shortcut=QKeySequence("Ctrl+N"),
            tip="Limpar tudo e iniciar um novo projeto vazio"
        )
        m_file.addAction(self.act_novo)
        m_file.addSeparator()
        # <--- FIM DO BLOCO
        
        # ----------------------------------------------------
        # NOVOS BOTÕES: ABRIR E SALVAR PROJETO
        # ----------------------------------------------------
        self.act_abrir_proj = self._make_action(
            "A&brir Projeto (.mqo)...", 
            slot=self.abrir_projeto, 
            shortcut=QKeySequence("Ctrl+O")
        )
        
        self.act_salvar_proj = self._make_action(
            "&Salvar Projeto (.mqo)...", 
            slot=self.salvar_projeto, 
            shortcut=QKeySequence("Ctrl+S")
        )
        # ----------------------------------------------------
        
        # 1. Carregar Dados (Mercado)
        self.act_load_csv = self._make_action(
            "Carregar &Dados...", 
            slot=self.load_csv, 
            shortcut=QKeySequence.StandardKey.Open
        )
        
        # 2. Carregar Avaliandos (Alvos)
        self.act_load_avaliandos = self._make_action(
            "Carregar &Avaliandos...", 
            slot=self.load_avaliandos_csv, 
            shortcut=QKeySequence("Ctrl+Shift+A"),
            tip="Carregar planilha com os imóveis a serem avaliados"
        )
        
        # 3. Exportar
        self.act_export = self._make_action(
            "Exportar Laudo PDF...", 
            slot=self.exportar_laudo_pdf, 
            shortcut=QKeySequence("Ctrl+Shift+E")
        )
        
        # 4. Sair
        self.act_exit = self._make_action(
            "Sa&ir", 
            slot=self.close, 
            shortcut=QKeySequence.StandardKey.Quit
        )

        # Adicionando as ações ao menu na ordem lógica
        m_file.addAction(self.act_abrir_proj)
        m_file.addAction(self.act_salvar_proj)
        m_file.addSeparator()
        
        # ADICIONA NO MENU AQUI:
        m_file.addAction(self.act_load_csv)
        m_file.addAction(self.act_load_avaliandos)
        m_file.addSeparator()
        
        m_file.addAction(self.act_export)
        m_file.addSeparator()
        
        # Ação de Configurações
        self.act_settings = self._make_action(
            "C&onfigurações...", 
            slot=self.open_settings, 
            shortcut=QKeySequence("Ctrl+P")
        )
        m_file.addAction(self.act_settings)
        m_file.addSeparator()
        
        m_file.addAction(self.act_exit)

        # --- MENU MODELO ---
        m_model = mb.addMenu("&Modelo")
        self.act_set_preco = self._make_action("&Definir variável dependente...", slot=self.set_preco, shortcut=QKeySequence("Ctrl+D"))
        self.act_fit = self._make_action("&Calcular (Fit MQO)", slot=self.fit_model, shortcut=QKeySequence("F5"))
        self.act_cancel = self._make_action("&Cancelar execução", slot=self.cancel_current, shortcut=QKeySequence("Esc"))
        self.act_clean_outliers = self._make_action("Limpar &Outliers", slot=self.run_outliers_exc, shortcut=QKeySequence("Ctrl+L"))
        
        self.act_use_clean = self._make_action(
            "Usar o modelo sem outliers?",
            slot=self.toggle_usar_limpo,
            shortcut=QKeySequence("Ctrl+U"),
            checkable=True
        )
        self.act_use_clean.setChecked(False)
        
        self.act_cooks = self._make_action("Distância de &Cook", slot=self.run_cooks, shortcut=QKeySequence("Ctrl+Shift+C"))
        
        m_model.addAction(self.act_set_preco)
        m_model.addSeparator()
        m_model.addAction(self.act_fit)
        m_model.addAction(self.act_cancel)
        m_model.addSeparator()
        m_model.addAction(self.act_clean_outliers)
        m_model.addAction(self.act_use_clean)
        
        # --- MENU RELATÓRIOS ---
        m_rep = mb.addMenu("&Relatórios")
        self.act_summary = self._make_action("&Resumo (summary)", slot=self.run_summary, shortcut=QKeySequence("Ctrl+M"))
        self.act_elast = self._make_action("&Elasticidades", slot=self.run_elasticidades, shortcut=QKeySequence("Ctrl+E"))
        self.act_dist_res = self._make_action("Dist. &Resíduos", slot=self.run_distribuicao_residuos, shortcut=QKeySequence("Ctrl+Shift+R"))
        
        m_rep.addAction(self.act_summary)
        m_rep.addAction(self.act_elast)
        m_rep.addSeparator()
        m_rep.addAction(self.act_dist_res)

        # --- MENU TESTES ---
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

        # --- MENU AJUDA ---
        m_help = mb.addMenu("A&juda")

        def _about():
            QMessageBox.information(
                self,
                "AvalIm (MQO)",
                "AvalIm — MQO\n\n"
                "Software para avaliação de imóveis por regressão linear.\n"
                "Suporte a saneamento de amostra e enquadramento NBR 14653.\n\n"
                "Desenvolvido para Perícias Judiciais e Avaliações Urbanas."
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

        # Agora a interface sabe que desenhar gráficos e tabelas também é "trabalho em andamento"
        is_running = (self._current_fit_thread is not None) or \
                     (len(self._blocking_threads) > 0) or \
                     getattr(self, '_is_rendering_plots', False) or \
                     getattr(self, '_is_building_table', False)
        
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

        # -------------------------------------------------------------
        # BLOCO DO SPINBOX DE MODELOS (RANKING)
        # -------------------------------------------------------------
        if has_any_fit and (not is_running):
            ranking = self._get_model_ranking()
            cur = self._current_idx()
            
            # Descobre o índice (0-based) no ranking
            try:
                cur_rank = ranking.index(cur) if cur is not None else 0
            except ValueError:
                cur_rank = 0

            self._updating_model_spin = True
            try:
                self.spin_modelo.setEnabled(True)
                if hasattr(self.spin_modelo, 'set_ranking'):
                    self.spin_modelo.set_ranking(ranking)
                
                # O range vai do pior modelo (negativo) até o melhor (0)
                self.spin_modelo.setMinimum(-(len(ranking) - 1))
                self.spin_modelo.setMaximum(0)
                # Define o valor atual
                self.spin_modelo.setValue(-cur_rank)
            finally:
                self._updating_model_spin = False
        else:
            self._updating_model_spin = True
            try:
                self.spin_modelo.setEnabled(False)
                if hasattr(self.spin_modelo, 'set_ranking'):
                    self.spin_modelo.set_ranking([])
                self.spin_modelo.setMinimum(0)
                self.spin_modelo.setMaximum(0)
                self.spin_modelo.setValue(0)
            finally:
                self._updating_model_spin = False
        # -------------------------------------------------------------

        is_running = (self._current_fit_thread is not None) or (len(self._blocking_threads) > 0)
        
        # Verifica se o modelo limpo está pronto e pertence ao modelo atual
        limpo_disponivel = self._limpo_ready and (self._limpo_ready_idx == self._current_idx())

        if hasattr(self, 'chk_sem_outliers'):
            # Só habilita se não estiver calculando e o modelo limpo existir
            self.chk_sem_outliers.setEnabled(not is_running and limpo_disponivel)
            
            # Garante que a marcação visual da checkbox esteja correta (sem disparar sinais)
            self.chk_sem_outliers.blockSignals(True)
            self.chk_sem_outliers.setChecked(self._usar_limpo_flag)
            self.chk_sem_outliers.blockSignals(False)
        
        has_df = self.df is not None
        has_any_fit = self.model is not None and len(getattr(self.model, "r2s", [])) > 0
        
        # Controle do botão de Variável Dependente na Toolbar
        if hasattr(self, 'btn_dep_tool'):
            # Habilita se houver dados e não estiver no meio de um cálculo
            self.btn_dep_tool.setEnabled(has_df and not is_running)
        
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
        
        # Controle do botão Limpar Outliers na Toolbar
        if hasattr(self, 'btn_clean_tool'):
            # Só habilita se não estiver calculando e se houver um modelo para limpar
            self.btn_clean_tool.setEnabled(not is_running and has_any_fit)
        
        # Controle do botão de PDF na Toolbar
        if hasattr(self, 'btn_pdf_tool'):
            # Habilitado apenas se NÃO estiver calculando e se HOUVER um modelo pronto
            self.btn_pdf_tool.setEnabled(not is_running and has_any_fit)
        
        # Controle dos novos botões de Projeto na Toolbar
        if hasattr(self, 'btn_abrir_proj_tool'):
            # Pode abrir projeto a qualquer momento, desde que não esteja rodando cálculo
            self.btn_abrir_proj_tool.setEnabled(not is_running)
            
        if hasattr(self, 'btn_salvar_proj_tool'):
            # Só pode salvar se não estiver rodando cálculo E houver um modelo matemático pronto
            self.btn_salvar_proj_tool.setEnabled(not is_running and has_any_fit)

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
        # Se já houver algum texto na tela, dá um "Enter" antes de desenhar o novo separador
        if self.log_box.document().characterCount() > 1:
            self.log("")
            
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
            self._is_rendering_plots = False
            self._update_action_states()
            return

        usar_limpo = self.usar_limpo()
        
        plot_tasks = [
            ("Boxplot", "box", lambda: self.model.boxplot(usar_limpo=usar_limpo, show=False)),
            ("Gráficos de Ajuste", "graficos", lambda: self.model.graficos(usar_limpo=usar_limpo, show=False)),
            ("Resíduos Padronizados", "residuos", lambda: self.model.residuos_grafico(usar_limpo=usar_limpo, show=False)),
            ("Distância de Cook", "cooks", lambda: self.model.cooks_distance_grafico(usar_limpo=usar_limpo, show=False)),
            ("Correlação", "corr", lambda: self.model.matrix_corr(usar_limpo=usar_limpo, show=False)),
            ("Aderência", "aderencia", lambda: self.model.aderencia(usar_limpo=usar_limpo, show=False)),
            ("Histograma", "hist", lambda: self.model.histograma(usar_limpo=usar_limpo, show=False))
        ]

        panels = {
            "box": self.panel_box, "graficos": self.panel_graficos, 
            "residuos": self.panel_residuos, "cooks": self.panel_cooks,
            "corr": self.panel_corr, "aderencia": self.panel_aderencia, "hist": self.panel_hist
        }

        task_iter = iter(plot_tasks)
        total_plots = len(plot_tasks)
        plots_feitos = 0
        
        # Pega o progresso atual (deixado pelas equações) para calcular o passo
        prog_inicial = self.progress.value()
        passo = (100 - prog_inicial) / total_plots if total_plots > 0 else 0

        def _render_next():
            nonlocal plots_feitos
            if job_id != self._plots_job_id: return # Abortado por novo clique
            
            try:
                task_name, name, call = next(task_iter)
                
                # Avisa ao usuário qual gráfico está desenhando
                self.lbl_status.setText(f"Desenhando {task_name} ({plots_feitos+1}/{total_plots})...")
                from PyQt6.QtWidgets import QApplication
                QApplication.processEvents() 
                
                fig = call()
                panels[name].set_figure(fig)
                
                plots_feitos += 1
                self.progress.setValue(int(prog_inicial + (plots_feitos * passo)))
                
                QTimer.singleShot(10, _render_next)
                
            except StopIteration:
                self._is_rendering_plots = False
                
                # Se a Tabela Gigante já acabou de carregar, dizemos 100% Concluído
                if not getattr(self, '_is_building_table', False):
                    self.lbl_status.setText(f"Concluído (Modelo selecionado: {self._current_idx()})")
                    self.progress.setValue(100)
                else:
                    self.lbl_status.setText("Finalizando colunas da Tabela...")
                    
                self._update_action_states()
                
            except Exception as e:
                self.log(f"⚠️ Erro no gráfico '{name}': {e}")
                QTimer.singleShot(10, _render_next) 

        _render_next()

    # ============================================================
    # SPINBOX "Modelo"
    # ============================================================
    def _on_model_spin_value_changed(self, *args):
        # O getattr garante que não dará erro caso a flag ainda não exista
        if getattr(self, "_updating_model_spin", False):
            return
        if not self.spin_modelo.isEnabled():
            return
        if not self.model:
            return
            
        # Pega o valor numérico (negativo) diretamente do componente
        rank_idx = -self.spin_modelo.value()
        ranking = self._get_model_ranking()
        
        if 0 <= rank_idx < len(ranking):
            raw_idx = ranking[rank_idx]
            
            # Garante que só recalcula tudo se o modelo realmente mudou
            if self._current_idx() != raw_idx:
                self._apply_model_idx(raw_idx, log_source="(spin: setas)")

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
            # Deixe o update de estados corrigir o erro automaticamente
            self._update_action_states() 
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
            self.log("")
            
            from PyQt6.QtWidgets import QApplication
            
            # Trava a interface
            self._is_rendering_plots = True
            self._update_action_states()
            
            # Se veio do Fit, aproveita o progresso. Se veio de um clique do usuário, começa do 0.
            base_prog = self.progress.value() if log_source == "(force)" else 0

            # --- 1. GERAÇÃO DA EQUAÇÃO DE REGRESSÃO ---
            self.lbl_status.setText("Extraindo equações matemáticas...")
            self.progress.setValue(base_prog + 5)
            QApplication.processEvents()
            
            try:
                res = self.model.modelos[idx]
                params = res.params
                comb = self.model.combinacoes[idx]
                colunas = list(self.model.colunas)
                y_name = self.model.preco
                
                y_idx = colunas.index(y_name)
                y_form = self.model.transformada_print(int(comb[y_idx]), y_name)
                
                b0 = params.get('const', 0)
                equacao = f"{y_form} = {self.fmt_num(b0, 6)}"
                
                for i, col in enumerate(colunas):
                    if col == y_name: continue
                    if col in params:
                        val = params[col]
                        transf_idx = int(comb[i])
                        x_form = self.model.transformada_print(transf_idx, col)
                        sinal = " + " if val >= 0 else " - "
                        equacao += f"{sinal}{self.fmt_num(abs(val), 6)} * {x_form}"
                
                eq_reg, eq_est = self._get_equacoes_texto(idx)
                
                self.log_sep("EQUAÇÃO DE REGRESSÃO (Espaço Estatístico)")
                self.log(eq_reg)
                self.log("")
                self.log_sep("EQUAÇÃO ESTIMATIVA (Espaço de Valor)")
                self.log(eq_est)
                self.log("")
                self.log_sep()
                
            except Exception as e_eq:
                self.log(f"Aviso: Falha na equação: {e_eq}")

            # --- 2. GERAÇÃO DO RESUMO (SUMMARY) AUTOMÁTICO ---
            self.lbl_status.setText("Calculando sumário estatístico...")
            self.progress.setValue(base_prog + 10)
            QApplication.processEvents()
            try:
                summary_text = self.model.resumo(usar_limpo=self.usar_limpo())
                if summary_text:
                    # Aplica a máscara para formatar o quadro estatístico!
                    self.log(self.fmt_summary(summary_text))
                    self.log_sep()
                    self.log("")
            except Exception as e_sum:
                self.log(f"Aviso: Falha ao gerar o resumo estatístico: {e_sum}")
            
            # --- 3. BATERIA DE TESTES AUTOMÁTICOS EM SEQUÊNCIA ---
            self.lbl_status.setText("Executando bateria de testes...")
            self.progress.setValue(base_prog + 15)
            QApplication.processEvents()
            
            # self.run_shapiro()
            # self.run_kstest()
            # self.run_bp()
            # self.run_dw()
            # self.run_vif()
            # self.run_enquadramento()
            
            # --- 4. PREPARAR PAINÉIS VISUAIS ---
            self.lbl_status.setText("Atualizando dashboard...")
            self.progress.setValue(base_prog + 20)
            QApplication.processEvents()
            
            self._update_action_states()
            self._update_dashboard()
            self._refresh_plot_panels() # Isso inicia o loop final dos gráficos
            
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

        # O diálogo continua pedindo o índice original (bruto) do modelo
        dlg = SelectModelDialog(self, max_idx=n - 1, current=(self._current_idx() or 0))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            idx = dlg.selected()
            cur = self._current_idx()
            
            # Se o usuário escolheu o modelo que já está ativo, não fazemos nada
            if cur is not None and int(idx) == int(cur):
                return

            # Basta chamar a aplicação do modelo. 
            # A interface (incluindo a nova posição no RankingSpinBox) 
            # será atualizada automaticamente no final do processo via _update_action_states.
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
    
    def open_settings(self):
        self.log_action("Configurações")
        
        # Como o SettingsDialog não estava definido no arquivo original, 
        # ele já deve estar criado aí no seu código, chamamos ele normalmente:
        dlg = SettingsDialog(self, self.config)
        
        if dlg.exec() == QDialog.DialogCode.Accepted:
            novas_configs = dlg.get_config()
            self.config.update(novas_configs)
            self.log(f"Novas configurações salvas: {self.config}")
            
            # 1. Repassa as configurações para o núcleo matemático ativo
            if self.model:
                try:
                    self.model.outliers_lim = float(self.config['outliers_lim'])
                    # Se tiver outras variáveis no futuro que o modelo use na hora, atualize aqui
                except Exception as e:
                    self.log(f"Aviso ao atualizar limite no modelo ativo: {e}")
                    
            # 2. Atualiza APENAS a parte visual (Cards e Gráficos), 
            # pulando a trava de segurança do _apply_model_idx
            if self.model and self._current_idx() is not None:
                self.log("Aplicando novas configurações na interface (Cards e Gráficos)...")
                
                # Atualiza os Cards imediatamente (o Card de Outliers vai de 1 para 0)
                self._update_dashboard()
                
                # Inicia a fila de regeração dos gráficos nos painéis
                self._refresh_plot_panels()
    
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
    # CALLBACK: pós-fit
    # ============================================================
    def _after_fit(self, _result):
        th = self._current_fit_thread
        if th is not None:
            try:
                if th.isInterruptionRequested():
                    self._finish_fit_ui("Fit cancelado.")
                    self._update_action_states()
                    return
            except Exception:
                pass

        self._usar_limpo_flag = False
        self._limpo_ready = False
        self._limpo_ready_idx = None

        try:
            from PyQt6.QtWidgets import QApplication
            self.tabs.setCurrentWidget(self.log_box)

            n = self._num_modelos()
            if n <= 0:
                self.log("Fit concluído, mas não há modelos disponíveis.")
                self._finish_fit_ui("Nenhum modelo gerado.")
                return

            best_idx = self._current_idx() if self._current_idx() is not None else 0
            best_idx = max(0, min(int(best_idx), n - 1))

            # =========================================================
            # ORQUESTRANDO O PROGRESSO PÓS-FIT
            # =========================================================
            self.progress.setValue(50)
            self.lbl_status.setText("Montando Tabela de Resultados...")
            QApplication.processEvents() # Força a tela a atualizar AGORA

            # Fase 1: Dispara a Tabela Grande
            self._is_building_table = True
            self.resultados() 

            # Fase 2: Dispara a UI do Modelo (Textos, Testes e Gráficos)
            # O _apply_model_idx cuidará de levar o progresso de 50% a 100%
            self._apply_model_idx(best_idx, log_source="(force)")

        except Exception as e:
            self.log(f"Erro no processamento pós-fit: {e}")
            self._finish_fit_ui("Erro pós-fit.")

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
                outliers_lim=float(self.config['outliers_lim']), # <- AQUI
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
                    kwargs={"qtd": int(self.config.get('max_modelos_tabela', 500))},
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
            # Passa o texto do sumário pelo formatador inteligente que criamos
            self.log(self.fmt_summary(txt))

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
        self.log("")

    # ============================================================
    # TABELA
    # ============================================================
    def _show_table_df(self, df: pd.DataFrame):
        from PyQt6.QtCore import Qt
        
        # Guarda a tabela para poder ser salva no projeto (passo anterior)
        self._last_df_tabela = df  
        
        if df is None or df.empty:
            self.log("Tabela vazia.")
            return

        # ==========================================================
        # O SEGREDO DA VELOCIDADE: Desligar atualizações visuais e ordenação!
        # ==========================================================
        self.table.setSortingEnabled(False)
        self.table.setUpdatesEnabled(False)
        self.table.blockSignals(True)

        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for r in range(len(df)):
            for c in range(len(df.columns)):
                val = df.iat[r, c]
                if isinstance(val, (int, float)):
                    item = NumericItem(val, self.fmt_num(val, 6))
                else:
                    item = QTableWidgetItem(str(val))
                self.table.setItem(r, c, item)

        self.table.resizeColumnsToContents()

        # ==========================================================
        # RELIGAR TUDO APÓS O PREENCHIMENTO ESTAR COMPLETO
        # ==========================================================
        self.table.setSortingEnabled(True)
        self.table.setUpdatesEnabled(True)
        self.table.blockSignals(False)
        
        self._is_building_table = False
        self._update_action_states()

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
                self.log(f"Unitário Estimado: R$ {self.fmt_num(res['valor_pontual'], 2)}")
                self.log(f"I.C. Unitário: R$ {self.fmt_num(res['ic_inferior'], 2)} a R$ {self.fmt_num(res['ic_superior'], 2)}")

                self.log(f"\n--- VALORES TOTAIS (x {col_multi}: {self.fmt_num(fator, 2)}) ---")
                val_total = res['valor_pontual'] * fator
                ic_inf_total = res['ic_inferior'] * fator
                ic_sup_total = res['ic_superior'] * fator
                arb_inf_total = res['arbitrio_inferior'] * fator
                arb_sup_total = res['arbitrio_superior'] * fator

                self.log(f"VALOR TOTAL ESTIMADO: R$ {self.fmt_num(val_total, 2)}")
                self.log(f"I.C. Total: R$ {self.fmt_num(ic_inf_total, 2)} a R$ {self.fmt_num(ic_sup_total, 2)}")
                self.log(f"Campo de Arbítrio Total (±15%): R$ {self.fmt_num(arb_inf_total, 2)} a R$ {self.fmt_num(arb_sup_total, 2)}")

                # Enquadramento de Precisão
                info_p = self.model.enquadramento_nbr(usar_limpo=self.usar_limpo(), amplitude_percentual=amplitude)
                graus = ["Inidôneo", "I", "II", "III"]
                self.log(f"\nAmplitude: {self.fmt_num(amplitude*100, 2)}% (Grau de Precisão {graus[info_p['precisao']]})")

                # Alerta Visual com Valor Total
                QMessageBox.information(self, "Resultado Final", 
                    f"VALOR TOTAL: R$ {self.fmt_num(val_total, 2)}\n"
                    f"Limite Inferior: R$ {self.fmt_num(ic_inf_total, 2)}\n"
                    f"Limite Superior: R$ {self.fmt_num(ic_sup_total, 2)}\n"
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
            
            self.log_sep()

        except Exception as e:
            self.log(f"Erro ao enquadrar: {e}")

    def exportar_laudo_pdf(self):
        # 1. Validação inicial
        if not self.model or self._current_idx() is None:
            QMessageBox.warning(self, "Erro", "Ajuste o modelo antes de exportar o laudo.")
            return

        # 2. Diálogo para salvar o arquivo
        path, _ = QFileDialog.getSaveFileName(self, "Salvar Laudo", "Laudo_AvalIm.pdf", "PDF (*.pdf)")
        if not path:
            return

        try:
            from fpdf import FPDF
            import tempfile
            import os
            import platform
            import subprocess
            import numpy as np
            import pandas as pd
            import re

            def abrir_documento(caminho):
                sistema = platform.system()
                try:
                    if sistema == "Windows": os.startfile(caminho)
                    elif sistema == "Darwin": subprocess.run(["open", caminho], check=True)
                    else: subprocess.run(["xdg-open", caminho], check=True)
                except Exception as e:
                    self.log(f"Laudo salvo, mas erro ao abrir: {e}")

            # Preparar dados do Modelo e NBR
            usar_limpo = self.usar_limpo()
            info_nbr = self.model.enquadramento_nbr(
                usar_limpo=usar_limpo, 
                amplitude_percentual=getattr(self, "_ultima_amplitude", None)
            )
            graus = ["Inidôneo", "I", "II", "III"]
            
            # Identificar modelo ativo para extrair variáveis e outliers
            if usar_limpo and getattr(self.model, 'modelo_limpo', None) is not None:
                modelo_ativo = self.model.modelo_limpo
                amostra_ativa = self.model.amostra_limpa
            else:
                modelo_ativo = self.model.modelo
                amostra_ativa = self.model.amostra
                
            var_dep = self.model.preco
            vars_indep = [col for col in amostra_ativa.columns if col != var_dep]
            vars_str = ", ".join(vars_indep)
            n_vars = len(vars_indep)
            
            # Cálculo de Outliers Atuais
            influ = modelo_ativo.get_influence()
            resid = influ.resid_studentized_internal
            limite_outliers = float(self.config.get('outliers_lim', 2.0)) if hasattr(self, 'config') else 2.0
            qtd_outliers = int(np.sum(np.abs(resid) > limite_outliers))

            # ==========================================================
            # CAPTURA SILENCIOSA DOS TESTES (HOOK NO LOG DO FRAMEWORK)
            # ==========================================================
            old_log = self.model.gui_log
            old_gui_mode = self.model.gui_mode
            self.model.gui_mode = True # Força o modo de captura
            
            # Capturar Dist. Resíduos
            buffer_residuos = []
            def log_residuos(msg): buffer_residuos.append(re.sub(r'\033\[[0-9;]*m', '', str(msg))) # Limpa as cores ANSI
            self.model.gui_log = log_residuos
            try:
                self.model.distribuicao_residuos(usar_limpo=usar_limpo)
            except Exception as e:
                buffer_residuos.append(f"Erro na dist. de resíduos: {e}")
            txt_dist_residuos = "\n".join(buffer_residuos)

            # Capturar os 5 Testes Estatísticos
            buffer_testes = []
            def log_testes(msg):
                # Se for o DataFrame do VIF, imprime como tabela de texto
                if isinstance(msg, pd.DataFrame): buffer_testes.append(msg.to_string())
                else: buffer_testes.append(re.sub(r'\033\[[0-9;]*m', '', str(msg)))
            
            self.model.gui_log = log_testes
            try:
                self.model.teste_shapiro(usar_limpo=usar_limpo)
                self.model.teste_kstest(usar_limpo=usar_limpo)
                self.model.heterocedasticidade(usar_limpo=usar_limpo)
                self.model.autocorrelacao(usar_limpo=usar_limpo)
                self.model.multicolinearidade(usar_limpo=usar_limpo)
            except Exception as e:
                buffer_testes.append(f"Erro nos testes: {e}")
            txt_testes = "\n".join(buffer_testes)

            # Restaurar estado original do framework
            self.model.gui_log = old_log
            self.model.gui_mode = old_gui_mode

            # 3. Gerar gráficos em pasta temporária
            with tempfile.TemporaryDirectory() as tmpdir:
                self.lbl_status.setText("Gerando gráficos...")
                graficos_paths = self.model.salvar_todos_graficos(tmpdir, usar_limpo=usar_limpo)

                # 4. Construir o PDF
                pdf = FPDF()
                
                # --- REGISTRO DE FONTE UNICODE (OBRIGATÓRIO PARA MATEMÁTICA) ---
                font_name = 'DejaVuMono'
                try:
                    pdf.add_font(font_name, '', 'DejaVuSansMono.ttf')
                    if os.path.exists('DejaVuSansMono-Bold.ttf'): pdf.add_font(font_name, 'B', 'DejaVuSansMono-Bold.ttf')
                    else: pdf.add_font(font_name, 'B', 'DejaVuSansMono.ttf')
                    if os.path.exists('DejaVuSansMono-Oblique.ttf'): pdf.add_font(font_name, 'I', 'DejaVuSansMono-Oblique.ttf')
                    else: pdf.add_font(font_name, 'I', 'DejaVuSansMono.ttf')
                except Exception as e:
                    self.log("Erro: Arquivo 'DejaVuSansMono.ttf' não encontrado na pasta.")
                    QMessageBox.critical(
                        self, "Fonte Ausente", 
                        "Para imprimir equações (√, R²), o arquivo 'DejaVuSansMono.ttf' precisa estar na pasta."
                    )
                    return 

                pdf.set_auto_page_break(auto=True, margin=10)
                pdf.add_page()
                
                # Título Principal
                pdf.set_font(font_name, "B", 18)
                pdf.cell(0, 15, "RELATÓRIO DE AVALIAÇÃO IMOBILIÁRIA", ln=True, align="C")
                pdf.set_font(font_name, "I", 10)
                pdf.cell(0, 5, "Gerado pelo Sistema AvalIm - MQO", ln=True, align="C")
                pdf.ln(10)

                # ===================================================================
                # 1. INFORMAÇÕES DO MODELO
                # ===================================================================
                pdf.set_font(font_name, "B", 12)
                pdf.set_fill_color(240, 240, 240)
                pdf.cell(0, 10, " 1. INFORMAÇÕES", ln=True, fill=True)
                pdf.set_font(font_name, "", 10)
                pdf.ln(2)
                
                arquivo_nome = os.path.basename(self.csv_path) if hasattr(self, 'csv_path') and self.csv_path else "Não informado"
                pdf.cell(0, 7, f"Arquivo: {arquivo_nome}", ln=True)
                pdf.cell(0, 7, f"Variável Dependente (Y): {var_dep}", ln=True)
                pdf.cell(0, 7, f"Tamanho da Amostra: {info_nbr['n']} dados", ln=True)
                pdf.multi_cell(0, 7, f"Variáveis Independentes: {n_vars} ({vars_str})")
                pdf.ln(5)

                # ===================================================================
                # 2. EQUAÇÕES DO MODELO
                # ===================================================================
                pdf.set_font(font_name, "B", 12)
                pdf.cell(0, 10, " 2. EQUAÇÕES", ln=True, fill=True)
                pdf.set_font(font_name, "", 9)
                eq_reg, eq_est = self._get_equacoes_texto(self._current_idx())
                pdf.ln(2)
                pdf.set_text_color(100, 100, 100)
                pdf.multi_cell(0, 5, f"Regressão: {eq_reg}")
                pdf.ln(2)
                pdf.set_text_color(0, 0, 0)
                pdf.set_font(font_name, "B", 10)
                pdf.multi_cell(0, 5, f"Estimativa: {eq_est}")
                pdf.set_font(font_name, "", 10)
                pdf.ln(5)

                # ===================================================================
                # 3. RESULTADOS ESTATÍSTICOS (Summary, Outliers e Dist. Resíduos)
                # ===================================================================
                pdf.set_font(font_name, "B", 12)
                pdf.cell(0, 10, " 3. RESULTADOS ESTATÍSTICOS", ln=True, fill=True)
                pdf.ln(2)
                
                pdf.set_font(font_name, "", 8)
                texto_resumo = self.model.resumo(usar_limpo=usar_limpo)
                
                # ---> APLICA O FORMATADOR INTELIGENTE AQUI <---
                texto_resumo_formatado = self.fmt_summary(texto_resumo)
                
                pdf.multi_cell(0, 4, texto_resumo_formatado)
                pdf.ln(3)
                
                pdf.set_font(font_name, "B", 9)
                pdf.cell(0, 6, f"Quantidade de Outliers Encontrados (Limite {self.fmt_num(limite_outliers, 2)}σ): {qtd_outliers}", ln=True)
                
                pdf.set_font(font_name, "", 8)
                pdf.multi_cell(0, 4, txt_dist_residuos)
                pdf.ln(5)

                # ===================================================================
                # 4. TESTES ESTATÍSTICOS
                # ===================================================================
                # pdf.add_page() # Quebra de página para não cortar os testes ao meio
                pdf.add_page()
                pdf.set_font(font_name, "B", 12)
                pdf.cell(0, 10, " 4. TESTES ESTATÍSTICOS", ln=True, fill=True)
                pdf.ln(2)
                
                pdf.set_font(font_name, "", 8)
                pdf.multi_cell(0, 4, txt_testes)
                pdf.ln(5)

                # ===================================================================
                # 5. FUNDAMENTAÇÃO E PRECISÃO E AVALIANDOS
                # ===================================================================
                pdf.add_page()
                pdf.set_font(font_name, "B", 12)
                pdf.cell(0, 10, " 5. ENQUADRAMENTO (FUNDAMENTAÇÃO E PRECISÃO) E VALORES", ln=True, fill=True)
                pdf.set_font(font_name, "", 10)
                pdf.ln(2)
                
                pdf.cell(0, 7, f"- Grau de Fundamentação: {graus[info_nbr['fundamentacao']]}", ln=True)
                if "precisao" in info_nbr:
                    pdf.cell(0, 7, f"- Grau de Precisão: {graus[info_nbr['precisao']]} ({info_nbr['amplitude']:.2f}%)", ln=True)
                pdf.ln(5)
                
                if hasattr(self, 'table_avaliandos') and self.table_avaliandos.rowCount() > 0:
                    pdf.set_font(font_name, 'B', 10)
                    pdf.cell(0, 8, "Resultados das Previsões (Avaliandos):", ln=True)
                    
                    pdf.set_font(font_name, '', 8)
                    col_count = self.table_avaliandos.columnCount()
                    row_count = self.table_avaliandos.rowCount()
                    col_width = 190 / max(col_count, 1) 
                    
                    pdf.set_font(font_name, 'B', 8)
                    for c in range(col_count):
                        header_item = self.table_avaliandos.horizontalHeaderItem(c)
                        txt_header = header_item.text() if header_item else f"Col{c}"
                        pdf.cell(col_width, 6, txt_header[:15], border=1, align='C')
                    pdf.ln()
                    
                    pdf.set_font(font_name, '', 8)
                    for r in range(row_count):
                        for c in range(col_count):
                            item = self.table_avaliandos.item(r, c)
                            txt_cell = item.text() if item else ""
                            pdf.cell(col_width, 6, txt_cell[:15], border=1, align='C')
                        pdf.ln()
                else:
                    pdf.set_font(font_name, 'I', 10)
                    pdf.cell(0, 6, "Nenhum dado de Avaliandos foi calculado ou inserido.", ln=True)
                pdf.ln(5)

                # ===================================================================
                # 6. GRÁFICOS
                # ===================================================================
                pdf.add_page()
                pdf.set_font(font_name, "B", 14)
                pdf.cell(0, 10, "6. ANEXOS GRÁFICOS", ln=True, align="C")
                pdf.ln(5)

                y_pos = 30
                for i, (nome, g_path) in enumerate(graficos_paths.items()):
                    if i > 0 and i % 2 == 0:
                        pdf.add_page()
                        y_pos = 20
                    pdf.set_font(font_name, "B", 10)
                    pdf.cell(0, 10, f"Gráfico: {nome.upper()}", ln=True)
                    pdf.image(g_path, x=15, y=y_pos + 10, w=180)
                    y_pos += 125

                # Finalizar arquivo
                pdf.output(path)
                self.lbl_status.setText(f"Laudo exportado com sucesso!")
                QMessageBox.information(self, "Sucesso", "O laudo PDF foi gerado com sucesso!")
                abrir_documento(path)

        except Exception as e:
            self.log(f"Erro na exportação: {e}")
            QMessageBox.critical(self, "Erro", f"Falha na exportação:\n{e}")

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
                if isinstance(val, (float, int)):
                    item = NumericItem(val, self.fmt_num(val, 4))
                else:
                    item = QTableWidgetItem(str(val))
                table_widget.setItem(r, c + 1, item)
        
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
        
        for col in range(self.table_dados.columnCount()):
            cell = self.table_dados.item(row, col)
            if cell:
                if is_active:
                    # Remove qualquer cor forçada, restaurando a cor original do tema
                    cell.setData(Qt.ItemDataRole.ForegroundRole, None)
                else:
                    # Pinta de cinza quando a linha for desmarcada
                    cell.setForeground(Qt.GlobalColor.gray)

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
        
        # ==========================================================
        # 1. ABRIR E SALVAR PROJETO (.mqo)
        # ==========================================================
        # Ícone padrão de 'Abrir Pasta' do sistema
        icon_open_proj = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton)
        self.btn_abrir_proj_tool = toolbar.addAction(icon_open_proj, "Abrir Projeto (.mqo)")
        self.btn_abrir_proj_tool.triggered.connect(self.abrir_projeto)
        self.btn_abrir_proj_tool.setToolTip("Abrir um projeto salvo anteriormente (Ctrl+O)")

        # Ícone padrão de 'Disquete / Salvar' do sistema
        icon_save_proj = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        self.btn_salvar_proj_tool = toolbar.addAction(icon_save_proj, "Salvar Projeto (.mqo)")
        self.btn_salvar_proj_tool.triggered.connect(self.salvar_projeto)
        self.btn_salvar_proj_tool.setToolTip("Salvar o projeto atual (Ctrl+S)")
        
        toolbar.addSeparator() # <-- SEPARADOR SOLICITADO
        
        # CARREGAR DADOS (+)
        # Usamos o ícone de 'Adicionar' ou 'Abrir' do sistema
        icon_add = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder)
        # Se você tiver um arquivo: icon_add = QIcon("caminho/para/mais.png")
        act_toolbar_load = toolbar.addAction(icon_add, "Carregar Dados")
        act_toolbar_load.triggered.connect(self.load_csv)
        act_toolbar_load.setToolTip("Carregar Dados (Arquivo ou Arrastar)")
        
        # Botão para Carregar Avaliandos
        icon_av = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView)
        self.btn_load_av_tool = toolbar.addAction(icon_av, "Carregar Avaliandos")
        self.btn_load_av_tool.triggered.connect(self.load_avaliandos_csv)
        self.btn_load_av_tool.setToolTip("Carregar Planilha de Avaliandos (Ctrl+Shift+A)")
        
        toolbar.addSeparator()

        # DEFINIR VARIÁVEL DEPENDENTE (NOVO)
        # Ícone que remete a uma lista de escolha/detalhes
        icon_dep = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
        self.btn_dep_tool = toolbar.addAction(icon_dep, "Definir Variável Dependente (Y)")
        self.btn_dep_tool.triggered.connect(self.set_preco)
        self.btn_dep_tool.setToolTip("Escolher qual coluna é o Preço/Valor (Y)")
        
        # CALCULAR (RAIO)
        # Usamos o ícone de 'Play' do sistema como padrão para 'Calcular'
        icon_calc = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        # Se você tiver um arquivo: icon_calc = QIcon("caminho/para/raio.png")
        self.btn_calc_tool = toolbar.addAction(icon_calc, "Calcular (Fit MQO)")
        self.btn_calc_tool.triggered.connect(self._handle_calc_action)
        self.btn_calc_tool.setToolTip("Executar Cálculo MQO (F5)")
        
        # LIMPAR OUTLIERS (NOVO)
        # Ícone que remete a limpeza/saneamento
        icon_clean = self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        self.btn_clean_tool = toolbar.addAction(icon_clean, "Limpar Outliers")
        # Conectamos à sua função de execução de exclusão
        self.btn_clean_tool.triggered.connect(self.run_outliers_exc)
        self.btn_clean_tool.setToolTip("Remover outliers automaticamente para melhorar o R²")
        
        toolbar.addSeparator()
        
        # 4. EXPORTAR PDF (ÍCONE DE DISQUETE/PDF)
        # Ícone padrão de 'Salvar' do sistema. 
        # DICA: Se tiver um ícone próprio, use: QIcon("caminho/pdf_icon.png")
        icon_pdf = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        self.btn_pdf_tool = toolbar.addAction(icon_pdf, "Exportar Laudo PDF")
        self.btn_pdf_tool.triggered.connect(self.exportar_laudo_pdf)
        self.btn_pdf_tool.setToolTip("Exportar Laudo Completo em PDF (Ctrl+Shift+E)")
        
        # CONFIGURAÇÕES (NOVO BOTÃO NO FIM DA BARRA)
        toolbar.addSeparator()
        
        # Tenta carregar uma engrenagem personalizada (basta ter um arquivo 'gear.png' na pasta)
        import os
        from PyQt6.QtGui import QIcon
        
        if os.path.exists("gear.png"):
            icon_settings = QIcon("gear.png")
        else:
            # Ícone "estepe" nativo enquanto você não baixa um PNG de engrenagem
            icon_settings = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.btn_settings_tool = toolbar.addAction(icon_settings, "Configurações")
        self.btn_settings_tool.triggered.connect(self.open_settings)
        self.btn_settings_tool.setToolTip("Abrir painel de configurações (Ctrl+P)")
        
        # Adicionamos as referências à lista de ações para desabilitar durante o Fit
        # (Isso garante que o usuário não clique em calcular enquanto já está calculando)
        self._toolbar_actions = [
            self.btn_load_av_tool,
            self.btn_dep_tool,
            self.btn_calc_tool, 
            self.btn_clean_tool,
            self.btn_pdf_tool,
            self.btn_settings_tool
        ]
        
    def _handle_calc_action(self):
        """Decide se inicia o Fit ou se interrompe a execução atual."""
        is_running = (self._current_fit_thread is not None) or (len(self._blocking_threads) > 0)
        
        if is_running:
            self.cancel_current()
        else:
            self.fit_model()

    def _update_dashboard(self):
        if not self.model or self._current_idx() is None:
            return

        try:
            usar_limpo = self.usar_limpo()
            idx = self._current_idx()
            
            # Seleção do modelo ativo
            if usar_limpo and self.model.modelo_limpo:
                res = self.model.modelo_limpo
            else:
                res = self.model.modelos[idx]

            if res is None: return

            # Métricas de Ajuste
            r2 = res.rsquared
            self.card_r2.set_value(self.fmt_num(r2, 4), "#4CAF50" if r2 >= 0.75 else "#FFC107" if r2 >= 0.50 else "#F44336")
            
            r2_adj = res.rsquared_adj
            self.card_r2_adj.set_value(self.fmt_num(r2_adj, 4), "#4CAF50" if r2_adj >= 0.75 else "#FFC107" if r2_adj >= 0.50 else "#F44336")
            
            # Testes Estatísticos (Retorno Booleano)
            try:
                # 1. Normalidade (Shapiro-Wilk)
                is_sw = self.model.check_normalidade(idx, usar_limpo)
                self.card_norm.set_value("✔" if is_sw else "✘", "#4CAF50" if is_sw else "#F44336")
                
                # 2. Normalidade (Kolmogorov-Smirnov)
                is_ks = self.model.check_normalidade_ks(idx, usar_limpo)
                self.card_norm_ks.set_value("✔" if is_ks else "✘", "#4CAF50" if is_ks else "#F44336")
                
                # 3. Homocedasticidade (Breusch-Pagan)
                is_homo = self.model.check_homocedasticidade(idx, usar_limpo)
                self.card_homoc.set_value("✔" if is_homo else "✘", "#4CAF50" if is_homo else "#F44336")
                
                # 4. Autocorrelação (Durbin-Watson)
                is_auto = self.model.check_autocorrelacao(idx, usar_limpo)
                self.card_auto.set_value("✔" if is_auto else "✘", "#4CAF50" if is_auto else "#F44336")
                
                # 5. Multicolinearidade (VIF)
                is_vif = self.model.check_multicolinearidade(idx, usar_limpo)
                self.card_vif.set_value("✔" if is_vif else "✘", "#4CAF50" if is_vif else "#F44336")

            except Exception as e_tests:
                print(f"Erro nos testes do Dashboard: {e_tests}")

            # NBR 14653 (Fundamentação)
            info_nbr = self.model.enquadramento_nbr(usar_limpo=usar_limpo, amplitude_percentual=getattr(self, "_ultima_amplitude", None))
            graus = ["Inidôneo", "I", "II", "III"]
            g_fund = info_nbr['fundamentacao']
            self.card_fund.set_value(graus[g_fund], "#4CAF50" if g_fund >= 2 else "#FFC107" if g_fund == 1 else "#F44336")
            
            # --- LÓGICA DO NOVO CARD: OUTLIERS ---
            n_outliers = self.model.outliers_qtd(idx, usar_limpo=usar_limpo)
            n_total = int(res.nobs)
            pct_out = (n_outliers / n_total) if n_total > 0 else 0
            
            if n_outliers == 0:
                cor_out = "#4CAF50" # Verde: Perfeito
            elif pct_out <= 0.05:
                cor_out = "#FFC107" # Amarelo: Até 5% (Aceitável)
            else:
                cor_out = "#F44336" # Vermelho: Acima de 5% (Crítico)
            
            self.card_outliers.set_value(str(n_outliers), cor_out)
            
            # --- LÓGICA DO CARD: RESÍDUOS ---
            stats_resid = self.model.get_dist_residuos_stats(usar_limpo=usar_limpo)
            if stats_resid:
                p1, p164, p196 = stats_resid
                # Passa a usar o fmt_num!
                txt_resid = f"{self.fmt_num(p1*100, 0)}% {self.fmt_num(p164*100, 0)}% {self.fmt_num(p196*100, 0)}%"
                
                diff = abs(p1-0.68) + abs(p164-0.90) + abs(p196-0.95)
                cor_resid = "#4CAF50" if diff < 0.15 else "#FFC107" if diff < 0.30 else "#F44336"
                
                self.card_resid.set_value(txt_resid, cor_resid)
            
        except Exception as e:
            print(f"Erro ao atualizar dashboard: {e}")
        
        if hasattr(self, 'table_avaliandos') and self.table_avaliandos.rowCount() > 0:
            self._update_avaliandos_predictions()
        
    def run_outliers_exc(self):
        """Inicia o processo de exclusão iterativa de outliers via Thread."""
        if not self.model:
            self.log("Nenhum modelo ajustado para limpar.")
            return

        self.log_action("Limpar Outliers (Automático)")
        
        # 1. Garante que o modelo está usando o limite de desvios atualizado nas Configurações
        try:
            self.model.outliers_lim = float(self.config['outliers_lim'])
        except Exception as e:
            self.log(f"Aviso ao definir outliers_lim no modelo: {e}")

        # 2. Informa ao usuário os parâmetros que estão sendo usados (baseado nas configurações)
        self.log(f"Parâmetros ativos: R² Alvo = {self.config['r2_alvo']} | Limite = {self.config['outliers_lim']}σ | Máx Remoção = {self.config['max_outliers_pct']*100}%")
        self.lbl_status.setText("Saneando amostra...")
        
        # 3. Dispara a Thread de limpeza
        th = start_worker(
            self.model.outliers_exc,
            self.log,
            self.progress_slot,
            callback=self._on_outliers_finished,
            kwargs={
                "R2_alvo": self.config['r2_alvo'], 
                "out_lim": self.config['max_outliers_pct'],   
                "conv_lim": self.config.get('conv_lim', 0.5)  
            },
            owner=self
        )
        self.threads.append(th)
        self._register_blocking_thread(th)

    def _on_outliers_finished(self, result):
        """Atualiza a interface após a remoção dos outliers."""
        if not result:
            self.lbl_status.setText("Limpeza de outliers falhou ou foi cancelada.")
            self._update_action_states() # Garante reabilitação mesmo em falha
            return

        # Desempacota o retorno NOVO do model.py (agora com 5 variáveis)
        amostra_limpa, modelo_limpo, amostra_limpa_orig, linhas_removidas, resolvidos = result

        # Log inteligente detalhando o fenômeno matemático
        self.log(f"Saneamento concluído. {linhas_removidas} linha(s) removida(s) resolveram {resolvidos} outlier(s).")
        self.lbl_status.setText(f"Concluído: {linhas_removidas} linhas removidas.")

        self._limpo_ready = True
        self._limpo_ready_idx = self._current_idx()
        
        self._usar_limpo_flag = True
        if hasattr(self, 'act_use_clean'):
            self.act_use_clean.setChecked(True)

        self._update_dashboard()
        self._refresh_plot_panels()
        
        self._apply_model_idx(self._current_idx(), log_source="(modelo saneado)")
        self._update_action_states()

    def _on_chk_outliers_toggled(self, checked: bool):
        """Reage à checkbox e sincroniza com o menu superior."""
        # Sincroniza com a ação do menu (se ela existir) para evitar confusão
        if hasattr(self, 'act_use_clean'):
            if self.act_use_clean.isChecked() != checked:
                self.act_use_clean.blockSignals(True)
                self.act_use_clean.setChecked(checked)
                self.act_use_clean.blockSignals(False)
        
        # Atualiza a flag de estado
        self._usar_limpo_flag = checked
        
        # Log da mudança
        status = "Modelo SANEADO (Sem Outliers)" if checked else "Modelo BRUTO"
        self.log(f"Visualização alterada para: {status}")

        # Atualiza Dashboard e Gráficos instantaneamente
        self._update_dashboard()
        self._refresh_plot_panels()

    def _get_equacoes_texto(self, idx):
        """Retorna uma tupla (equacao_regressao, equacao_estimativa)."""
        if not self.model: return "", ""
        
        res = self.model.modelos[idx]
        params = res.params
        comb = self.model.combinacoes[idx]
        colunas = list(self.model.colunas)
        y_name = self.model.preco
        y_idx = colunas.index(y_name)
        t_y = int(comb[y_idx])

        # 1. Monta o Lado Direito (RHS) comum a ambas
        b0 = params.get('const', 0)
        rhs = f"{self.fmt_num(b0, 6)}"
        for i, col in enumerate(colunas):
            if col == y_name: continue
            if col in params:
                val = params[col]
                
                # ADICIONE "usar_nome=True" na linha abaixo:
                x_form = self.model.transformada_print(int(comb[i]), col, usar_nome=True)
                
                sinal = " + " if val >= 0 else " - "
                rhs += f"{sinal}{self.fmt_num(abs(val), 6)} * {x_form}"

        # 2. Equação de Regressão (Escala Transformada)
        
        # ADICIONE "usar_nome=True" na linha abaixo:
        y_transf = self.model.transformada_print(t_y, y_name, usar_nome=True)
        
        eq_regressao = f"{y_transf} = {rhs}"

        # 3. Equação Estimativa (Isolando a variável dependente Y)
        inversas = {
            0: f"{y_name} = {rhs}",
            1: f"{y_name} = 1 / [{rhs}]",
            2: f"{y_name} = e^[{rhs}]",
            3: f"{y_name} = sqrt[{rhs}]",
            4: f"{y_name} = [{rhs}]^2",
            5: f"{y_name} = 1 / sqrt[{rhs}]",
            6: f"{y_name} = 1 / [{rhs}]^2"
        }
        eq_estimativa = inversas.get(t_y, f"{y_name} = f^-1[{rhs}]")

        return eq_regressao, eq_estimativa

    def load_avaliandos_csv(self):
        """Abre o diálogo manual para carregar a planilha de avaliandos."""
        filtros = "Dados (*.csv *.txt *.tsv *.xls *.xlsx *.ods);;Todos (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "Carregar Avaliandos", "", filtros)
        
        if path:
            self.log_action("Carregar Avaliandos")
            self._process_avaliandos_logic(path)

    def _process_avaliandos_logic(self, path: str):
        """Lógica centralizada que lê o arquivo e valida se as colunas batem com o modelo."""
        try:
            df_av, info = self._read_table_file(path)
            
            # --- VALIDAÇÃO DE SEGURANÇA ---
            if self.df is not None:
                # Pegamos as colunas que o modelo realmente usa (X)
                colunas_necessarias = [c for c in self.df.columns if c != self.preco]
                colunas_planilha = list(df_av.columns)
                
                faltando = [c for c in colunas_necessarias if c not in colunas_planilha]
                
                if faltando:
                    msg = (f"A planilha de avaliandos está incompleta!\n\n"
                           f"Faltam as seguintes colunas: {', '.join(faltando)}\n\n"
                           "Dica: Os nomes devem ser idênticos aos da base de dados (case-sensitive).")
                    QMessageBox.warning(self, "Erro de Compatibilidade", msg)
                    return # Interrompe o carregamento se não for compatível
            # ------------------------------
            
            if self.preco and self.preco in df_av.columns:
                df_av = df_av.drop(columns=[self.preco])
            
            self.df_avaliandos = df_av
            cols_originais = list(df_av.columns)
            novas_cols = ["Mínimo", "Médio", "Máximo", "Amplitude", "Precisão"]
            
            self.table_avaliandos.setColumnCount(len(cols_originais) + len(novas_cols))
            self.table_avaliandos.setHorizontalHeaderLabels(cols_originais + novas_cols)
            
            self.table_avaliandos.setRowCount(len(df_av))
            for r in range(len(df_av)):
                for c in range(len(cols_originais)):
                    val = df_av.iat[r, c]
                    if isinstance(val, (float, int)):
                        item = NumericItem(val, self.fmt_num(val, 4))
                    else:
                        item = QTableWidgetItem(str(val))
                    self.table_avaliandos.setItem(r, c, item)
            
            self.log(f"Avaliandos carregados: {len(df_av)} imóveis (Colunas validadas).")
            self.tabs.setCurrentWidget(self.table_avaliandos)
            
            if self.model:
                self._update_avaliandos_predictions()
                
        except Exception as e:
            self.log(f"Erro ao carregar avaliandos: {e}")
            QMessageBox.critical(self, "Erro", f"Falha ao carregar avaliandos:\n{e}")

    def _update_avaliandos_predictions(self):
        """Calcula as predições e o Grau de Precisão para todos os imóveis avaliandos."""
        if not self.model or not hasattr(self, 'df_avaliandos'):
            return

        usar_limpo = self.usar_limpo()
        
        # Define o multiplicador (Tenta 'Área', senão usa a primeira coluna da planilha)
        col_multi = "Área" if "Área" in self.df_avaliandos.columns else self.df_avaliandos.columns[0]

        self.table_avaliandos.blockSignals(True) 

        for r in range(self.table_avaliandos.rowCount()):
            try:
                # --- IGUAL À FUNÇÃO PREDIZER VALOR DO MENU ---
                # 1. Coleta TODOS os valores da linha de forma genérica
                valores_dict = {}
                for c in range(len(self.df_avaliandos.columns)):
                    col_name = self.df_avaliandos.columns[c]
                    item = self.table_avaliandos.item(r, c)
                    if item:
                        # Limpeza de strings (Tratamento robusto de pontos e vírgulas)
                        text = item.text().strip().replace('R$', '').replace(' ', '')
                        
                        if ',' in text and '.' in text: # Padrão Brasileiro 1.234,56
                            text = text.replace('.', '').replace(',', '.')
                        elif ',' in text: # Padrão 1234,56
                            text = text.replace(',', '.')
                        
                        try:
                            valores_dict[col_name] = float(text)
                        except ValueError:
                            valores_dict[col_name] = 0.0
                    else:
                        valores_dict[col_name] = 0.0

                # 2. Chama o modelo passando o dicionário inteiro (Sem filtrar colunas)
                # O model.py faz a inversão de ln/log automaticamente
                res_pred = self.model.predicao_completa(valores_dict, usar_limpo=usar_limpo)
                
                v_unitario = res_pred['valor_pontual']
                
                # 3. Cálculo da Amplitude (Exatamente a mesma fórmula do menu superior)
                amplitude = (res_pred['ic_superior'] - res_pred['ic_inferior']) / v_unitario
                
                # 4. Enquadramento de Precisão NBR 14653
                info_p = self.model.enquadramento_nbr(usar_limpo=usar_limpo, amplitude_percentual=amplitude)
                graus = ["Inidôneo", "I", "II", "III"]
                grau_idx = info_p.get('precisao', 0)

                # 5. Preenchimento das colunas de resultado na tabela
                base_c = len(self.df_avaliandos.columns)
                fator = valores_dict.get(col_multi, 1.0)
                
                # Resgata os limites de confiança usando a variável correta: res_pred
                v_medio = res_pred['valor_pontual'] * fator
                v_min = res_pred['ic_inferior'] * fator
                v_max = res_pred['ic_superior'] * fator
                
                # 1. Mínimo
                self.table_avaliandos.setItem(r, base_c, NumericItem(v_min, self.fmt_num(v_min, 2)))
                
                # 2. Médio (antigo "Total")
                self.table_avaliandos.setItem(r, base_c + 1, NumericItem(v_medio, self.fmt_num(v_medio, 2)))
                
                # 3. Máximo
                self.table_avaliandos.setItem(r, base_c + 2, NumericItem(v_max, self.fmt_num(v_max, 2)))
                
                # 4. Amplitude (%)
                self.table_avaliandos.setItem(r, base_c + 3, NumericItem(amplitude * 100, self.fmt_num(amplitude * 100, 2) + "%"))
                
                # 5. Grau de Precisão (Com cor dinâmica)
                item_grau = QTableWidgetItem(graus[grau_idx])
                cor_hex = "#4CAF50" if grau_idx >= 2 else "#FFC107" if grau_idx == 1 else "#F44336"
                item_grau.setForeground(QColor(cor_hex))
                item_grau.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_avaliandos.setItem(r, base_c + 4, item_grau)

            except Exception as e:
                print(f"Erro no cálculo automático da linha {r}: {e}")
                continue

        self.table_avaliandos.blockSignals(False)
        self.table_avaliandos.resizeColumnsToContents()

    def load_avaliandos_from_path(self, path: str):
        """Processa o arquivo de avaliandos vindo do Drag & Drop."""
        if not path:
            return

        self.log_action("Carregar Avaliandos (Drag & Drop)")
        
        # Chama a lógica centralizada de processamento
        self._process_avaliandos_logic(path)

    def focus_results_tab(self):
        """Muda o foco para a aba de Resultados."""
        self.tabs.setCurrentWidget(self.log_box)

    def _get_model_ranking(self) -> list[int]:
        """Retorna a lista de índices (raw_idx) ordenados do maior para o menor R²."""
        if not self.model or getattr(self.model, "r2s", None) is None:
            return []
            
        # O SEGREDO: Olha apenas para a lista de R2!
        # Ignoramos a lista 'modelos' porque agora usamos Lazy Loading (são quase todos None)
        validos = [(i, float(r2)) for i, r2 in enumerate(self.model.r2s) if float(r2) > 0.0]
        validos.sort(key=lambda x: x[1], reverse=True)
        return [i for i, r2 in validos]

    def salvar_projeto(self):
        if not self.model or not getattr(self.model, 'r2s', None):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Aviso", "Não há nenhuma análise processada para salvar.")
            return

        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar Projeto", "", "Projeto Avaliação (*.mqo);;Todos os Arquivos (*)"
        )
        
        if path:
            try:
                if not path.endswith('.mqo'):
                    path += '.mqo'
                
                # Coleta os Avaliandos (se existirem) para salvar junto no arquivo
                extra = {}
                if hasattr(self, 'df_avaliandos') and self.df_avaliandos is not None:
                    extra['avaliandos'] = self.df_avaliandos
                
                # ==========================================================
                # NOVO: Salvar também a Tabela de Resultados já processada!
                if hasattr(self, '_last_df_tabela') and self._last_df_tabela is not None:
                    extra['tabela_resultados'] = self._last_df_tabela
                extra['config'] = self.config # <- SALVA AQUI
                # ==========================================================
                
                self.model.salvar_projeto(path, extra_data=extra)
                self.log(f"💾 Projeto salvo com sucesso em:\n{path}")
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Erro", f"Falha ao salvar o projeto:\n{e}")

    def abrir_projeto(self):
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir Projeto", "", "Projeto Avaliação (*.mqo);;Todos os Arquivos (*)"
        )
        
        if not path:
            return

        self.log_box.clear()
        self.log(f"📂 Iniciando carregamento em segundo plano...\n{path}")
        self.lbl_status.setText("Reconstruindo matrizes matemáticas... (Isso pode levar alguns segundos)")
        self.progress.setValue(0)

        # Guarda o caminho numa variável da classe para o callback poder ler depois
        self._caminho_projeto_carregando = path 

        # ==========================================================
        # Chama o Worker passando um "Bound Method" legítimo
        # ==========================================================
        from gui_worker import start_worker
        th = start_worker(
            self._tarefa_carregar_projeto, 
            self.log, 
            self.progress_slot, 
            callback=self._ao_terminar_carregamento, 
            kwargs={"path": path}, # Passamos o caminho como argumento
            owner=self
        )
        self.threads.append(th)
        self._register_blocking_thread(th) # Trava a tela para segurança

    # ============================================================
    # MÉTODOS AUXILIARES PARA A THREAD (Obrigatórios para o Worker)
    # ============================================================
    def _tarefa_carregar_projeto(self, path):
        """Executa a recriação pesada em background (Thread secundária)."""
        from model import MQO 
        # O SEGREDO CONTRA O SEGFAULT: Não enviamos a GUI para dentro da Thread!
        # Passamos None para que o carregamento aconteça de forma silenciosa e 100% segura.
        return MQO.carregar_projeto(path, gui_log=None, gui_progress=None)

    def _ao_terminar_carregamento(self, model_carregado):
        """Reconstrói a Interface Gráfica assim que a Thread Matemática termina."""
        from PyQt6.QtWidgets import QMessageBox, QTableWidgetItem
        import os

        if model_carregado is None:
            self.lbl_status.setText("Falha ao carregar o projeto.")
            return

        try:
            # 1. Recebe o núcleo matemático reconstruído
            self.model = model_carregado
            
            # 2. RESTAURA OS CALLBACKS DE GUI!
            # Agora que estamos de volta à Thread Principal (segura), reconectamos os painéis
            self.model.gui_log = self.log
            self.model.gui_progress = self.progress_slot

            # 3. Restaura o estado da Janela Principal
            self.df = self.model.amostras[0] 
            self.preco = self.model.preco
            
            # Restaura as configurações, se existirem no projeto salvo
            if self.model.extra_data and 'config' in self.model.extra_data:
                self.config.update(self.model.extra_data['config'])
            
            # Puxa a variável guardada no iniciar do processo
            self.csv_path = getattr(self, "_caminho_projeto_carregando", "projeto.mqo")
            self.lbl_arquivo.setText(f"Arquivo: {os.path.basename(self.csv_path)}")
            
            # 4. Preenche visualmente a aba de Dados
            self._fill_table_from_df(self.table_dados, self.df)

            # 5. Restaura os Avaliandos
            if self.model.extra_data and 'avaliandos' in self.model.extra_data:
                self.df_avaliandos = self.model.extra_data['avaliandos']
                
                if self.preco and self.preco in self.df_avaliandos.columns:
                    self.df_avaliandos = self.df_avaliandos.drop(columns=[self.preco])
                
                df_av = self.df_avaliandos
                cols_originais = list(df_av.columns)
                novas_cols = ["Mínimo", "Médio", "Máximo", "Amplitude", "Precisão"]
                
                self.table_avaliandos.setColumnCount(len(cols_originais) + len(novas_cols))
                self.table_avaliandos.setHorizontalHeaderLabels(cols_originais + novas_cols)
                self.table_avaliandos.setRowCount(len(df_av))
                
                for r in range(len(df_av)):
                    for c in range(len(cols_originais)):
                        val = df_av.iat[r, c]
                        text = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
                        self.table_avaliandos.setItem(r, c, QTableWidgetItem(text))

            # 6. Atualiza as travas de menus e Spinbox
            self._update_action_states()

            # 7. Força a Interface a re-escrever tudo
            if self.model._modelo_idx is not None:
                idx_salvo = self.model._modelo_idx
                self.model._modelo_idx = None # Truque para engatilhar redesenho completo
                self._apply_model_idx(idx_salvo, log_source="(Projeto Carregado)")

            # 8. Restaura a Tabela Grande sem recalcular!
            if self.model.extra_data and 'tabela_resultados' in self.model.extra_data:
                self.log("A restaurar a Tabela (Resultados) guardada...\n")
                self._show_table_df(self.model.extra_data['tabela_resultados'])
            else:
                self.resultados() # Só calcula se for um ficheiro mqo antigo
            
            self.lbl_status.setText("Projeto carregado com sucesso.")
            self.progress.setValue(100)
            self.tabs.setCurrentWidget(self.log_box) 
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao desenhar a interface do projeto:\n{e}")
            self.lbl_status.setText("Erro ao carregar.")

    # ============================================================
    # NOVO PROJETO (RESET TOTAL)
    # ============================================================
    def novo_projeto(self):
        """Limpa toda a memória, tabelas e gráficos, voltando ao estado inicial."""
        
        # 1. Pede confirmação se já houver algum dado carregado
        if self.df is not None:
            from PyQt6.QtWidgets import QMessageBox
            resposta = QMessageBox.question(
                self, "Novo Projeto",
                "Tem certeza que deseja iniciar um novo projeto?\nTodos os dados não salvos serão perdidos.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if resposta == QMessageBox.StandardButton.No:
                return
                
        self.log_action("Novo Projeto")

        # 2. Resetar variáveis de estado na memória
        self.df = None
        self.model = None
        self.preco = None
        self.csv_path = None
        
        if hasattr(self, 'df_avaliandos'):
            self.df_avaliandos = None

        self._usar_limpo_flag = False
        self._limpo_ready = False
        self._limpo_ready_idx = None
        self._ultima_amplitude = None
        
        # 3. Limpar a Interface Gráfica (Textos e Labels)
        self.lbl_arquivo.setText("Arquivo: (nenhum)")
        self.lbl_arquivo.setToolTip("Nenhum arquivo carregado.")
        self.lbl_status.setText("Pronto.")
        self.progress.setValue(0)

        # Reseta o Checkbox de Outliers
        self.chk_sem_outliers.blockSignals(True)
        self.chk_sem_outliers.setChecked(False)
        self.chk_sem_outliers.blockSignals(False)

        # Reseta o SpinBox de Modelos
        self._updating_model_spin = True
        self.spin_modelo.setEnabled(False)
        if hasattr(self.spin_modelo, 'set_ranking'):
            self.spin_modelo.set_ranking([])
        self.spin_modelo.setMinimum(0)
        self.spin_modelo.setMaximum(0)
        self.spin_modelo.setValue(0)
        self._updating_model_spin = False

        # 4. Limpar todas as Abas (Tabelas e Logs)
        self.log_box.clear()
        
        self.table_dados.setRowCount(0)
        self.table_dados.setColumnCount(0)
        
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        
        self.table_avaliandos.setRowCount(0)
        self.table_avaliandos.setColumnCount(0)

        # 5. Limpar todos os Gráficos
        paineis = [
            self.panel_box, self.panel_graficos, self.panel_residuos, 
            self.panel_cooks, self.panel_corr, self.panel_aderencia, self.panel_hist
        ]
        for panel in paineis:
            panel.set_figure(None)

        # 6. Limpar o Dashboard (Zerar os Cards)
        cards = [
            self.card_r2, self.card_r2_adj, self.card_fund, self.card_outliers, 
            self.card_resid, self.card_norm, self.card_norm_ks, self.card_homoc, 
            self.card_auto, self.card_vif
        ]
        for card in cards:
            card.set_value("-", color="#000000")

        # 7. Volta o foco para a aba inicial (Dados)
        self.tabs.setCurrentWidget(self.table_dados)

        # 8. Reavalia os botões (Isso vai desativar tudo, já que apagamos os dados)
        self._update_action_states()
        self.log("Sistema pronto para um novo projeto.")