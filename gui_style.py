# ===============================================================
# GUI STYLE — Tema Dark Gray para PyQt6
# ===============================================================

def load_light_style():
    return """
        QMainWindow, QDialog {
            background-color: #ffffff;
            color: #000000;
        }
        QTabWidget::pane {
            border: 1px solid #cccccc;
            background: #ffffff;
        }
        QTabBar::tab {
            background: #f0f0f0;
            color: #333333;
            padding: 8px 20px;
            border: 1px solid #cccccc;
            border-bottom: none;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            font-weight: bold;
        }
        QPlainTextEdit, QTableWidget {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #dddddd;
            gridline-color: #eeeeee;
        }
        QHeaderView::section {
            background-color: #f5f5f5;
            color: #333333;
            padding: 4px;
            border: 1px solid #dddddd;
        }
        QLabel {
            color: #333333;
        }
    """

def load_dark_style():
    """
    Retorna o stylesheet completo para tema dark moderno.
    Compatível com todos os widgets usados na aplicação.
    """
    return """
        QWidget {
            background-color: #2b2b2b;
            color: #dddddd;
            font-size: 14px;
            font-family: Arial;
        }

        QMainWindow {
            background-color: #2b2b2b;
        }

        QPushButton {
            background-color: #3c3f41;
            border: 1px solid #555555;
            padding: 6px;
            border-radius: 4px;
            color: #f0f0f0;
        }

        QPushButton:hover {
            background-color: #4b4e50;
        }

        QPushButton:pressed {
            background-color: #282828;
        }

        QLineEdit, QComboBox, QTextEdit {
            background-color: #3c3f41;
            border: 1px solid #555555;
            padding: 4px;
            border-radius: 4px;
            color: #ffffff;
        }

        QProgressBar {
            border: 1px solid #3c3f41;
            border-radius: 3px;
            text-align: center;
            color: #ffffff;
        }

        QProgressBar::chunk {
            background-color: #007acc;
            width: 20px;
        }

        QTableWidget {
            gridline-color: #444444;
            background-color: #3c3f41;
            color: #ffffff;
        }

        QHeaderView::section {
            background-color: #444444;
            padding: 4px;
            border: 1px solid #222222;
        }

        QTextEdit {
            background-color: #1e1e1e;
            border: 1px solid #444444;
        }

    """
