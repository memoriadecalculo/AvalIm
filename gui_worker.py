# ======================================================================
# gui_worker.py — Worker seguro para rodar métodos do framework (MQO) em QThread
# + Cancelamento cooperativo via requestInterruption()
# (RF removido: nada aqui é específico de RF; mantido apenas o necessário p/ MQO)
# ======================================================================

from __future__ import annotations

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
import io
import contextlib


class CancelledError(Exception):
    """Usada para abortar a execução quando o usuário cancela."""
    pass


class Worker(QObject):
    finished = pyqtSignal(object)          # resultado da função
    progress = pyqtSignal(int, int, str)   # atual, total, prefixo
    log = pyqtSignal(str)                  # texto para log

    def __init__(self, func, log_fn=None, progress_fn=None, args=None, kwargs=None):
        super().__init__()
        self.func = func
        self.user_log = log_fn
        self.user_progress = progress_fn
        self.args = args or ()
        self.kwargs = kwargs or {}

    def _safe_emit(self, sig, *args):
        """Emite sinal ignorando condição clássica: Qt object já deletado."""
        try:
            sig.emit(*args)
        except RuntimeError:
            pass

    @pyqtSlot()
    def run(self):
        # callbacks injetados no modelo (framework / MQO)

        def gui_log(msg):
            self._safe_emit(self.log, str(msg))

        def gui_progress(atual=None, total=None, prefixo=""):
            try:
                total_i = int(total) if total is not None else 1
                if total_i <= 0:
                    total_i = 1
                atual_i = int(atual) if atual is not None else 0
                self._safe_emit(self.progress, atual_i, total_i, str(prefixo))
            except Exception as e:
                self._safe_emit(self.log, f"Erro no gui_progress: {e}")

        def gui_cancel_requested():
            th = QThread.currentThread()
            try:
                return bool(th.isInterruptionRequested())
            except Exception:
                return False

        def gui_check_cancel(where=""):
            if gui_cancel_requested():
                raise CancelledError("Cancelado pelo usuário." + (f" ({where})" if where else ""))

        # exige bound method (método de instância)
        model = getattr(self.func, "__self__", None)
        if model is None:
            self._safe_emit(self.log, "Erro no worker: função fornecida não é método de instância (bound method).")
            self._safe_emit(self.finished, None)
            return

        # salva estado anterior do modelo para restaurar no finally
        old = {
            "gui_log": getattr(model, "gui_log", None),
            "gui_progress": getattr(model, "gui_progress", None),
            "_should_cancel": getattr(model, "_should_cancel", None),
            "_log": getattr(model, "_log", None),
            "_progress": getattr(model, "_progress", None),
        }

        # injeta no modelo durante a execução do worker (compatível com o MQO atual)
        try:
            model.gui_log = gui_log
        except Exception:
            pass
        try:
            model.gui_progress = gui_progress
        except Exception:
            pass
        try:
            model._should_cancel = gui_cancel_requested
        except Exception:
            pass

        # se o modelo usar _log/_progress internamente, direciona pro worker também
        try:
            model._log = gui_log
        except Exception:
            pass
        try:
            model._progress = gui_progress
        except Exception:
            pass

        buf = io.StringIO()
        result = None

        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                result = self.func(*self.args, **self.kwargs)

        except CancelledError as e:
            out = buf.getvalue()
            if out and out.strip():
                self._safe_emit(self.log, out.rstrip())
            self._safe_emit(self.log, str(e))
            result = None

        except Exception as e:
            out = buf.getvalue()
            if out and out.strip():
                self._safe_emit(self.log, out.rstrip())
            self._safe_emit(self.log, f"Erro no worker: {e}")
            result = None

        else:
            out = buf.getvalue()
            if out and out.strip():
                self._safe_emit(self.log, out.rstrip())

        finally:
            # restaura o modelo (evita crash por closures do Worker)
            try:
                model.gui_log = old["gui_log"]
            except Exception:
                pass
            try:
                model.gui_progress = old["gui_progress"]
            except Exception:
                pass
            try:
                model._should_cancel = old["_should_cancel"]
            except Exception:
                pass
            try:
                model._log = old["_log"]
            except Exception:
                pass
            try:
                model._progress = old["_progress"]
            except Exception:
                pass

        self._safe_emit(self.finished, result)


def start_worker(func, log_slot, progress_slot, callback=None, args=None, kwargs=None, owner=None):
    """
    Inicia um Worker em QThread.
    - owner: passe sua MainWindow (self) para manter referências vivas e evitar GC.
    Retorna o QThread.
    """
    thread = QThread(owner) if owner is not None else QThread()
    worker = Worker(func, log_slot, progress_slot, args=args, kwargs=kwargs)

    worker.moveToThread(thread)

    if log_slot:
        worker.log.connect(log_slot)
    if progress_slot:
        worker.progress.connect(progress_slot)
    if callback:
        worker.finished.connect(callback)

    thread.started.connect(worker.run)

    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    # mantém refs vivas para evitar GC prematuro
    if owner is not None:
        if not hasattr(owner, "_active_threads"):
            owner._active_threads = set()
        if not hasattr(owner, "_active_workers"):
            owner._active_workers = set()

        owner._active_threads.add(thread)
        owner._active_workers.add(worker)

        def _cleanup_refs():
            try:
                owner._active_threads.discard(thread)
            except Exception:
                pass
            try:
                owner._active_workers.discard(worker)
            except Exception:
                pass

        thread.finished.connect(_cleanup_refs)
    else:
        thread.worker = worker

    thread.start()
    return thread
