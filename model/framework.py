import itertools
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import threading
from matplotlib.figure import Figure

__all__ = ["MQO"]


# ================================================================
# CORES ANSI
# ================================================================
class Cor:
    RESET = ""
    BOLD = ""
    CYAN = ""
    GREEN = ""
    YELLOW = ""
    MAGENTA = ""


def ativar_cores():
    Cor.RESET = "\033[0m"
    Cor.BOLD = "\033[1m"
    Cor.CYAN = "\033[96m"
    Cor.GREEN = "\033[92m"
    Cor.YELLOW = "\033[93m"
    Cor.MAGENTA = "\033[95m"


def desativar_cores():
    Cor.RESET = ""
    Cor.BOLD = ""
    Cor.CYAN = ""
    Cor.GREEN = ""
    Cor.YELLOW = ""
    Cor.MAGENTA = ""


# ================================================================
# BARRA DE PROGRESSO (terminal)
# ================================================================
def barra_progresso(atual, total, prefixo="", tamanho=30, verbose=True):
    if not verbose:
        return
    if total <= 0:
        total = 1
    frac = atual / total
    cheio = int(tamanho * frac)
    barra = "█" * cheio + "-" * (tamanho - cheio)
    msg = f"\r{prefixo} |{barra}| {int(frac*100):3d}% ({atual}/{total})"
    sys.stdout.write(msg)
    sys.stdout.flush()
    if atual == total:
        sys.stdout.write("\n")

def _maybe_show_fig(fig, show: bool):
    # Nunca tente abrir GUI do Matplotlib fora da thread principal
    if not show:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        fig.show()
    except Exception:
        pass

# ================================================================
# CLASSE — MQO com suporte GUI
# ================================================================
class MQO:
    """Classe para MQO (OLS) com suporte GUI opcional."""

    def __init__(
        self,
        amostra_ini: pd.DataFrame,
        preco="Unitário",
        qtd_transf=7,
        verbose=True,
        outliers_lim=2.0,
        gui_log=None,
        gui_progress=None
    ):
        # ============================
        # MODO GUI
        # ============================
        self.gui_log = gui_log
        self.gui_progress = gui_progress
        self.gui_mode = (gui_log is not None) or (gui_progress is not None)

        # hook de cancelamento (o gui_worker pode injetar um callable aqui)
        self._should_cancel = None

        # Se GUI ativo → remove cores
        if verbose and not self.gui_mode:
            ativar_cores()
        else:
            desativar_cores()

        self.verbose = bool(verbose)

        # ============================
        # Dados da classe
        # ============================
        self.amostras = [amostra_ini.copy()]
        self.preco = preco
        self.qtd_transf = int(qtd_transf)
        self.combinacoes = []
        self.modelos = []
        self.r2s = []
        self.colunas = list(amostra_ini.columns)

        self._modelo_idx = None
        self.amostra = None
        self.outliers_lim = float(outliers_lim)

        # Outliers / versão “limpa”
        self.amostra_limpa = None
        self.amostra_limpa_orig = None
        self.modelo_limpo = None

    # ============================================================
    # HELPERS
    # ============================================================
    def _cancelled(self) -> bool:
        fn = getattr(self, "_should_cancel", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
        return False

    def total_combinacoes(self) -> int:
        if self.combinacoes:
            return int(len(self.combinacoes))
        return int(self.qtd_transf ** len(self.colunas))

    # ============================================================
    # MÉTODOS PARA GUI
    # ============================================================
    def _log(self, msg):
        if self.gui_mode and self.gui_log:
            self.gui_log(msg)
        else:
            print(msg)

    def _progress(self, atual, total, prefixo=""):
        if self.gui_mode and self.gui_progress:
            self.gui_progress(atual, total, prefixo)
        else:
            barra_progresso(atual, total, prefixo, verbose=self.verbose)

    # ============================================================
    # GERAÇÃO DAS COMBINAÇÕES
    # ============================================================
    def combinacoes_calc(self):
        total = self.qtd_transf ** len(self.colunas)
        self.combinacoes = []

        for i, comb in enumerate(
            itertools.product(range(self.qtd_transf), repeat=len(self.colunas)),
            start=1
        ):
            if self._cancelled():
                self._log(f"{Cor.YELLOW}Geração de combinações cancelada. Mantendo resultados parciais.{Cor.RESET}")
                break
            self._progress(i, total, prefixo=f"{Cor.CYAN}Gerando combinações{Cor.RESET}")
            self.combinacoes.append(comb)

    # ============================================================
    # TRANSFORMAÇÃO
    # ============================================================
    def transformar(self, n: int, v: np.ndarray) -> np.ndarray:
        res = np.full_like(v, np.nan, dtype=float)

        if n == 0:
            res = v
        elif n == 1:
            mask = v != 0
            res[mask] = 1 / v[mask]
        elif n == 2:
            mask = v > 0
            res[mask] = np.log(v[mask])
        elif n == 3:
            res = v ** 2
        elif n == 4:
            mask = v >= 0
            res[mask] = np.sqrt(v[mask])
        elif n == 5:
            mask = v != 0
            res[mask] = 1 / (v[mask] ** 2)
        elif n == 6:
            mask = v > 0
            res[mask] = np.sqrt(1 / v[mask])

        return res

    # ============================================================
    # INVERSA DA TRANSFORMAÇÃO
    # ============================================================
    def transformar_inversa(self, n: int, v: np.ndarray) -> np.ndarray:
        res = np.full_like(v, np.nan, dtype=float)

        if n == 0:
            res = v
        elif n == 1:
            res = 1 / v
        elif n == 2:
            res = np.exp(v)
        elif n == 3:
            res = np.sqrt(v)
        elif n == 4:
            res = v ** 2
        elif n == 5:
            res = 1 / np.sqrt(v)
        elif n == 6:
            res = 1 / (v ** 2)

        return res

    # ============================================================
    # GERAR AMOSTRAS TRANSFORMADAS
    # ============================================================
    def amostra_combinar(self):
        if not self.combinacoes:
            self._log(f"{Cor.YELLOW}Nenhuma combinação disponível. Rode combinacoes_calc() antes.{Cor.RESET}")
            return

        base_np = self.amostras[0].to_numpy()
        total = len(self.combinacoes)

        # recomeça da base (evita empilhar duplicado)
        self.amostras = [self.amostras[0].copy()]

        for idx, comb in enumerate(self.combinacoes, start=1):
            if self._cancelled():
                self._log(f"{Cor.YELLOW}Geração de amostras cancelada. Mantendo resultados parciais.{Cor.RESET}")
                break

            self._progress(idx, total, prefixo=f"{Cor.CYAN}Gerando amostras transformadas{Cor.RESET}")

            if idx == 1:
                continue

            arr = np.zeros_like(base_np, dtype=float)
            for i, transf in enumerate(comb):
                arr[:, i] = self.transformar(transf, base_np[:, i])

            arr = np.where(np.isfinite(arr), arr, np.nan)
            df = pd.DataFrame(arr, columns=self.colunas)
            self.amostras.append(df)

    # ============================================================
    # FIT (sequencial)
    # ============================================================
    def fit(self):
        self.combinacoes_calc()
        if not self.combinacoes:
            self._log(f"{Cor.YELLOW}Fit abortado: nenhuma combinação gerada.{Cor.RESET}")
            return

        self.amostra_combinar()
        self.modelos = []
        self.r2s = []

        preco = self.preco
        total = len(self.amostras)

        best_idx = None
        best_r2 = -np.inf

        for i, am in enumerate(self.amostras):
            if self._cancelled():
                self._log(f"{Cor.YELLOW}Fit MQO cancelado. Mantendo resultados parciais.{Cor.RESET}")
                break

            self._progress(i + 1, total, prefixo="MQO (seq.)")

            if am.isna().any().any():
                self.modelos.append(None)
                self.r2s.append(0.0)
                continue

            try:
                mdl = sm.OLS(am[preco], sm.add_constant(am.drop(columns=[preco]))).fit()
                r2 = float(mdl.rsquared)
                self.modelos.append(mdl)
                self.r2s.append(r2)

                if r2 > best_r2:
                    best_r2 = r2
                    best_idx = int(i)

            except Exception:
                self.modelos.append(None)
                self.r2s.append(0.0)

        if best_idx is not None:
            self.modelo = best_idx
            self.modelo_limpo = self.modelos[best_idx]
            self.amostra_limpa = self.amostras[best_idx].copy()
            self.amostra_limpa_orig = self.amostras[0].copy()
        else:
            self._log(f"{Cor.YELLOW}Nenhum modelo MQO ajustado (score válido).{Cor.RESET}")

    # ============================================================
    # Imprimir transformação
    # ============================================================
    def transformada_print(self, n: int, varname=None):
        base_symbol = "y" if varname == self.preco else "x"
        ops = {
            0: f"{base_symbol}",
            1: f"1/{base_symbol}",
            2: f"ln({base_symbol})",
            3: f"{base_symbol}²",
            4: f"√{base_symbol}",
            5: f"1/{base_symbol}²",
            6: f"√(1/{base_symbol})"
        }
        return ops.get(n, "?")

    # ============================================================
    # OUTLIERS QTD
    # ============================================================
    def outliers_qtd(self, modelo_n=None, usar_limpo=False):
        if usar_limpo:
            if self.modelo_limpo is None or self.amostra_limpa is None:
                raise ValueError("modelo_limpo/amostra_limpa não definidos.")
            modelo = self.modelo_limpo
        else:
            if modelo_n is None:
                modelo_n = self._modelo_idx
            if modelo_n is None:
                raise ValueError("Nenhum modelo selecionado.")
            modelo = self.modelos[modelo_n]

        if modelo is None:
            return 0

        if not hasattr(modelo, "get_influence"):
            raise TypeError("Modelo não suporta get_influence().")

        res = modelo.get_influence().resid_studentized_internal
        return int(np.sum(np.abs(res) > self.outliers_lim))

    # ============================================================
    # RESULTADOS (print)
    # ============================================================
    def resultados(self, qtd=20):
        if not self.r2s:
            self._log(f"{Cor.YELLOW}Nenhum modelo ajustado.{Cor.RESET}")
            return

        qtd = int(qtd)
        if qtd <= 0:
            self._log(f"{Cor.YELLOW}Nada a listar (qtd={qtd}).{Cor.RESET}")
            return

        valid = [(i, float(r2)) for i, r2 in enumerate(self.r2s) if (i < len(self.modelos) and self.modelos[i] is not None)]
        if not valid:
            self._log(f"{Cor.YELLOW}Nenhum modelo válido para listar.{Cor.RESET}")
            return

        valid.sort(key=lambda x: x[1], reverse=True)
        ordenado = valid[:qtd]

        largura_idx = len(str(max(i for i, _ in valid)))

        descr_top = {}
        for idx, _r2 in ordenado:
            comb = self.combinacoes[idx]
            descr_top[idx] = [self.transformada_print(t, varname) for t, varname in zip(comb, self.colunas)]

        larg_cols = []
        for j, col in enumerate(self.colunas):
            w = len(str(col))
            for idx, _ in ordenado:
                w = max(w, len(descr_top[idx][j]))
            larg_cols.append(w)

        header_res = f"Outliers > {float(self.outliers_lim):g}"
        larg_res = max(
            len(header_res),
            max(len(str(self.outliers_qtd(i))) for i, _ in ordenado)
        )

        header_desc = " | ".join(
            f"{Cor.BOLD}{str(self.colunas[i]).ljust(larg_cols[i])}{Cor.RESET}"
            for i in range(len(self.colunas))
        )

        title = (
            f"{Cor.CYAN}{'idx'.ljust(largura_idx)}{Cor.RESET} | "
            f"{Cor.GREEN}{'R²'.ljust(10)}{Cor.RESET} | "
            f"{Cor.YELLOW}{header_res.rjust(larg_res)}{Cor.RESET} | "
            f"{header_desc}"
        )

        self._log(title)

        for idx, r2 in ordenado:
            descricoes = [descr_top[idx][i].ljust(larg_cols[i]) for i in range(len(self.colunas))]
            linha_desc = " | ".join(descricoes)
            out = self.outliers_qtd(idx)

            linha = (
                f"{Cor.CYAN}{idx:0{largura_idx}d}{Cor.RESET} | "
                f"{Cor.GREEN}{r2:10.6f}{Cor.RESET} | "
                f"{Cor.YELLOW}{str(out).rjust(larg_res)}{Cor.RESET} | "
                f"{linha_desc}"
            )
            self._log(linha)

    # ============================================================
    # PROPRIEDADE MODELO
    # ============================================================
    @property
    def modelo(self):
        return self.modelos[self._modelo_idx] if self._modelo_idx is not None else None

    @modelo.setter
    def modelo(self, idx):
        if not isinstance(idx, int):
            raise ValueError("Índice deve ser inteiro.")
        if idx < 0 or idx >= len(self.modelos) or self.modelos[idx] is None:
            raise ValueError(f"Modelo {idx} inexistente ou inválido.")
        self._modelo_idx = idx
        self.amostra = self.amostras[idx]

        # reseta “limpo” ao trocar modelo
        self.amostra_limpa = None
        self.amostra_limpa_orig = None
        self.modelo_limpo = None

    # ============================================================
    # VALOR PREVISTO
    # ============================================================
    def valor(self, valores_dict, modelo_n=None, usar_limpo=False):
        if usar_limpo:
            if self.modelo_limpo is None:
                raise ValueError("modelo_limpo não definido.")
            modelo = self.modelo_limpo
            comb_idx = self._modelo_idx
        else:
            if modelo_n is None:
                modelo_n = self._modelo_idx
            if modelo_n is None:
                raise ValueError("Nenhum modelo selecionado.")
            modelo = self.modelos[modelo_n]
            comb_idx = modelo_n

        comb = self.combinacoes[comb_idx]
        cols_all = list(self.colunas)
        preco = self.preco
        x_cols = [c for c in cols_all if c != preco]

        X = []
        for col in x_cols:
            v_orig = float(valores_dict[col])
            idx_col = cols_all.index(col)
            t_idx = comb[idx_col]
            X.append(self.transformar(t_idx, np.array([v_orig], dtype=float))[0])

        X = np.array(X, dtype=float)

        if not hasattr(modelo, "model"):
            raise TypeError("Modelo não é OLS (statsmodels).")

        exog = np.concatenate(([1.0], X))[None, :]
        y_t = float(modelo.predict(exog)[0])

        t_y = comb[cols_all.index(preco)]
        y = float(self.transformar_inversa(t_y, np.array([y_t], dtype=float))[0])
        return y

    # ============================================================
    # ELASTICIDADES
    # ============================================================
    def elasticidades(self, modelo_n=None, usar_limpo=False):
        if usar_limpo:
            modelo = self.modelo_limpo
            amostra_ref = self.amostra_limpa
        else:
            if modelo_n is None:
                modelo_n = self._modelo_idx
            if modelo_n is None:
                raise ValueError("Nenhum modelo selecionado.")
            modelo = self.modelos[modelo_n]
            amostra_ref = self.amostras[modelo_n]

        cols_all = list(self.colunas)
        prec = self.preco
        elastic = {}

        for i, col in enumerate(cols_all):
            if col == prec:
                continue
            # iloc evita warning de Series.__getitem__
            b = modelo.params.iloc[i + 1]
            x_mean = amostra_ref[col].mean()
            y_mean = amostra_ref[prec].mean()
            elastic[col] = float(b * x_mean / y_mean)

        return elastic

    # ============================================================
    # GRÁFICOS (adaptado p/ GUI)
    # ============================================================
    def graficos(self, usar_limpo=False, show=True):
        if self._modelo_idx is None:
            raise ValueError("Nenhum modelo selecionado.")

        amostra_plot = self.amostra_limpa if usar_limpo else self.amostra
        if amostra_plot is None:
            raise ValueError("Amostra não definida.")

        n_axes = max(1, len(self.colunas) - 1)
        fig = Figure(figsize=(4 * n_axes, 4))
        axs = [fig.add_subplot(1, n_axes, i + 1) for i in range(n_axes)]

        comb = self.combinacoes[self._modelo_idx]
        preco = self.preco
        cols_all = list(self.colunas)
        x_cols = [c for c in self.colunas if c != preco]

        for ax_idx, col in enumerate(x_cols):
            ax = axs[ax_idx]

            X_transf = amostra_plot[col].to_numpy()
            t_idx = comb[cols_all.index(col)]
            X_plot = self.transformar_inversa(t_idx, X_transf)

            y_vals = []
            for xi in X_plot:
                val_dict = {}
                for c in self.colunas:
                    if c == preco:
                        continue
                    t_c = comb[cols_all.index(c)]
                    val_dict[c] = float(self.transformar_inversa(t_c, np.array([amostra_plot[c].mean()]))[0])
                val_dict[col] = float(xi)
                y_vals.append(self.valor(val_dict, usar_limpo=usar_limpo))

            ax.scatter(X_plot, y_vals)
            ax.set_xlabel(col)
            ax.set_ylabel(preco)

        fig.tight_layout()
        _maybe_show_fig(fig, show)
        return None if show else fig

    # ============================================================
    # EXCLUSÃO ITERATIVA DE OUTLIERS
    # ============================================================
    def outliers_exc(self, R2_alvo=0.75, out_lim=0.5, conv_lim=0.5):
        if self.amostra is None or self.modelo is None:
            raise ValueError("Selecione um modelo antes.")

        self.amostra_limpa = self.amostra.copy().reset_index(drop=True)
        self.amostra_limpa_orig = self.amostras[0].copy().reset_index(drop=True)
        self.modelo_limpo = self.modelo

        ycolN = self.preco

        out_n = int(float(out_lim) * len(self.amostra_limpa.index))
        conv_n = int(float(conv_lim) * max(out_n, 1))
        conv_i = 0

        modelo = sm.OLS(
            self.amostra_limpa[ycolN],
            sm.add_constant(self.amostra_limpa.drop([ycolN], axis=1))
        ).fit()

        R2_velho = float(modelo.rsquared)
        removidos = 0

        for i in range(max(out_n, 0)):
            if self._cancelled():
                self._log(f"{Cor.YELLOW}Outliers_exc cancelado pelo usuário.{Cor.RESET}")
                break

            influence = modelo.get_influence()
            resid_pad = influence.resid_studentized_internal
            resid_max = float(np.max(np.abs(resid_pad)))

            if resid_max < float(self.outliers_lim):
                self._log("Resíduo máximo atingido!")
                break

            pos = int(np.argmax(np.abs(resid_pad)))

            self.amostra_limpa = self.amostra_limpa.drop(self.amostra_limpa.index[pos]).reset_index(drop=True)
            self.amostra_limpa_orig = self.amostra_limpa_orig.drop(self.amostra_limpa_orig.index[pos]).reset_index(drop=True)

            modelo = sm.OLS(
                self.amostra_limpa[ycolN],
                sm.add_constant(self.amostra_limpa.drop([ycolN], axis=1))
            ).fit()

            R2_novo = float(modelo.rsquared)
            removidos = i + 1

            if R2_novo < R2_velho:
                if conv_i >= conv_n:
                    self._log("Exclusão de outliers não convergiu.")
                    self._log(f"Outliers removidos: {removidos}")
                    break
                conv_i += 1
            else:
                R2_velho = R2_novo

            if R2_novo >= float(R2_alvo):
                self._log(f"Outliers removidos: {removidos}")
                break

        self.modelo_limpo = modelo
        return self.amostra_limpa, self.modelo_limpo, self.amostra_limpa_orig, removidos

    # ============================================================
    # MATRIZ DE CORRELAÇÃO (adaptado p/ GUI)
    # ============================================================
    def matrix_corr(self, usar_limpo=False, show=True):
        if usar_limpo:
            if self.amostra_limpa is None:
                raise ValueError("amostra_limpa não definida.")
            amostra_ref = self.amostra_limpa
        else:
            if self.amostra is None:
                raise ValueError("Nenhum modelo selecionado.")
            amostra_ref = self.amostra

        matrix_corr = amostra_ref.corr()

        import seaborn as sns
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        sns.heatmap(matrix_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title("Matriz de Correlação" + (" (Limpa)" if usar_limpo else ""))

        fig.tight_layout()
        _maybe_show_fig(fig, show)
        return None if show else fig

    # ============================================================
    # BOXPLOT (com transformações inversas)
    # ============================================================
    def boxplot(self, usar_limpo=False, show=True):
        if self._modelo_idx is None:
            raise ValueError("Nenhum modelo selecionado.")

        amostra_ref = self.amostra_limpa if usar_limpo else self.amostra
        if amostra_ref is None:
            raise ValueError("Amostra não definida.")

        df = amostra_ref.copy()
        cols = list(df.columns)

        comb = self.combinacoes[self._modelo_idx]
        cols_all = list(self.colunas)

        for col in cols:
            t = comb[cols_all.index(col)]
            df[col] = self.transformar_inversa(t, df[col].values)

        n = len(cols)
        ncols = 3
        nrows = int(np.ceil(n / ncols))

        fig = Figure(figsize=(5 * ncols, 4 * nrows))
        axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)]

        for i, col in enumerate(cols):
            axes[i].boxplot(df[col].dropna(), vert=True)
            axes[i].set_title(col)
            axes[i].set_ylabel("Valor (escala original)")

        for j in range(len(cols), len(axes)):
            fig.delaxes(axes[j])

        sufixo = " (Limpa)" if usar_limpo else ""
        fig.suptitle(f"Boxplots das Variáveis{sufixo}", fontsize=14)

        fig.tight_layout()
        _maybe_show_fig(fig, show)
        return None if show else fig

    # ============================================================
    # TESTES / DIAGNÓSTICOS
    # ============================================================
    def teste_shapiro(self, usar_limpo=False, significancia=0.05):
        from scipy.stats import shapiro

        modelo = self.modelo_limpo if usar_limpo else self.modelo
        stat, pvalue = shapiro(modelo.resid)

        print("\nTeste de Normalidade - Shapiro-Wilk")
        print("-------------------------------------------")
        print(f"Statistic : {stat:.6f}")
        print(f"P-value   : {pvalue:.6f}")
        print(f"Significância adotada: {significancia:.3f}")
        print("-------------------------------------------")
        print("Interpretação estatística:")
        print(" - Hipótese nula: resíduos seguem distribuição normal")
        print(" - p-value > significância → NÃO rejeita H0 → Normalidade OK")
        print(" - p-value ≤ significância → Rejeita H0 → Resíduos NÃO normais")
        print("-------------------------------------------")

        if pvalue > significancia:
            conclusao = (
                "Conclusão: NÃO há evidências contra a normalidade dos resíduos.\n"
                "O pressuposto de normalidade do modelo OLS está ATENDIDO."
            )
        else:
            conclusao = (
                "Conclusão: Há evidências de NÃO normalidade dos resíduos.\n"
                "⚠️ O modelo OLS pode violar o pressuposto de normalidade."
            )

        print(conclusao + "\n")
        # return {"statistic": stat, "pvalue": pvalue}
        return pvalue > significancia

    def teste_kstest(self, usar_limpo=False, significancia=0.05):
        from scipy.stats import kstest, zscore

        modelo = self.modelo_limpo if usar_limpo else self.modelo
        residuos_pad = zscore(modelo.resid)

        stat, pvalue = kstest(residuos_pad, 'norm')

        print("\nTeste de Normalidade - Kolmogorov-Smirnov")
        print("-------------------------------------------")
        print(f"Statistic : {stat:.6f}")
        print(f"P-value   : {pvalue:.6f}")
        print(f"Significância adotada: {significancia:.3f}")
        print("-------------------------------------------")

        if pvalue > significancia:
            conclusao = (
                "Conclusão: NÃO há evidências contra a normalidade dos resíduos.\n"
                "O pressuposto de normalidade do modelo OLS está ATENDIDO."
            )
        else:
            conclusao = (
                "Conclusão: Há evidências de NÃO-normalidade dos resíduos.\n"
                "⚠️ O modelo OLS pode violar o pressuposto de normalidade."
            )

        print(conclusao + "\n")
        return {"statistic": stat, "pvalue": pvalue}

    def histograma(self, usar_limpo=False, show=True):
        import seaborn as sns

        modelo = self.modelo_limpo if usar_limpo else self.modelo
        resid = np.array(modelo.resid)

        fig = Figure(figsize=(7, 5))
        ax = fig.add_subplot(1, 1, 1)
        sns.histplot(resid, kde=True, ax=ax)
        ax.set_xlabel("Resíduos")
        ax.set_ylabel("Frequência")
        ax.set_title("Histograma dos Resíduos" + (" (Limpo)" if usar_limpo else ""))
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        _maybe_show_fig(fig, show)
        return None if show else fig


    def distribuicao_residuos(self, usar_limpo=False):
        import scipy.stats as st

        modelo = self.modelo_limpo if usar_limpo else self.modelo
        resid = np.array(modelo.resid)

        z_scores = st.zscore(resid)
        z_tot = len(z_scores)

        i1 = np.sum((z_scores >= -1) & (z_scores <= 1))
        i164 = np.sum((z_scores >= -1.64) & (z_scores <= 1.64))
        i196 = np.sum((z_scores >= -1.96) & (z_scores <= 1.96))

        p1 = i1 / z_tot
        p164 = i164 / z_tot
        p196 = i196 / z_tot

        print("\nDistribuição dos Resíduos (comparação com a Normal):")
        print("------------------------------------------------------")
        print(f"Faixa ±1.00σ  → Observado: {p1:6.3%} | Teórico ≈ 68%")
        print(f"Faixa ±1.64σ  → Observado: {p164:6.3%} | Teórico ≈ 90%")
        print(f"Faixa ±1.96σ  → Observado: {p196:6.3%} | Teórico ≈ 95%")
        print("------------------------------------------------------")

    def heterocedasticidade(self, usar_limpo=False):
        import statsmodels.stats.api as sms

        modelo = self.modelo_limpo if usar_limpo else self.modelo

        names = ['LM Statistic', 'LM p-value', 'F Statistic', 'F p-value']
        test_result = sms.het_breuschpagan(modelo.resid, modelo.model.exog)
        resultado = dict(zip(names, test_result))

        print("\nTeste de Heterocedasticidade - Breusch-Pagan")
        print("------------------------------------------------")
        for n, v in resultado.items():
            print(f"{n:20s}: {v:.6f}")
        print("------------------------------------------------")

        lm_pvalue = resultado['LM p-value']
        f_pvalue = resultado['F p-value']

        if (lm_pvalue > 0.05) and (f_pvalue > 0.05):
            conclusao = (
                "Conclusão: NÃO há evidências de heterocedasticidade.\n"
                "O modelo OLS atende ao pressuposto de homocedasticidade."
            )
        else:
            conclusao = (
                "Conclusão: Há evidências de heterocedasticidade nos resíduos.\n"
                "⚠️ O modelo OLS pode violar o pressuposto de homocedasticidade."
            )

        print(conclusao + "\n")
        # return resultado
        return (lm_pvalue > 0.05) and (f_pvalue > 0.05)

    def autocorrelacao(self, usar_limpo=False):
        from statsmodels.stats.stattools import durbin_watson

        modelo = self.modelo_limpo if usar_limpo else self.modelo
        dw = durbin_watson(modelo.resid)

        print("\nTeste de Autocorrelação - Durbin-Watson")
        print("-------------------------------------------")
        print(f"Durbin-Watson: {dw:.4f}")
        print("-------------------------------------------")

        if 1.5 <= dw <= 2.5:
            conclusao = (
                "Conclusão: O valor está na faixa aceitável.\n"
                "Não há evidências significativas de autocorrelação."
            )
        else:
            conclusao = (
                "Conclusão: Há indícios de autocorrelação.\n"
                "O modelo OLS pode violar o pressuposto de independência."
            )

        print(conclusao + "\n")
        # return {"durbin_watson": dw}
        return 1.5 <= dw <= 2.5

    def multicolinearidade(self, usar_limpo=False):
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant

        amostra = self.amostra_limpa if usar_limpo else self.amostra
        if amostra is None:
            raise ValueError("Amostra não encontrada.")

        y_col = self.preco
        X = amostra.drop(columns=[y_col])
        X_const = add_constant(X, has_constant="add")

        vif_df = pd.DataFrame({
            "Variável": X_const.columns,
            "VIF": [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
        })

        vif_sem_const = vif_df[vif_df["Variável"] != "const"]

        print("\nTeste de Multicolinearidade - VIF")
        print("-------------------------------------------")
        print(vif_df)
        print("-------------------------------------------")

        max_vif = vif_sem_const["VIF"].max()

        if max_vif < 5:
            conclusao = "Conclusão: VIFs abaixo de 5 → sem multicolinearidade relevante."
        elif max_vif < 10:
            conclusao = "Conclusão: VIF entre 5 e 10 → atenção, multicolinearidade moderada."
        else:
            conclusao = "Conclusão: VIF ≥ 10 → multicolinearidade séria."

        print(conclusao + "\n")
        return vif_df

    def residuos_grafico(self, usar_limpo=False, show=True):
        modelo = self.modelo_limpo if usar_limpo else self.modelo
        amostra = self.amostra_limpa if usar_limpo else self.amostra

        influence = modelo.get_influence()
        standardized_residuals = influence.resid_studentized_internal

        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        
        # AJUSTE AQUI: Somamos 1 ao index para que a contagem visual comece em 1
        ax.scatter(amostra.index + 1, standardized_residuals)
        
        ax.set_xlabel("Número do Dado")
        ax.set_ylabel("Resíduo Padronizado")
        ax.axhline(y=float(self.outliers_lim), color='red', linestyle='--', linewidth=1)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axhline(y=-float(self.outliers_lim), color='red', linestyle='--', linewidth=1)
        ax.set_title("Resíduos Padronizados" + (" (Limpo)" if usar_limpo else ""))

        fig.tight_layout()
        _maybe_show_fig(fig, show)
        return None if show else fig

    def aderencia(self, usar_limpo=False, show=True):
        if usar_limpo:
            amostra = self.amostra_limpa
            modelo = self.modelo_limpo
        else:
            amostra = self.amostra
            modelo = self.modelo

        if amostra is None or modelo is None:
            raise ValueError("Amostra ou modelo não definidos.")

        modelo_n = self._modelo_idx
        comb = self.combinacoes[modelo_n]

        cols = list(self.colunas)
        y_col = self.preco
        idx_y = cols.index(y_col)

        y_t_test = amostra[y_col].to_numpy()
        t_y = comb[idx_y]
        y_test = self.transformar_inversa(t_y, y_t_test)

        X_exog = modelo.model.exog
        y_t_pred = modelo.predict(X_exog)
        y_pred = self.transformar_inversa(t_y, np.array(y_t_pred))

        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)

        fig = Figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
        ax.plot([y_test.min(), y_test.max()], p([y_test.min(), y_test.max()]), "r-", label="Trendline")
        ax.set_xlabel("Observado")
        ax.set_ylabel("Estimado")
        ax.set_title("Aderência" + (" (Limpa)" if usar_limpo else ""))
        #ax.legend()

        fig.tight_layout()
        _maybe_show_fig(fig, show)
        return None if show else fig


    # ============================================================
    # SUMMARY DO MODELO (texto)
    # ============================================================
    def resumo(self, usar_limpo=False):
        modelo = self.modelo_limpo if usar_limpo else self.modelo
        if modelo is None:
            raise ValueError("Modelo não definido (rode fit e selecione um modelo).")
        return modelo.summary().as_text()

    # ============================================================
    # RESULTADOS EM TABELA (para GUI)
    # ============================================================
    def resultados_tabela(self, qtd=20) -> pd.DataFrame:
        if not self.r2s:
            return pd.DataFrame()

        qtd = int(qtd)
        if qtd <= 0:
            return pd.DataFrame()

        valid = [(i, float(r2)) for i, r2 in enumerate(self.r2s) if (i < len(self.modelos) and self.modelos[i] is not None)]
        if not valid:
            return pd.DataFrame()

        valid.sort(key=lambda x: x[1], reverse=True)
        ordenado = valid[:qtd]

        rows = []
        for idx, r2 in ordenado:
            out = self.outliers_qtd(idx)
            comb = self.combinacoes[idx]
            descr = [self.transformada_print(t, varname) for t, varname in zip(comb, self.colunas)]

            row = {
                "idx": int(idx),
                "R²": float(r2),
                f"Outliers > {float(self.outliers_lim):g}": int(out),
            }
            for j, col in enumerate(self.colunas):
                row[str(col)] = descr[j]
            rows.append(row)

        return pd.DataFrame(rows)

    def predicao_completa(self, valores_dict, usar_limpo=False, alpha=0.20):
        """
        Calcula valor predito, Intervalo de Confiança (80% por padrão NBR) 
        e Campo de Arbítrio (15%).
        """
        if usar_limpo:
            modelo = self.modelo_limpo
            comb_idx = self._modelo_idx
        else:
            modelo = self.modelo
            comb_idx = self._modelo_idx

        if modelo is None:
            raise ValueError("Modelo não ajustado ou não selecionado.")

        comb = self.combinacoes[comb_idx]
        cols_all = list(self.colunas)
        prec_col = self.preco
        x_cols = [c for c in cols_all if c != prec_col]

        # 1. Transformar as entradas do usuário conforme o modelo selecionado
        X_input = []
        for col in x_cols:
            v_orig = float(valores_dict[col])
            idx_col = cols_all.index(col)
            t_idx = comb[idx_col]
            X_input.append(self.transformar(t_idx, np.array([v_orig], dtype=float))[0])

        # 2. Preparar exógena para o statsmodels (adicionando a constante)
        exog = np.concatenate(([1.0], X_input))[None, :]
        
        # 3. Obter predição e intervalo de confiança (Mean CI)
        prediction_obj = modelo.get_prediction(exog)
        pred_frame = prediction_obj.summary_frame(alpha=alpha) # alpha 0.20 = 80% confiança

        y_t_pred = float(pred_frame['mean'].iloc[0])
        y_t_lower = float(pred_frame['mean_ci_lower'].iloc[0])
        y_t_upper = float(pred_frame['mean_ci_upper'].iloc[0])

        # 4. Transformação Inversa para voltar à escala de moeda (R$)
        t_y = comb[cols_all.index(prec_col)]
        y_final = float(self.transformar_inversa(t_y, np.array([y_t_pred]))[0])
        li_final = float(self.transformar_inversa(t_y, np.array([y_t_lower]))[0])
        ls_final = float(self.transformar_inversa(t_y, np.array([y_t_upper]))[0])

        # Garantir ordem correta (algumas transformações como 1/x invertem os limites)
        limites = sorted([li_final, ls_final])

        return {
            "valor_pontual": y_final,
            "ic_inferior": limites[0],
            "ic_superior": limites[1],
            "arbitrio_inferior": y_final * 0.85,
            "arbitrio_superior": y_final * 1.15,
            "alpha": alpha
        }

    def enquadramento_nbr(self, usar_limpo=False, amplitude_percentual=None):
        """
        Analisa o modelo atual e retorna o Grau de Fundamentação 
        e Grau de Precisão (se amplitude_percentual for fornecida).
        """
        modelo = self.modelo_limpo if usar_limpo else self.modelo
        if modelo is None:
            raise ValueError("Modelo não ajustado.")

        # 1. Variáveis e Dados
        n = int(modelo.nobs) # Número de dados
        k = int(len(modelo.params) - 1) # Número de variáveis independentes
        
        # Critério: n >= x(k+1)
        if n >= 6 * (k + 1): grau_n = 3
        elif n >= 4 * (k + 1): grau_n = 2
        elif n >= 3 * (k + 1): grau_n = 1
        else: grau_n = 0 # Inidôneo pela norma para alguns fins

        # 2. Significância Global (Teste F)
        p_f = modelo.f_pvalue
        if p_f <= 0.01: grau_f = 3
        elif p_f <= 0.02: grau_f = 2
        elif p_f <= 0.05: grau_f = 1
        else: grau_f = 0

        # 3. Significância dos Regressores (p-values)
        # Pegamos o pior p-value entre os regressores (excluindo a constante)
        p_max = modelo.pvalues.drop('const').max()
        if p_max <= 0.10: grau_p = 3
        elif p_max <= 0.20: grau_p = 2
        elif p_max <= 0.30: grau_p = 1
        else: grau_p = 0

        # O Grau de Fundamentação é o MENOR dos três
        fundamentacao = min(grau_n, grau_f, grau_p)

        resumo = {
            "n": n, "k": k, "grau_n": grau_n,
            "p_f": p_f, "grau_f": grau_f,
            "p_max": p_max, "grau_p": grau_p,
            "fundamentacao": fundamentacao
        }

        # 4. Grau de Precisão (Depende da amplitude do Intervalo de Confiança)
        if amplitude_percentual is not None:
            amp = amplitude_percentual * 100
            if amp <= 30: grau_prec = 3
            elif amp <= 40: grau_prec = 2
            elif amp <= 50: grau_prec = 1
            else: grau_prec = 0
            resumo["amplitude"] = amp
            resumo["precisao"] = grau_prec

        return resumo
        
    def salvar_todos_graficos(self, diretorio, usar_limpo=False):
        import os
        caminhos = {}
        planos = {
            "boxplot": self.boxplot,
            "graficos": self.graficos,
            "residuos": self.residuos_grafico,
            "cooks": self.cooks_distance_grafico, # <--- ADICIONADO AQUI
            "corr": self.matrix_corr,
            "aderencia": self.aderencia,
            "hist": self.histograma
        }
        
        for nome, func in planos.items():
            fig = func(usar_limpo=usar_limpo, show=False)
            if fig:
                path = os.path.join(diretorio, f"{nome}.png")
                fig.savefig(path, dpi=150)
                caminhos[nome] = path
                plt.close(fig) # Limpa memória
        return caminhos

    def cooks_distance_grafico(self, usar_limpo=False, show=True):
        modelo = self.modelo_limpo if usar_limpo else self.modelo
        influence = modelo.get_influence()
        (c, p) = influence.cooks_distance
        n = len(c)
        
        threshold_nbr = 4 / n
        threshold_classico = 1.0 

        fig = Figure(figsize=(7, 5))
        ax = fig.add_subplot(1, 1, 1)

        # AJUSTE 1: Criamos o intervalo de 1 até n (inclusive)
        indices_ajustados = np.arange(1, n + 1)
        ax.stem(indices_ajustados, c, markerfmt=",", linefmt="C0-", basefmt="k-")
        
        # Linha 1: O limite de 4/n (Atenção)
        ax.axhline(y=threshold_nbr, color='orange', linestyle='--', label=f'Atenção (4/n: {threshold_nbr:.2f})')
        
        # Linha 2: O limite de 1.0 (Crítico)
        ax.axhline(y=threshold_classico, color='red', linestyle='--', linewidth=2, label='Crítico (1.0)')
        
        # AJUSTE 2: Etiquetar pontos influentes somando 1 ao índice
        # Onde c > threshold_nbr, pegamos o índice original (0-based) para acessar o array 'c'
        influential_points = np.where(c > threshold_nbr)[0]
        for i in influential_points:
            # Exibimos i + 1 no texto e posicionamos em i + 1 no eixo X
            ax.annotate(str(i + 1), (i + 1, c[i]), textcoords="offset points", 
                        xytext=(0, 5), ha='center', fontsize=8, color='darkred')

        ax.set_xlabel("Número do Dado") # Adicionado para manter a consistência
        ax.set_ylim(0, max(max(c)*1.1, 1.1)) 
        ax.set_title("Influência - Distância de Cook")
        ax.legend()
        
        fig.tight_layout()
        _maybe_show_fig(fig, show)
        return None if show else fig

    def check_normalidade(self, idx, usar_limpo=False):
        """Verificação silenciosa para o Dashboard."""
        from scipy.stats import shapiro
        mdl = self.modelo_limpo if usar_limpo else self.modelos[idx]
        if mdl is None: return False
        _, pvalue = shapiro(mdl.resid)
        return pvalue > 0.05

    def check_homocedasticidade(self, idx, usar_limpo=False):
        """Verificação silenciosa para o Dashboard."""
        import statsmodels.stats.api as sms
        mdl = self.modelo_limpo if usar_limpo else self.modelos[idx]
        if mdl is None: return False
        _, p_valor, _, _ = sms.het_breuschpagan(mdl.resid, mdl.model.exog)
        return p_valor > 0.05

    def check_autocorrelacao(self, idx, usar_limpo=False):
        """Verificação silenciosa para o Dashboard."""
        from statsmodels.stats.stattools import durbin_watson
        mdl = self.modelo_limpo if usar_limpo else self.modelos[idx]
        if mdl is None: return False
        dw = durbin_watson(mdl.resid)
        return 1.5 <= dw <= 2.5
