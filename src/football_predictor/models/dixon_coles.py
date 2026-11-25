"""Time-decayed Dixon-Coles (1997) bivariate Poisson football model from scratch.

Uses log-space parameterization with exact sum-to-zero attack identifiability
(sum(log(alpha_i)) = 0), vectorized analytical gradients for fast L-BFGS-B
convergence, warm-starting for walk-forward backtests, and derived market
probabilities (1X2, Over/Under 2.5, BTTS, Top-N exact scorelines).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from football_predictor.config import DixonColesConfig
from football_predictor.models.base import (
    BaseMatchModel,
    MarketProbabilities,
    ScorelineProbability,
    normalize_probabilities,
)

logger = logging.getLogger(__name__)


def dixon_coles_tau(
    x: np.ndarray | int,
    y: np.ndarray | int,
    lam: np.ndarray | float,
    mu: np.ndarray | float,
    rho: float,
) -> np.ndarray:
    """Compute the Dixon-Coles low-score dependence multiplier tau(x, y; lam, mu, rho)."""
    x_arr = np.asarray(x, dtype=np.int64)
    y_arr = np.asarray(y, dtype=np.int64)
    lam_arr = np.asarray(lam, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)

    tau = np.ones(np.broadcast_shapes(x_arr.shape, y_arr.shape, lam_arr.shape, mu_arr.shape))
    tau = np.where((x_arr == 0) & (y_arr == 0), 1.0 - lam_arr * mu_arr * rho, tau)
    tau = np.where((x_arr == 1) & (y_arr == 0), 1.0 + mu_arr * rho, tau)
    tau = np.where((x_arr == 0) & (y_arr == 1), 1.0 + lam_arr * rho, tau)
    tau = np.where((x_arr == 1) & (y_arr == 1), 1.0 - rho, tau)
    return tau


class DixonColesModel(BaseMatchModel):
    """Maximum-likelihood Dixon-Coles model with exponential time-decay."""

    name: str = "dixon_coles"

    def __init__(self, config: DixonColesConfig | None = None) -> None:
        self.config = config or DixonColesConfig()
        self.team_to_idx: dict[str, int] = {}
        self.idx_to_team: list[str] = []
        self.team_display_names: dict[str, str] = {}
        self.log_attack_: np.ndarray = np.array([], dtype=np.float64)
        self.log_defense_: np.ndarray = np.array([], dtype=np.float64)
        self.home_adv_: float = 0.0
        self.rho_: float = 0.0
        self.reference_date_: date | None = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.nll_: float = float("nan")
        self._warm_state: dict[str, object] | None = None

    def set_warm_start(self, other: DixonColesModel) -> None:
        """Store parameters from a previously fitted model to accelerate walk-forward refits."""
        if not other.idx_to_team:
            return
        self._warm_state = {
            "log_attack": dict(zip(other.idx_to_team, other.log_attack_, strict=True)),
            "log_defense": dict(zip(other.idx_to_team, other.log_defense_, strict=True)),
            "home_adv": other.home_adv_,
            "rho": other.rho_,
        }

    def _unpack_params(
        self,
        theta: np.ndarray,
        n_teams: int,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Unpack free vector `theta` enforcing exact sum(log_attack) == 0."""
        log_attack = np.empty(n_teams, dtype=np.float64)
        log_attack[:-1] = theta[: n_teams - 1]
        log_attack[-1] = -np.sum(theta[: n_teams - 1])
        log_defense = theta[n_teams - 1 : 2 * n_teams - 1]
        home_adv = float(theta[2 * n_teams - 1])
        rho = float(theta[2 * n_teams])
        return log_attack, log_defense, home_adv, rho

    def _neg_log_likelihood_and_grad(
        self,
        theta: np.ndarray,
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        fthg: np.ndarray,
        ftag: np.ndarray,
        weights: np.ndarray,
        n_teams: int,
    ) -> tuple[float, np.ndarray]:
        """Vectorized weighted negative log-likelihood and exact analytical gradient."""
        log_attack, log_defense, home_adv, rho = self._unpack_params(theta, n_teams)

        eta_lam = np.clip(log_attack[home_idx] + log_defense[away_idx] + home_adv, -6.0, 4.0)
        eta_mu = np.clip(log_attack[away_idx] + log_defense[home_idx], -6.0, 4.0)
        lam = np.exp(eta_lam)
        mu = np.exp(eta_mu)

        m00 = (fthg == 0) & (ftag == 0)
        m10 = (fthg == 1) & (ftag == 0)
        m01 = (fthg == 0) & (ftag == 1)
        m11 = (fthg == 1) & (ftag == 1)

        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m11] = 1.0 - rho

        # Guard against invalid rho combinations during line-search trials
        if np.any(tau <= 1e-8):
            return 1e12, np.zeros_like(theta)

        # Weighted log-likelihood (omitting factorial constants)
        ll_vec = fthg * eta_lam - lam + ftag * eta_mu - mu + np.log(tau)
        nll = -float(np.dot(weights, ll_vec))

        # Regularization on team log-parameters (centered on defense to preserve goal scale)
        reg = float(self.config.l2_reg)
        b_mean = float(np.mean(log_defense))
        b_centered = log_defense - b_mean
        if reg > 0.0:
            penalty = float(np.dot(log_attack, log_attack) + np.dot(b_centered, b_centered))
            nll += 0.5 * reg * penalty

        # Analytical derivatives d(log tau)/d(eta_lam), d(log tau)/d(eta_mu), d(log tau)/d(rho)
        d_logtau_d_etalam = np.zeros_like(lam)
        d_logtau_d_etamu = np.zeros_like(lam)
        d_logtau_d_rho = np.zeros_like(lam)

        if np.any(m00):
            prod00 = lam[m00] * mu[m00]
            t00 = tau[m00]
            d_logtau_d_etalam[m00] = (-prod00 * rho) / t00
            d_logtau_d_etamu[m00] = (-prod00 * rho) / t00
            d_logtau_d_rho[m00] = -prod00 / t00
        if np.any(m10):
            t10 = tau[m10]
            d_logtau_d_etamu[m10] = (mu[m10] * rho) / t10
            d_logtau_d_rho[m10] = mu[m10] / t10
        if np.any(m01):
            t01 = tau[m01]
            d_logtau_d_etalam[m01] = (lam[m01] * rho) / t01
            d_logtau_d_rho[m01] = lam[m01] / t01
        if np.any(m11):
            d_logtau_d_rho[m11] = -1.0 / tau[m11]

        # Negative log-likelihood gradients w.r.t. linear predictors
        g_lam = -weights * ((fthg - lam) + d_logtau_d_etalam)
        g_mu = -weights * ((ftag - mu) + d_logtau_d_etamu)

        grad_a_full = np.bincount(home_idx, weights=g_lam, minlength=n_teams) + np.bincount(
            away_idx, weights=g_mu, minlength=n_teams
        )
        grad_b_full = np.bincount(away_idx, weights=g_lam, minlength=n_teams) + np.bincount(
            home_idx, weights=g_mu, minlength=n_teams
        )

        if reg > 0.0:
            grad_a_full += reg * log_attack
            grad_b_full += reg * b_centered

        grad = np.empty_like(theta)
        # Chain rule for constraint a_{N-1} = -sum_{j=0}^{N-2} a_j
        grad[: n_teams - 1] = grad_a_full[:-1] - grad_a_full[-1]
        grad[n_teams - 1 : 2 * n_teams - 1] = grad_b_full
        grad[2 * n_teams - 1] = float(np.sum(g_lam))
        grad[2 * n_teams] = -float(np.dot(weights, d_logtau_d_rho))

        return nll, grad

    def fit(self, df: pd.DataFrame, reference_date: date | None = None) -> Self:
        """Fit Dixon-Coles parameters on historical matches using L-BFGS-B."""
        required = {"home_team_id", "away_team_id", "fthg", "ftag", "match_date"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DixonColesModel.fit missing required columns: {missing}")
        if df.empty:
            raise ValueError("Cannot fit DixonColesModel on an empty DataFrame.")

        teams = sorted(set(df["home_team_id"].astype(str)) | set(df["away_team_id"].astype(str)))
        n_teams = len(teams)
        if n_teams < 2:
            raise ValueError("DixonColesModel requires at least 2 distinct teams.")

        self.idx_to_team = teams
        self.team_to_idx = {tid: idx for idx, tid in enumerate(teams)}

        # Capture display names when present
        self.team_display_names = {}
        if "home_team" in df.columns:
            for tid, name in zip(df["home_team_id"], df["home_team"], strict=True):
                self.team_display_names[str(tid)] = str(name)
        if "away_team" in df.columns:
            for tid, name in zip(df["away_team_id"], df["away_team"], strict=True):
                self.team_display_names.setdefault(str(tid), str(name))

        home_idx = df["home_team_id"].astype(str).map(self.team_to_idx).to_numpy(dtype=np.int64)
        away_idx = df["away_team_id"].astype(str).map(self.team_to_idx).to_numpy(dtype=np.int64)
        fthg = df["fthg"].to_numpy(dtype=np.int64)
        ftag = df["ftag"].to_numpy(dtype=np.int64)

        dates = pd.to_datetime(df["match_date"]).dt.date
        ref_date = reference_date or dates.max()
        self.reference_date_ = ref_date
        delta_days = np.array([(ref_date - d).days for d in dates], dtype=np.float64)
        delta_days = np.maximum(delta_days, 0.0)
        weights = np.exp(-float(self.config.xi) * delta_days)

        # Build initial vector (with optional warm-start from previous walk-forward refit)
        theta0 = np.zeros(2 * n_teams + 1, dtype=np.float64)
        mean_goals = max(0.5, float(np.mean(np.concatenate([fthg, ftag]))))
        init_b = float(np.log(mean_goals))

        if self._warm_state is not None:
            prev_a: dict[str, float] = self._warm_state["log_attack"]  # type: ignore[assignment]
            prev_b: dict[str, float] = self._warm_state["log_defense"]  # type: ignore[assignment]
            a_init = np.array([prev_a.get(t, 0.0) for t in teams], dtype=np.float64)
            a_init -= np.mean(a_init)
            b_init = np.array([prev_b.get(t, init_b) for t in teams], dtype=np.float64)
            theta0[: n_teams - 1] = a_init[:-1]
            theta0[n_teams - 1 : 2 * n_teams - 1] = b_init
            theta0[2 * n_teams - 1] = float(self._warm_state["home_adv"])  # type: ignore[arg-type]
            theta0[2 * n_teams] = float(self._warm_state["rho"])  # type: ignore[arg-type]
        else:
            theta0[n_teams - 1 : 2 * n_teams - 1] = init_b
            theta0[2 * n_teams - 1] = 0.25
            theta0[2 * n_teams] = -0.05

        bounds: list[tuple[float | None, float | None]] = (
            [(-2.5, 2.5)] * (n_teams - 1)
            + [(-2.5, 2.5)] * n_teams
            + [(-1.0, 1.5), (-0.25, 0.25)]
        )

        result = minimize(
            fun=self._neg_log_likelihood_and_grad,
            x0=theta0,
            args=(home_idx, away_idx, fthg, ftag, weights, n_teams),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": self.config.max_iter, "ftol": 1e-10},
        )

        self.log_attack_, self.log_defense_, self.home_adv_, self.rho_ = self._unpack_params(
            result.x, n_teams
        )
        self.converged_ = bool(result.success)
        self.n_iter_ = int(result.nit)
        self.nll_ = float(result.fun)

        if not self.converged_:
            logger.warning(
                "Dixon-Coles optimizer did not fully converge (nit=%d, msg=%s)",
                self.n_iter_,
                result.message,
            )
        else:
            logger.debug(
                "Dixon-Coles converged in %d iters (nll=%.2f, gamma=%.4f, rho=%.4f)",
                self.n_iter_,
                self.nll_,
                self.home_adv_,
                self.rho_,
            )
        return self

    def expected_goals(self, home_team_id: str, away_team_id: str) -> tuple[float, float]:
        """Return expected Poisson intensities (lambda_home, mu_away) for a matchup."""
        if not self.idx_to_team:
            raise RuntimeError("DixonColesModel must be fitted before calling expected_goals.")

        mean_b = float(np.mean(self.log_defense_))
        a_h = (
            float(self.log_attack_[self.team_to_idx[home_team_id]])
            if home_team_id in self.team_to_idx
            else 0.0
        )
        b_h = (
            float(self.log_defense_[self.team_to_idx[home_team_id]])
            if home_team_id in self.team_to_idx
            else mean_b
        )
        a_a = (
            float(self.log_attack_[self.team_to_idx[away_team_id]])
            if away_team_id in self.team_to_idx
            else 0.0
        )
        b_a = (
            float(self.log_defense_[self.team_to_idx[away_team_id]])
            if away_team_id in self.team_to_idx
            else mean_b
        )

        lam = float(np.exp(np.clip(a_h + b_a + self.home_adv_, -4.0, 2.5)))
        mu = float(np.exp(np.clip(a_a + b_h, -4.0, 2.5)))
        return lam, mu

    def predict_score_matrix(
        self,
        home_team_id: str,
        away_team_id: str,
        max_goals: int | None = None,
    ) -> np.ndarray:
        """Compute normalized (max_goals+1, max_goals+1) scoreline probability matrix."""
        k = self.config.max_goals if max_goals is None else max_goals
        lam, mu = self.expected_goals(home_team_id, away_team_id)

        goals = np.arange(0, k + 1, dtype=np.int64)
        p_home_goals = poisson.pmf(goals, lam)
        p_away_goals = poisson.pmf(goals, mu)
        matrix = np.outer(p_home_goals, p_away_goals)

        # Apply exact Dixon-Coles low-score correlation adjustment
        matrix[0, 0] *= max(0.0, 1.0 - lam * mu * self.rho_)
        matrix[1, 0] *= max(0.0, 1.0 + mu * self.rho_)
        matrix[0, 1] *= max(0.0, 1.0 + lam * self.rho_)
        matrix[1, 1] *= max(0.0, 1.0 - self.rho_)

        matrix = np.clip(matrix, 0.0, None)
        total = float(matrix.sum())
        if total <= 0.0:
            raise RuntimeError("Degenerate score matrix with zero total probability.")
        return matrix / total

    def predict_markets(
        self,
        home_team_id: str,
        away_team_id: str,
        max_goals: int | None = None,
        top_n: int = 5,
    ) -> tuple[MarketProbabilities, np.ndarray]:
        """Derive 1X2, Over/Under 2.5, BTTS, and Top-N exact scorelines from the score matrix."""
        lam, mu = self.expected_goals(home_team_id, away_team_id)
        mat = self.predict_score_matrix(home_team_id, away_team_id, max_goals=max_goals)
        n = mat.shape[0]
        i_idx, j_idx = np.indices((n, n))

        p_home = float(mat[i_idx > j_idx].sum())
        p_draw = float(mat[i_idx == j_idx].sum())
        p_away = float(mat[i_idx < j_idx].sum())
        p_1x2 = normalize_probabilities(np.array([p_home, p_draw, p_away]))

        p_over_25 = float(mat[(i_idx + j_idx) >= 3].sum())
        p_btts_yes = float(mat[(i_idx >= 1) & (j_idx >= 1)].sum())

        flat_order = np.argsort(mat.ravel())[::-1][:top_n]
        top_scores: list[ScorelineProbability] = []
        for flat_idx in flat_order:
            hg, ag = divmod(int(flat_idx), n)
            top_scores.append(
                ScorelineProbability(
                    score=f"{hg}-{ag}",
                    home_goals=hg,
                    away_goals=ag,
                    probability=float(mat[hg, ag]),
                )
            )

        markets = MarketProbabilities(
            p_home=float(p_1x2[0]),
            p_draw=float(p_1x2[1]),
            p_away=float(p_1x2[2]),
            p_over_25=float(np.clip(p_over_25, 0.0, 1.0)),
            p_under_25=float(np.clip(1.0 - p_over_25, 0.0, 1.0)),
            p_btts_yes=float(np.clip(p_btts_yes, 0.0, 1.0)),
            p_btts_no=float(np.clip(1.0 - p_btts_yes, 0.0, 1.0)),
            expected_home_goals=lam,
            expected_away_goals=mu,
            top_scorelines=top_scores,
        )
        return markets, mat

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict 1X2 probabilities [P(H), P(D), P(A)] for every row in `df`."""
        probs = np.empty((len(df), 3), dtype=np.float64)
        for idx, rec in enumerate(df.itertuples(index=False)):
            mat = self.predict_score_matrix(str(rec.home_team_id), str(rec.away_team_id))
            n = mat.shape[0]
            i_idx, j_idx = np.indices((n, n))
            probs[idx, 0] = mat[i_idx > j_idx].sum()
            probs[idx, 1] = mat[i_idx == j_idx].sum()
            probs[idx, 2] = mat[i_idx < j_idx].sum()
        return normalize_probabilities(probs)

    def get_team_ratings(self) -> pd.DataFrame:
        """Return a DataFrame of estimated attack (alpha) and defense (beta) power ratings."""
        if not self.idx_to_team:
            return pd.DataFrame()
        rows = []
        for idx, tid in enumerate(self.idx_to_team):
            a_log = float(self.log_attack_[idx])
            b_log = float(self.log_defense_[idx])
            rows.append(
                {
                    "team_id": tid,
                    "team": self.team_display_names.get(tid, tid),
                    "attack_alpha": float(np.exp(a_log)),
                    "defense_beta": float(np.exp(b_log)),
                    "log_attack": a_log,
                    "log_defense": b_log,
                    # Higher overall_power = stronger attack & lower goals conceded (lower beta)
                    "overall_power": a_log - b_log,
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("overall_power", ascending=False)
            .reset_index(drop=True)
        )
