import math
import numpy as np
from rilacs.strategies import (
    apriori_Kelly_bet,
    get_conv_weights_from_dist,
    square_gamma_dist,
)
from abc import ABC, abstractmethod


class Bettor(ABC):
    def __init__(self):
        self.t = 0

    @abstractmethod
    def update_bet(self, ballot, m_t):
        pass


class AbstractKelly_Bettor(Bettor):
    def __init__(self, conv_weights=np.array([1])):
        super().__init__()
        self.conv_weights = conv_weights

    @abstractmethod
    def update_bet(self, ballot, m_t):
        pass


class DistKelly_Bettor(AbstractKelly_Bettor):
    def __init__(self, dist=square_gamma_dist, D=10, c=1):
        conv_weights = get_conv_weights_from_dist(dist, D=D)
        assert np.abs(np.sum(conv_weights) - 1) < 10e-12
        super().__init__(conv_weights=conv_weights)
        self.c = c

    def update_bet(self, ballot, m_t):
        self.t += 1
        D = len(self.conv_weights)
        d = np.arange(1, D + 1)
        # Create a matrix of bets by taking the
        # outer product between d and 1/((D+1)*m_t)
        with np.errstate(divide="ignore"):
            bets = d[:, None] / ((D + 1) * m_t)
            trunc_bets = np.maximum(0, np.minimum(bets, self.c / m_t))
        return trunc_bets


class Kelly_Bettor(AbstractKelly_Bettor):
    def __init__(self, n_A, n_B, breaks=500, c=1 / 2):
        super().__init__(conv_weights=np.array([1]))

        self.c = c
        self.bets = self._compute_kelly_bets(n_A, n_B, breaks)

    def _compute_kelly_bets(self, n_A, n_B, breaks):
        apriori_bet = apriori_Kelly_bet(n_A=n_A, n_B=n_B)
        bets = np.array([np.repeat(apriori_bet, breaks + 1)])

        return apriori_bet

    def update_bet(self, ballot, m_t):
        self.t += 1
        with np.errstate(divide="ignore"):
            bet = np.maximum(0, np.minimum(self.bets, self.c / m_t))
        return bet


class Audit(ABC):
    def __init__(self, N, alpha=0.05):
        self.l = 0
        self.l_logical = 0
        self.S_t = 0
        self.N = N
        self.alpha = alpha
        self.t = 0

    def _update_l_logical(self, ballot):
        # accumulating S_t instead of S_t / N for numerical stability
        self.S_t += ballot
        self.l_logical = self.S_t / self.N
        return self.l_logical

    @abstractmethod
    def update_cs(self, ballot: float) -> float:
        pass


class Betting_Audit(Audit):
    def __init__(self, N, bettor=DistKelly_Bettor(), breaks=500, alpha=0.05):
        super().__init__(N=N, alpha=alpha)
        self.breaks = breaks
        self.bettor = bettor
        self.m = np.arange(0, 1 + 1 / self.breaks, step=1 / self.breaks)
        self.m_t = np.arange(0, 1 + 1 / self.breaks, step=1 / self.breaks)
        self.capitals = 1

    def _update_m_t(self, ballot):
        if self.N != self.t:
            self.m_t = ((self.N - self.t + 1) * self.m_t - ballot) / (self.N - self.t)
        else:
            # If N == t, then we're at the final step and don't need to take any action
            pass

        return self.m_t

    def update_cs(self, ballot):
        bets = self.bettor.update_bet(ballot, self.m_t)
        self.t += 1
        assert self.t == self.bettor.t

        # Update the capital for each m_t
        # First compute capital process for each of the convex weights
        with np.errstate(invalid="ignore", over="ignore"):
            self.capitals *= np.where(
                self.m_t < 0, math.inf, 1 + bets * (ballot - self.m_t)
            )

        # Take convex combination of capitals to get capital
        convex_weights = self.bettor.conv_weights
        with np.errstate(invalid="ignore"):
            capital = np.sum(convex_weights[:, None] * self.capitals, axis=0)
        # Find the smallest m_t for which the capital is less than 1/alpha,
        # and take a superset
        l_idx = np.where(capital < 1 / self.alpha)[0][0]
        # l_idx = l_idx - 1 if l_idx != 0 else l_idx
        self.l = self.m[l_idx] - 1 / self.breaks

        # Also update logical lower confidence sequence and intersect l with it
        self._update_l_logical(ballot)
        self.l = max(self.l, self.l_logical)

        self._update_m_t(ballot)

        return self.l


class Hoeffding_Audit(Audit):
    def __init__(self, N, alpha=0.05, mu_init=1 / 2):
        super().__init__(N=N, alpha=alpha)
        self.mu = mu_init
        self.margin_numerator = np.log(1 / self.alpha)
        self.margin_denominator = 0
        self.margin = 0
        self.S_tminus1 = 0
        self.mu_numerator = 0
        self.mu_denominator = 0

    def _update_mu(self, ballot):
        l = self._get_lambda(self.t)
        self.mu_numerator += l * (ballot + 1 / (self.N - self.t + 1) * self.S_tminus1)
        self.mu_denominator += l * (1 + (self.t - 1) / (self.N - self.t + 1))
        self.mu = self.mu_numerator / self.mu_denominator

        return self.mu

    def _update_margin(self, ballot):
        l = self._get_lambda(self.t)
        self.margin_numerator += l**2 / 8
        self.margin_denominator += l * (1 + (self.t - 1) / (self.N - self.t + 1))
        self.margin = self.margin_numerator / self.margin_denominator
        return self.margin

    def _get_lambda(self, t):
        return np.minimum(np.sqrt(8 * np.log(1 / self.alpha) / t / np.log(t + 1)), 1)

    def _update_S_tminus1(self, ballot):
        self.S_tminus1 += ballot

    def update_cs(self, ballot):
        self.t += 1
        mu = self._update_mu(ballot)
        margin = self._update_margin(ballot)
        l_logical = self._update_l_logical(ballot)

        self.l = np.maximum(mu - margin, l_logical)

        self._update_S_tminus1(ballot)

        return self.l


class Empirical_Bernstein_Audit(Audit):
    def __init__(self, N, alpha=0.05, mu_init=1 / 2):
        super().__init__(N=N, alpha=alpha)
        self.mu = mu_init
        self.margin_numerator = np.log(1 / self.alpha)
        self.margin_denominator = 0
        self.margin = 0
        self.S_tminus1 = 0
        self.mu_numerator = 0
        self.mu_denominator = 0

    def _update_mu(self, ballot):
        l = self._get_lambda(self.t)
        self.mu_numerator += l * (ballot + 1 / (self.N - self.t + 1) * self.S_tminus1)
        self.mu_denominator += l * (1 + (self.t - 1) / (self.N - self.t + 1))
        self.mu = self.mu_numerator / self.mu_denominator

        return self.mu

    def _update_margin(self, ballot):
        l = self._get_lambda(self.t)
        self.margin_numerator += l**2 / 8
        self.margin_denominator += l * (1 + (self.t - 1) / (self.N - self.t + 1))
        self.margin = self.margin_numerator / self.margin_denominator
        return self.margin

    def _get_lambda(self, t):
        return np.minimum(np.sqrt(8 * np.log(1 / self.alpha) / t / np.log(t + 1)), 1)

    def _update_S_tminus1(self, ballot):
        self.S_tminus1 += ballot

    def update_cs(self, ballot):
        self.t += 1
        mu = self._update_mu(ballot)
        margin = self._update_margin(ballot)
        l_logical = self._update_l_logical(ballot)

        self.l = np.maximum(mu - margin, l_logical)

        self._update_S_tminus1(ballot)

        return self.l


def lower_cs_from_audit(x: np.ndarray, audit: Audit) -> np.ndarray:
    l = np.zeros(len(x))
    for t in range(len(x)):
        audit.update_cs(x[t])
        l[t] = audit.l
    return l
