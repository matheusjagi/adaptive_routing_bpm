"""Ambiente semissintético de deriva controlada para o experimento E2.

Desenho (não circular):
- k ramos roteáveis, cada um com custo VERDADEIRO médio mu_b(t) que varia no tempo
  segundo um padrão de deriva injetado (estacionário, choque abrupto, deriva gradual,
  oscilação), calibrado na variação real do log (custos ~100-260h).
- Fluxo de ANÚNCIOS por ramo (informação completa, como no log observacional real): para
  cada ramo, casos chegam (Poisson) e resolvem após seu custo (atraso de anúncio). Isso
  gera, por ramo, uma série (tempo_resolução, custo) que as políticas consultam.
- O custo só é conhecido quando o caso resolve (atraso = duração), preservando o cerne do
  problema. A escolha da política determina o custo INCORRIDO (para o regret), não o que é
  observado — todos os ramos são continuamente observados, como no experimento real.
- Como conhecemos mu_b(t), o regret é medido contra o ÓTIMO REAL, não um oráculo estimado.

Unidades: tempo em horas; custo em horas.
"""

import numpy as np

DAY = 24.0


def mean_fn(drift_type, base, amp, period, k):
    """Retorna mu(t) -> array [n, k] com o custo médio verdadeiro por ramo.
    base: [k] médias-base (h); amp: amplitude da deriva (h); period: escala temporal (h)."""
    base = np.asarray(base, dtype=float)

    if drift_type == "stationary":
        def mu(t):
            t = np.atleast_1d(t).astype(float)
            return np.tile(base, (t.shape[0], 1))

    elif drift_type == "abrupt":
        # o líder-base (ramo 0) congestiona (+amp) em janelas periódicas (1/3 do período),
        # cedendo a liderança ao ramo 1.
        def mu(t):
            t = np.atleast_1d(t).astype(float)
            m = np.tile(base, (t.shape[0], 1))
            shocked = np.mod(t, period) < (period / 3.0)
            m[shocked, 0] = base[0] + amp
            return m

    elif drift_type == "gradual":
        # senoides defasadas: a liderança rotaciona suavemente entre os ramos.
        phases = np.linspace(0, 2 * np.pi, k, endpoint=False)
        def mu(t):
            t = np.atleast_1d(t).astype(float)[:, None]
            return base[None, :] + amp * np.sin(2 * np.pi * t / period + phases[None, :])

    elif drift_type == "oscillating":
        # troca de regime abrupta: liderança alterna entre ramo 0 e ramo 1 a cada meio-período.
        def mu(t):
            t = np.atleast_1d(t).astype(float)
            m = np.tile(base, (t.shape[0], 1))
            hi = np.mod(t, period) < (period / 2.0)
            m[hi, 0] = base[0] + amp
            m[~hi, 1] = base[1] + amp
            return m
    else:
        raise ValueError(drift_type)

    return mu


def _lognormal_costs(mean_h, cv, rng):
    """Custos lognormais com média = mean_h e coef. de variação cv (cauda pesada)."""
    mean_h = np.maximum(mean_h, 1e-3)
    sigma2 = np.log(1.0 + cv * cv)
    sigma = np.sqrt(sigma2)
    mu = np.log(mean_h) - 0.5 * sigma2
    return rng.lognormal(mu, sigma)


class DriftEnv:
    def __init__(self, drift_type, horizon_h, k=3,
                 base=(110.0, 125.0, 140.0), amp=100.0, period_h=60 * DAY,
                 ann_rate_per_h=1.25, cv=1.0, seed=0):
        self.k = k
        self.horizon = horizon_h
        self.mu = mean_fn(drift_type, base[:k], amp, period_h, k)
        rng = np.random.default_rng(seed)

        self.res = []       # resolução (ordenado por resolução) por ramo
        self.res_csum = []  # cumsum de custo na ordem de resolução
        self.a_arr = []     # chegada (ordenado por chegada) por ramo
        self.a_res = []     # resolução correspondente (ordem de chegada)
        for b in range(k):
            n = rng.poisson(ann_rate_per_h * horizon_h)
            a = np.sort(rng.uniform(0, horizon_h, size=n))
            c = _lognormal_costs(self.mu(a)[:, b], cv, rng)
            r = a + c
            # ordenado por resolução (para média dos resolvidos)
            o = np.argsort(r)
            self.res.append(r[o])
            self.res_csum.append(np.concatenate([[0.0], np.cumsum(c[o])]))
            # ordenado por chegada (para casos abertos / triggered updates)
            self.a_arr.append(a)
            self.a_res.append(r)

    def announced(self, b, t, W):
        """T_b(t): média dos últimos W casos do ramo b resolvidos até t."""
        res = self.res[b]
        hi = np.searchsorted(res, t, side="right")
        lo = max(0, hi - W)
        n = hi - lo
        return np.nan if n <= 0 else (self.res_csum[b][hi] - self.res_csum[b][lo]) / n

    def open_elapsed(self, b, t, delta):
        """Tempo médio decorrido dos casos do ramo b que entraram nos últimos `delta` h e
        ainda estão abertos em t (para o triggered update)."""
        a = self.a_arr[b]
        lo = np.searchsorted(a, t - delta, side="left")
        hi = np.searchsorted(a, t, side="right")
        if hi <= lo:
            return 0.0
        arr = a[lo:hi]
        res = self.a_res[b][lo:hi]
        openmask = res > t
        if not openmask.any():
            return 0.0
        return float(np.mean(t - arr[openmask]))

    def true_means(self, t):
        return self.mu(np.array([t]))[0]
