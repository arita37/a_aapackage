class StochasticProcess(object):
    @classmethod
    def random(cls):
        return cls()

    @property
    def diffusion_driver(self):
        """
            diffusion driver are the underlying `dW` of each process `X` in a SDE like `dX = m dt + s dW`
        :return list(StochasticProcess):
        """
        if self._diffusion_driver is None:
            return (self,)
        if isinstance(self._diffusion_driver, list):
            return tuple(self._diffusion_driver)
        if isinstance(self._diffusion_driver, tuple):
            return self._diffusion_driver
        return (self._diffusion_driver,)  # return as a tuple

    def __init__(self, start):
        self.start = start
        self._diffusion_driver = self

    def __len__(self):
        return len(self.diffusion_driver)

    def __str__(self):
        return self.__class__.__name__ + "()"

    def evolve(self, x, s, e, q):
        """
        :param float x: current state value, i.e. value before evolution step
        :param float s: current point in time, i.e. start point of next evolution step
        :param float e: next point in time, i.e. end point of evolution step
        :param float q: standard normal random number to do step
        :return float: next state value, i.e. value after evolution step
        evolves process state `x` from `s` to `e` in time depending of standard normal random variable `q`
        """
        return 0.0

    def mean(self, t):
        """ expected value of time :math:`t` increment
        :param t:
        :rtype float
        :return:
        """
        return 0.0

    def variance(self, t):
        """ second central moment of time :math:`t` increment
        :param t:
        :rtype float
        :return:
        """
        return 0.0


class MultivariateStochasticProcess(StochasticProcess):
    pass


class WienerProcess(StochasticProcess):
    """
    class implementing general Gauss process between grid dates
    """

    def __init__(self, mu=0.0, sigma=1.0, start=0.0):
        super(WienerProcess, self).__init__(start)
        self._mu = mu
        self._sigma = sigma

    def __str__(self):
        return "N(mu=%0.4f, sigma=%0.4f)" % (self._mu, self._sigma)

    def _drift(self, x, s, e):
        return self._mu * (e - s)

    def _diffusion(self, x, s, e):
        return self._sigma * sqrt(e - s)

    def evolve(self, x, s, e, q):
        return x + self._drift(x, s, e) + self._diffusion(x, s, e) * q

    def mean(self, t):
        return self.start + self._mu * t

    def variance(self, t):
        return self._sigma ** 2 * t


class GeometricBrownianMotion(WienerProcess):
    """
    class implementing general Gauss process between grid dates
    """

    def __init__(self, mu=0.0, sigma=1.0, start=1.0):
        super(GeometricBrownianMotion, self).__init__(mu, sigma, start)
        self._diffusion_driver = super(GeometricBrownianMotion, self).diffusion_driver

    def __str__(self):
        return "LN(mu=%0.4f, sigma=%0.4f)" % (self._mu, self._sigma)

    def evolve(self, x, s, e, q):
        return x * exp(super(GeometricBrownianMotion, self).evolve(0.0, s, e, q))

    def mean(self, t):
        return self.start * exp(self._mu * t)

    def variance(self, t):
        return self.start ** 2 * exp(2 * self._mu * t) * (exp(self._sigma ** 2 * t) - 1)


class MultiGauss(Object):
    """
    class implementing multi dimensional brownian motion
    """

    def __init__(self, mu=list([0.0]), covar=list([[1.0]]), start=list([0.0])):
        super(MultiGauss, self).__init__(start)
        self._mu = mu
        self._dim = len(start)
        self._cholesky = None if covar is None else cholesky(covar).T
        self._variance = (
            [1.0] * self._dim if covar is None else [covar[i][i] for i in range(self._dim)]
        )
        self._diffusion_driver = [
            WienerProcess(m, sqrt(s)) for m, s in zip(self._mu, self._variance)
        ]

    def __str__(self):
        cov = self._cholesky.T * self._cholesky
        return "%d-MultiGauss(mu=%s, cov=%s)" % (len(self), str(self._mu), str(cov))

    def _drift(self, x, s, e):
        return [m * (e - s) for m in self._mu]

    def evolve(self, x, s, e, q):
        dt = sqrt(e - s)
        q = [qq * dt for qq in q]
        q = list(self._cholesky.dot(q))
        d = self._drift(x, s, e)
        return [xx + dd + qq for xx, dd, qq in zip(x, d, q)]

    def mean(self, t):
        return [s + m * t for s, m in zip(self.start, self._mu)]

    def variance(self, t):
        return [v * t for v in self._variance]
