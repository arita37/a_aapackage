
    def _subnetworklstm(self, x, i):
        with tf.variable_scope('Global_RNN', reuse=i > 0):
            lstm = tf.nn.rnn_cell.LSTMCell(self._config.n_hidden_lstm, name='lstm_cell', reuse=i > 0)
            x, s = tf.nn.static_rnn(lstm, x, dtype=TF_DTYPE)
            x = tf.layers.dense(x[-1], self._config.num_hiddens[-1], name='dense_out', reuse=i > 0)
            return x

    def _subnetwork(self, x, name):
        """
          FeedForward network, connected by Input.
          FInal layer no Activation function,
          dim : nb of assets.
          num_hiddens = [dim, dim+10, dim+10, dim]
        """
        with tf.variable_scope(name):
            # standardize the path input first
            # the affine  could be redundant, but helps converge faster
            hiddens = self._batch_norm(x, name="path_input_norm")
            for i in range(1, len(self._config.num_hiddens) - 1):
                hiddens = self._dense_batch_layer(
                    hiddens,
                    self._config.num_hiddens[i],
                    activation_fn=tf.nn.relu,
                    name="layer_{}".format(i),
                )
            output = self._dense_batch_layer(
                hiddens, self._config.num_hiddens[-1], activation_fn=None, name="final_layer"
            )
        return output

    def _dense_batch_layer(
            self, input_, output_size, activation_fn=None, stddev=5.0, name="linear"
    ):
        with tf.variable_scope(name):
            shape = input_.get_shape().as_list()
            weight = tf.get_variable(
                "Matrix",
                [shape[1], output_size],
                TF_DTYPE,
                tf.random_normal_initializer(stddev=stddev / np.sqrt(shape[1] + output_size)),
            )
            hiddens = tf.matmul(input_, weight)
            hiddens_bn = self._batch_norm(hiddens)

        if activation_fn:
            return activation_fn(hiddens_bn)
        else:
            return hiddens_bn

    def _batch_norm(self, x, affine=True, name="batch_norm"):
        """Batch normalization"""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable(
                "beta",
                params_shape,
                TF_DTYPE,
                initializer=tf.random_normal_initializer(0.0, stddev=0.1, dtype=TF_DTYPE),
            )

            gamma = tf.get_variable(
                "gamma",
                params_shape,
                TF_DTYPE,
                initializer=tf.random_uniform_initializer(0.1, 0.5, dtype=TF_DTYPE),
            )

            moving_mean = tf.get_variable(
                "moving_mean",
                params_shape,
                TF_DTYPE,
                initializer=tf.constant_initializer(0.0, TF_DTYPE),
                trainable=False,
            )

            moving_variance = tf.get_variable(
                "moving_variance",
                params_shape,
                TF_DTYPE,
                initializer=tf.constant_initializer(1.0, TF_DTYPE),
                trainable=False,
            )

            # These ops will only be preformed when training
            mean, variance = tf.nn.moments(x, [0], name="moments")
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(moving_mean, mean, MOMENTUM)
            )
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(moving_variance, variance, MOMENTUM)
            )
            mean, variance = tf.cond(
                self._is_training, lambda: (mean, variance), lambda: (moving_mean, moving_variance)
            )
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, EPSILON)
            y.set_shape(x.get_shape())
            return y
            
            
            





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
