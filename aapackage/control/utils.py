    












def jump_diffusion(S=1, X=0.5, T=1, mu=0.12, sigma=0.3, Lambda=0.25,
                   a=0.2, b=0.2, Nsteps=252, Nsim=100, alpha=0.05, seed=None):
    '''
    Monte Carlo simulation [1] of Merton's Jump Diffusion Model [2].
    The model is specified through the stochastic differential equation (SDE):
                        dS(t)
                        ----- = mu*dt + sigma*dW(t) + dJ(t)
                        S(t-)
    mu, sigma: constants, the drift and volatility coefficients of the stock
               price process;
    W: a standard one-dimensional Brownian motion;
    J: a jump process, independent of W, with piecewise constant sample paths.
       It is defined as the sum of multiplicative jumps Y(j).
    Input
    ---------------------------------------------------------------------------
    S: float. The current asset price.
    X: float. The strike price, i.e. the price at which the asset may be bought
       (call) or sold (put) in an option contract [3].
    T: int or float. The maturity of the option contract, i.e. the final
       monitoring date.
    mu, sigma: float. Respectively, the drift and volatility coefficients of
               the asset price process.
    Lambda: float. The intensity of the Poisson process in the jump diffusion
            model ('lambda' is a protected keyword in Python).
    a, b: float. Parameters required to calculate, respectively, the mean and
          variance of a standard lognormal distribution, log(x) ~ N(a, b**2).
          (see code).
    Nsteps: int. The number of monitoring dates, i.e. the time steps.
    Nsim: int. The number of Monte Carlo simulations (at least 10,000 required
          to generate stable results).
    alpha: float. The confidence interval significance level, in [0, 1].
    seed: int. Set random seed, for reproducibility of the results. Default
          value is None (the best seed available is used, but outcome will vary
          in each experiment).
    '''

    # Import required libraries
    import time
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set random seed
    np.random.seed(seed)

    '''
    Time the whole path-generating process, using a tic-toc method familiar
    to MATLAB users
    '''
    tic = time.time()

    # Calculate the length of the time step
    Delta_t = T/Nsteps

    '''
    Compute mean and variance of a standard lognormal distribution from user
    defined parameters a and b. The latter are useful to simulate the jump
    component in Monte Carlo.
    a and b are chosen such that log(Y(j)) ~ N(a, b**2). This implies that the
    mean and variance of the multiplicative jumps will be:
     * mean_Y = np.exp(a + 0.5*(b**2))
     * variance_Y = np.exp(2*a + b**2) * (np.exp(b**2)-1)
    '''
    mean_Y = np.exp(a + 0.5*(b**2))
    variance_Y = np.exp(2*a + b**2) * (np.exp(b**2)-1)

    '''
    Calculate the theoretical drift (M) and volatility (V) of the stock price
    process under Merton's jump diffusion model. These values can be used to
    monitor the rate of convergence of Monte Carlo estimates as the number of
    simulated experiments increases, and can help spot errors, if any, in
    implementing the model.
    '''
    M = S * np.exp(mu*T + Lambda*T*(mean_Y-1))
    V = S**2 * (np.exp((2*mu + sigma**2)*T \
        + Lambda*T*(variance_Y + mean_Y**2 - 1)) \
        - np.exp(2*mu*T + 2*Lambda*T*(mean_Y - 1)))

    '''
    Generate an Nsim x (Nsteps+1) array of zeros to preallocate the simulated
    paths of the Monte Carlo simulation. Each row of the matrix represents a
    full, possible path for the stock, each column all values of the asset at
    a particular instant in time.
    '''
    simulated_paths = np.zeros([Nsim, Nsteps+1])

    # Replace the first column of the array with the vector of initial price S
    simulated_paths[:,0] = S

    '''
    To account for the multiple sources of uncertainty in the jump diffusion
    process, generate three arrays of random variables.
     - The first one is related to the standard Brownian motion, the component
       epsilon(0,1) in epsilon(0,1) * np.sqrt(dt);
     - The second and third ones model the jump, a compound Poisson process:
       the former (a Poisson process with intensity Lambda) causes the asset
       price to jump randomly (random timing); the latter (a Gaussian variable)
       defines both the direction (sign) and intensity (magnitude) of the jump.
    '''
    Z_1 = np.random.normal(size=[Nsim, Nsteps])
    Z_2 = np.random.normal(size=[Nsim, Nsteps])
    Poisson = np.random.poisson(Lambda*Delta_t, [Nsim, Nsteps])

    # Populate the matrix with Nsim randomly generated paths of length Nsteps
    for i in range(Nsteps):
        simulated_paths[:,i+1] = simulated_paths[:,i]*np.exp((mu
                               - sigma**2/2)*Delta_t + sigma*np.sqrt(Delta_t) \
                               * Z_1[:,i] + a*Poisson[:,i] \
                               + np.sqrt(b**2) * np.sqrt(Poisson[:,i]) \
                               * Z_2[:,i])

    # Single out array of simulated prices at maturity T
    final_prices = simulated_paths[:,-1]

    # Compute mean, variance, standard deviation, skewness, excess kurtosis
    mean_jump = np.mean(final_prices)
    var_jump = np.var(final_prices)
    std_jump = np.std(final_prices)
    skew_jump = stats.skew(final_prices)
    kurt_jump = stats.kurtosis(final_prices)

    # Calculate confidence interval for the mean
    ci_low = mean_jump - std_jump/np.sqrt(Nsim)*stats.norm.ppf(1-0.5*alpha)
    ci_high = mean_jump + std_jump/np.sqrt(Nsim)*stats.norm.ppf(1-0.5*alpha)



    return simulated_paths
    
    
    
    
    
    
    
    
    
   
   
   
   