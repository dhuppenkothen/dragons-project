import numpy as np
import scipy.stats

def bessel_probability_corr(z, std1, std2, corr):
    """
    Probability distribution of the product of two zero-mean 
    Gaussian random variables, $Z = X1 * X2$, with standard 
    deviations `std1` and `std2`, and correlation coefficient 
    `corr`. Based on the variance-gamma distribution, as 
    suggested in Gaunt (2019).

    Called bessel probability because it includes Bessel 
    functions, and who doesn't like Bessel functions?
    
    Parameters
    ----------
    z : float or numpy.ndarray
        The product of two zero-mean, correlated Gaussian 
        random variables. Can be a single number or an array
        of these random variables.
        
    std1, std2 : float, float
        The standard deviations of the two Gaussian random 
        variables that generated `x`
        
    corr : float, [0,1]
        The Pearson correlation coefficient of X1 and X2.
    
    Returns
    -------
    prob : float in [0,1]
        The probability density for random variable $Z$
        
    Examples
    --------
    >>> cov = np.array([[2.0, 0.5],[0.5, 1.0]])
    >>> corr = 0.5 / np.sqrt(2.0)
    >>> np.random.seed(300)
    >>> [x1, x2] = scipy.stats.multivariate_normal([0,0], np.array(cov)).rvs()
    >>> z = x1*x2
    >>> bessel_probability_corr(z, cov[0,0], cov[1,1], corr)
    0.13284294893564183

    >>> [x1, x2] = scipy.stats.multivariate_normal([0,0], np.array(cov)).rvs(size=50000).T
    >>> z = x1 * x2
    >>> bessel_probability_corr(z, cov[0,0], cov[1,1], corr)
    array([0.42963344, 0.19773116, 0.45172443, ..., 0.24833808, 0.31325817,
       0.00752114])
    """
    z_abs = np.abs(z)
    both_std = std1 * std2
    corr_fac = (1.0 - corr**2.0)
    
    log_pre_fac = -np.log(np.pi) - np.log(std1) - np.log(std2) - 0.5*np.log(corr_fac)
    
    log_exp_fac = corr * z / (both_std * corr_fac)
    
    logy = np.log(z_abs) - np.log(std1) - np.log(std2) - np.log(corr_fac)
    
    order = 0
    log_bessel = np.log(scipy.special.kn(order, np.exp(logy)))

    return np.exp(log_pre_fac + log_exp_fac + log_bessel)


