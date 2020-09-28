import numpy as np
import matplotlib.pylab as plt
import scipy.stats as stats

def plot_log_population2(population_old, population_new, _title, _xlabel, _ylabel, _bins):
    try:
        if np.allclose(population_old, population_new):
            return
    except:
        pass
    plt.figure(figsize=(10,5))

    if False:
        n, x, _ = plt.hist(population_old, bins=_bins, histtype=u'step', density=True)  
        density_old = stats.gaussian_kde(population_old)
        plt.plot(x, density_old(x), color='green', label='old')

        n, x, _ = plt.hist(population_new, bins=_bins, histtype=u'step', density=True)  
        density_new = stats.gaussian_kde(population_new)
        plt.plot(x, density_new(x), color='blue', label='new')
    else:
        plt.hist(population_old,bins=_bins,label='old', color='red', histtype=u'step')
        plt.hist(population_new,bins=_bins,label='new', color='blue', histtype=u'step')
    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(_title)
    plt.yscale('log');
    plt.legend()
    plt.show()
