import matplotlib.pylab as plt

def plot_log_population(population, _title, _xlabel, _ylabel, _bins):
    plt.hist(population,bins=_bins)
    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(_title)
    plt.yscale('log');
    plt.show()
