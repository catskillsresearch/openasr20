import numpy as np

def sample_statistics(corpus, measurement, samples):
    return [(corpus, measurement, 'Mean', np.mean(samples)),
            (corpus, measurement, 'Median', np.median(samples)),
            (corpus, measurement, 'Min', np.min(samples)),
            (corpus, measurement, 'Max', np.max(samples))]
