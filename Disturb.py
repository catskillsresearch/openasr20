from copy import deepcopy
from nemo.collections.asr.parts import perturb, segment

class Disturb:

    def __init__(self, _sample_rate):
        self.sample_rate = _sample_rate
        self.white_noise = perturb.WhiteNoisePerturbation(min_level=-70, max_level=-35)
        self.speed = perturb.SpeedPerturbation(self.sample_rate, 'kaiser_best', min_speed_rate=0.8, max_speed_rate=1.2, num_rates=-1)
        self.time_stretch = perturb.TimeStretchPerturbation(min_speed_rate=0.8, max_speed_rate=1.2, num_rates=3)

    def __call__(self, _sample):
        sample=deepcopy(_sample)
        self.time_stretch.perturb(sample)
        self.speed.perturb(sample)
        self.white_noise.perturb(sample)
        return sample
