import numpy as np
class ADC:
    def __init__(self, sample_rate=8000, bits=16):
        self.sample_rate = sample_rate
        self.bits = bits
        self.max_val = 2**(bits - 1) - 1

    def convert(self, analog_signal):
        duration = len(analog_signal) / self.sample_rate
        t = np.linspace(0, duration, len(analog_signal), endpoint=False)
        sampled = analog_signal
        quantized = np.int16(sampled * self.max_val)
        return quantized