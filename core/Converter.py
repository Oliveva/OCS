import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod




class InputSource(ABC):
    @abstractmethod
    def get_sample_rate(self) -> int:
        """返回输入信号的采样率（Hz）"""
        pass

    @abstractmethod
    def read_signal(self) -> np.ndarray:
        """返回信号数据数组"""
        pass


class GeneratedSignalInput(InputSource):
    def __init__(self, sample_rate=8000, duration=1, freq=10):
        self.sample_rate = sample_rate
        self.t = np.linspace(0, duration, int(sample_rate * duration))
        self.signal = np.sin(2 * np.pi * freq * self.t)

    def get_sample_rate(self) -> int:
        return self.sample_rate

    def read_signal(self) -> np.ndarray:
        return self.signal

class ADC:
    def __init__(self, sample_rate=8000, bits=16):
        self.sample_rate = sample_rate
        self.bits = bits
        self.max_val = 2 ** (bits - 1) - 1

    def convert(self, signal_input: InputSource):
        sample_rate = signal_input.get_sample_rate()
        signal_data = signal_input.read_signal()
        duration = len(signal_data) / sample_rate
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        if sample_rate!=self.sample_rate:
            signal_data=self.resample_signal(signal_data,sample_rate)
        quantized = np.int16(signal_data * self.max_val)
        return t,quantized

    def resample_signal(self, signal, src_rate):
        """重采样到目标采样率"""
        from scipy.signal import resample_poly, firwin, filtfilt
        gcd = np.gcd(src_rate, self.sample_rate)
        up = self.sample_rate // gcd
        down = src_rate // gcd
        resampled = resample_poly(signal, up, down)
        cutoff = min(src_rate // 2, self.sample_rate // 2)
        taps = firwin(101, cutoff, fs=self.sample_rate)
        return filtfilt(taps, 1.0, resampled)


if __name__ == "__main__":
    ADConverter = ADC(sample_rate=16000)
    signal=GeneratedSignalInput()
    td,sig = ADConverter.convert(signal)

    plt.subplot(211)
    plt.plot(signal.t, signal.signal)
    plt.subplot(212)
    plt.plot(td, sig)
    plt.show()
