import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, firwin, filtfilt, lfilter
from abc import ABC, abstractmethod


class InputSource(ABC):
    """
    The abstract base class of the input source.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sampling rate of input signal (Hz)"""
        pass

    @property
    @abstractmethod
    def signal(self) -> np.ndarray:
        """Array of signal data"""
        pass

    @property
    @abstractmethod
    def t(self) -> np.ndarray:
        """Array of time"""
        pass


class GeneratedSignalInput(InputSource):
    """
    An input that uses a generated signal as an input source.

    The signal will be a sin signal.

    Attributes:
        sample_rate (int):      The sampling rate of Generated Signal.(@property)
        signal (np.ndarray):    The data array of the Generated Signal.(@property)
        t (np.ndarray):         The time array of the Generated Signal.(@property)

    Args:
        sample_rate (int):      The sampling rate of Generated Signal.
        duration (float):       The duration of the Generated Signal.
        freq (int):             The frequency of the Generated Signal.
    """

    sample_rate: int
    signal: np.ndarray
    t: np.ndarray

    def __init__(self, sample_rate: int = 8000, duration: float = 1.0, freq: int = 10):
        self._sample_rate = sample_rate
        self._t = np.linspace(0, duration, int(sample_rate * duration))
        self._signal = np.sin(2 * np.pi * freq * self._t)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def signal(self) -> np.ndarray:
        return self._signal

    @property
    def t(self) -> np.ndarray:
        return self._t


class WavFileInput(InputSource):
    """
    An input that uses a .wav file as an input source.

    Attributes:
        sample_rate (int):      The sampling rate of .wav Signal.(@property)
        signal (np.ndarray):    The data array of the .wav Signal.(@property)
        t (np.ndarray):         The time array of the .wav Signal.(@property)

    Args:
        file_path (str):        The path of .wav file.
    """

    sample_rate: int
    signal: np.ndarray
    t: np.ndarray

    def __init__(self, file_path: str):
        self._sample_rate, self._signal = wavfile.read(file_path)
        if self._signal.ndim == 2:
            self._signal = self._signal[:, 0]

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def signal(self) -> np.ndarray:
        return self._signal.astype(np.float32) / np.iinfo(self._signal.dtype).max

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self._signal)) / self._sample_rate




class ADC:
    """
    Analog-to-digital converter.

    Attributes:
        sample_rate (int):     The sampling rate of ADC.
        _bits (int):            The number of digits quantized.
        _max_val (int):         The maximum value after quantization。

    Args:
        sample_rate (int):      The sampling rate of ADC, should be bigger than twice of the
                                frequency of input signal generally.
        bits (int):             The number of digits quantized, generally 8 or 16.
    """
    sample_rate: int
    def __init__(self, sample_rate: int = 8000, bits: int = 16):
        self._sample_rate = sample_rate
        self._bits = bits
        self._max_val = 2 ** (bits - 1) - 1

    def convert(self, signal_input: InputSource):
        """
        Convert the input analog signal to a digital signal.

        Args:
            signal_input (InputSource):     Input analog signal.

        Returns:
            Tuple[np.ndarray, np.ndarray]:  The time and data array of the digital signal.
        """

        sample_rate = signal_input.sample_rate
        signal_data = signal_input.signal
        duration = len(signal_data) / sample_rate
        if sample_rate != self._sample_rate:
            signal_data = self._resample_signal(signal_data, sample_rate)
        quantized = np.int16(signal_data * self._max_val)
        t = np.linspace(0, duration, len(quantized), endpoint=False)
        return t, quantized

    def _resample_signal(self, src_signal: np.ndarray, src_rate: int):
        """
        Internal method to resample the input signal.

        Args:
            src_signal (np.ndarray):    Signal source.
            src_rate (np.ndarray):      Original sampling rate.

        Returns:
            np.ndarray:                 The data array of the signal after resampling.
        """

        gcd = np.gcd(src_rate, self._sample_rate)
        up = self._sample_rate // gcd
        down = src_rate // gcd
        resampled = resample_poly(src_signal, up, down)
        cutoff = min(src_rate // 2, self._sample_rate // 2)
        taps = firwin(101, cutoff, fs=self._sample_rate)
        return filtfilt(taps, 1.0, resampled)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


class DAC:
    """
    Digital-to-analog converter.

    Attributes:
        _sample_rate (int):         The sampling rate of target analog signal.
        _bits (int):                The number of digits quantized.
        _max_val (int):             The maximum value after quantization.
        _bandwidth (int):           Bandwidth of the analog signal.

    Args:
        sample_rate (int):          The sampling rate of target analog signal,default 48000.
        bits (int):                 The number of digits quantized, generally 8 or 16, should be same as the ADC's.
        analog_bandwidth (int):     Bandwidth of the analog signal (for designing anti-image filters)
    """

    def __init__(self, sample_rate: int = 48000, bits: int = 16, analog_bandwidth: int = 20000):
        self._sample_rate = sample_rate
        self._bits = bits
        self._max_val = 2 ** (bits - 1) - 1
        self._bandwidth = analog_bandwidth

    def convert(self, digital_signal: np.ndarray, input_sample_rate: int):
        """
        Convert the input digital signal to a analog signal.

        Args:
            digital_signal (np.ndarray):     Input digital signal.
            input_sample_rate (int):         The sampling rate of the digital signal.

        Returns:
            Tuple[np.ndarray, np.ndarray]:   The time and data array of the analog signal.
        """

        analog = digital_signal.astype(np.float32) / self._max_val

        if input_sample_rate != self._sample_rate:
            analog = self._resample(analog, input_sample_rate)

        tap = firwin(
            numtaps=101,
            cutoff=self._bandwidth,
            fs=self._sample_rate
        )
        analog_signal = lfilter(tap, 1.0, analog)
        t = np.arange(len(analog_signal))/self._sample_rate

        return t, analog_signal

    def _resample(self, src_signal: np.ndarray, src_rate: int) -> np.ndarray:
        gcd = np.gcd(src_rate, self._sample_rate)
        up = self._sample_rate // gcd
        down = src_rate // gcd
        return resample_poly(src_signal, up, down)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ADConverter = ADC(sample_rate=90000)
    signal = WavFileInput("C:/Users/Oliver/Downloads/sample-15s.wav")
    td, sig = ADConverter.convert(signal)
    print(type(sig))

    plt.subplot(211)
    plt.plot(signal.t, signal.signal)
    plt.subplot(212)
    plt.plot(td, sig)
    plt.show()
