from commpy.modulation import PSKModem
import numpy as np


class PSK(PSKModem):
    def __init__(self, mod_type: str = 'QPSK'):
        if mod_type == 'BPSK':
            super().__init__(2)
        elif mod_type == 'QPSK':
            super().__init__(4)
        else:
            raise ValueError(f"Unsupported Modulation Type: {mod_type}")

    def demodulate(self, input_symbols, demod_type='hard', noise_var=0):
        return super().demodulate(input_symbols, demod_type, noise_var)
