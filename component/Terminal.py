from component.base import BaseComponent
from core import Code, Inter, Module, Protocol, Converter
import numpy as np
import struct
from scipy.io import wavfile


class SignalTerminal_IN(BaseComponent):
    def __init__(self, config: dict):
        self.rate = config['adc_samplerate']
        self.signal = config['signal_params']
        self.pd_form = config['ip_id_mapping']
        self.src_ip = int(config['src_ip'])
        self.dst_ip = int(config['dst_ip'])
        self.id = config['terminal_id']
        self.gen = Converter.GeneratedSignalInput(self.signal['samplerate'], self.signal['duration'],
                                                  self.signal['frequency'])
        self.adc = Converter.ADC(self.rate)
        self.ip = Protocol.IPProtocol(self.src_ip, self.dst_ip)
        self.frame = Protocol.FrameProtocol(self.id, self.pd_form.get(config['dst_ip']))
        self.code = Code.ConvCode()
        self.mod = Module.PSK("QPSK")
        self.input_data: np.ndarray  # input A sig
        self.adc_data: np.ndarray  # ADC data
        self.t: np.ndarray
        self.ip_data: any  # ip
        self.f_data: bytes  # frame data
        self.c_data: np.ndarray = np.array([])  # code data
        self.output_data: np.ndarray = np.array([])  # output module symbols

    def run(self, *args, **kwargs):
        self.input_data = self.gen.signal
        self.t, self.adc_data = self.adc.convert(self.gen)
        self.ip_data = self.ip.pack(self.adc_data)
        f = []
        out = []
        for i in self.ip_data:
            f_data = self.frame.encapsulate(i)
            f.append(f_data)
            c_data = self.code.encode(f_data)
            self.c_data = np.concatenate([self.c_data, c_data])
            m_data = self.mod.modulate(c_data)
            out.append(m_data)
            self.output_data = np.concatenate([self.output_data, m_data])
        self.f_data = b''.join(f)
        return out

    def monitor(self):
        ipdata = []
        for i in self.ip_data:
            ip = self.ip.unpack(i)
            ipdata.append(ip)
        self.ip_data = ipdata
        dic = {
            "input_data": self.input_data,
            "adc_data": self.adc_data,
            "ip_data": self.ip_data,
            "f_data": self.f_data,
            "c_data": self.c_data,
            "output_data": self.output_data
        }
        return dic


class VoiceTerminal_IN(BaseComponent):
    def __init__(self, config: dict):
        self.rate = config['adc_samplerate']
        self.file_path = config['wav_path']
        self.pd_form = config['ip_id_mapping']
        self.src_ip = int(config['src_ip'])
        self.dst_ip = int(config['dst_ip'])
        self.id = config['terminal_id']
        self.gen = Converter.WavFileInput(self.file_path)
        self.adc = Converter.ADC(self.rate)
        self.ip = Protocol.IPProtocol(self.src_ip, self.dst_ip)
        self.frame = Protocol.FrameProtocol(self.id, self.pd_form.get(config['dst_ip']))
        self.code = Code.ConvCode()
        self.mod = Module.PSK("QPSK")
        self.input_data: np.ndarray  # input A sig
        self.adc_data: np.ndarray  # ADC data
        self.t: np.ndarray
        self.ip_data = []  # ip
        self.f_data: bytes  # frame data
        self.c_data: np.ndarray = np.array([])  # code data
        self.output_data: np.ndarray = np.array([])  # output module symbols

    def run(self):
        self.input_data = self.gen.signal
        self.t, self.adc_data = self.adc.convert(self.gen)
        self.ip_data = self.ip.pack(self.adc_data)
        f = []
        out = []
        for i in self.ip_data:
            f_data = self.frame.encapsulate(i)
            f.append(f_data)
            c_data = self.code.encode(f_data)
            self.c_data = np.concatenate([self.c_data, c_data])
            m_data = self.mod.modulate(c_data)
            out.append(m_data)
            self.output_data = np.concatenate([self.output_data, m_data])
        self.f_data = b''.join(f)
        return out

    def monitor(self):
        ipdata = []
        for i in self.ip_data:
            ip = self.ip.unpack(i)
            ipdata.append(ip)
        self.ip_data = ipdata
        dic = {
            "input_data": self.input_data,
            "adc_data": self.adc_data,
            "ip_data": self.ip_data,
            "f_data": self.f_data,
            "c_data": self.c_data,
            "output_data": self.output_data
        }
        return dic


class SignalTerminal_OUT(BaseComponent):
    def __init__(self, config: dict):
        self.rate = config['adc_samplerate']
        self.ip_addr = int(config['terminal_ip'])
        self.id = config['terminal_id']
        self.dac = Converter.DAC(self.rate)
        self.ip = Protocol.IPProtocol(self.ip_addr, 0x00000000)
        self.frame = Protocol.FrameProtocol(self.id, b'\00' * 6)
        self.code = Code.ConvCode()
        self.mod = Module.PSK("QPSK")
        self.input_data: np.ndarray  # input m sig
        self.adc_data: np.ndarray  # ADC data
        self.t: np.ndarray
        self.ip_data = []  # ip
        self.f_data: bytes  # frame data
        self.c_data: np.ndarray = np.array([])  # code data
        self.output_data: np.ndarray  # output a sig

    def run(self, *args, **kwargs):
        self.input_data = args[0] if args else kwargs.get('input_data')
        print(f"self.input_data{self.input_data}")
        if self.input_data is None:
            raise ValueError("No input_data.")
        f = []
        for data in self.input_data:
            c_data = self.mod.demodulate(data)
            self.c_data = np.concatenate([self.c_data, c_data])
            f_data = self.code.decode(c_data)
            f.append(f_data)
            f_content = self.frame.decapsulate(f_data)
            ip_data = self.ip.unpack(f_content["payload"])
            self.ip_data.append(ip_data)
        print(f"self.ip_data{self.ip_data}")
        self.adc_data = self.ip.reassemble(self.ip_data)
        print(f"self.adc_data{self.adc_data}")
        t, self.output_data = self.dac.convert(self.adc_data, 4000)
        self.f_data = b''.join(f)

        return self.output_data

    def monitor(self):
        input_data = self.input_data
        self.input_data = np.array([])
        for data in input_data:
            self.input_data = np.concatenate([self.input_data, data])
        dic = {
            "input_data": self.input_data,
            "adc_data": self.adc_data,
            "ip_data": self.ip_data,
            "f_data": self.f_data,
            "c_data": self.c_data,
            "output_data": self.output_data
        }
        return dic


class VoiceTerminal_OUT(BaseComponent):
    def __init__(self, config: dict):
        self.rate = config['adc_samplerate']
        self.ip_addr = int(config['terminal_ip'])
        self.id = config['terminal_id']
        self.file_path = config['file_path'] + "/output_1.wav"
        self.dac = Converter.DAC(self.rate)
        self.ip = Protocol.IPProtocol(self.ip_addr, 0x00000000)
        self.frame = Protocol.FrameProtocol(self.id, b'\00' * 6)
        self.code = Code.ConvCode()
        self.mod = Module.PSK("QPSK")
        self.input_data: np.ndarray  # input m sig
        self.adc_data: np.ndarray  # ADC data
        self.t: np.ndarray
        self.ip_data = []  # ip
        self.f_data: bytes  # frame data
        self.c_data: np.ndarray = np.array([])  # code data
        self.output_data: np.ndarray  # output a sig

    def run(self, *args, **kwargs):
        self.input_data = args[0] if args else kwargs.get('input_data')
        if self.input_data is None:
            raise ValueError("No input_data.")
        f = []
        for data in self.input_data:
            c_data = self.mod.demodulate(data)
            self.c_data = np.concatenate([self.c_data, c_data])
            f_data = self.code.decode(c_data)
            f.append(f_data)
            f_content = self.frame.decapsulate(f_data)
            ip_data = self.ip.unpack(f_content["payload"])
            self.ip_data.append(ip_data)
        self.adc_data = self.ip.reassemble(self.ip_data)
        t, self.output_data = self.dac.convert(self.adc_data, 4000)
        self.f_data = b''.join(f)
        wavfile.write(self.file_path, 4000, self.output_data)

    def monitor(self):
        input_data = self.input_data
        self.input_data = np.array([])
        for data in input_data:
            self.input_data = np.concatenate([self.input_data, data])
        dic = {
            "input_data": self.input_data,
            "adc_data": self.adc_data,
            "ip_data": self.ip_data,
            "f_data": self.f_data,
            "c_data": self.c_data,
            "output_data": self.output_data
        }
        return dic
