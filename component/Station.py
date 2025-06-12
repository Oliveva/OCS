from component.base import BaseComponent
from core import Code, Inter, Module, Protocol
import numpy as np


class BaseStation(BaseComponent):
    def __init__(self, config: dict):
        self.ip_addr = int(config["station_ip"])
        self.id = config['station_id']

        self.ip = Protocol.IPProtocol(self.ip_addr, 0x00000001)
        self.frame = Protocol.FrameProtocol(self.id, b'\x00\x01\x00\x02\x00\x03')
        self.code = Code.ConvCode()
        self.mod = Module.PSK("QPSK")
        self.route = Inter.Route()
        self.pd_form = config['ip_id_mapping']
        for dst_ip, next_ip in config['routing'].items():
            self.route.add_route(int(dst_ip), int(next_ip))
        self.input_data = []  # input module symbols
        self.c_data_d = np.array([])  # demodule -> code data
        self.f_data_d: bytes  # decode -> frame data
        self.f_content = []
        self.ip_data_d = []  # deframe -> ip data
        self.f_data_e: bytes  # enframe -> frame data
        self.c_data_e = np.array([])  # encode -> code data
        self.output_data: np.ndarray = np.array([])  # enmodule -> output module symbols

    def run(self, *args, **kwargs):
        self.input_data = args[0] if args else kwargs.get("input_data")
        if self.input_data is None:
            raise ValueError("No input_data")
        c_data_d = self.mod.demodulate(self.input_data[0])
        f_data_d = self.code.decode(c_data_d)
        ip_data_d = self.frame.decapsulate(f_data_d)["payload"]
        next_ip = self.route.route_data(ip_data_d)
        self.frame = Protocol.FrameProtocol(self.id, self.pd_form.get(next_ip))
        f_d = []
        f_e = []
        out = []
        for data in self.input_data:
            c_data_d = self.mod.demodulate(data)
            self.c_data_d = np.concatenate([self.c_data_d, c_data_d])
            f_data_d = self.code.decode(c_data_d)
            f_d.append(f_data_d)
            f_content = self.frame.decapsulate(f_data_d)
            ip_data_d = f_content["payload"]
            self.f_content.append(f_content)
            f_data_e = self.frame.encapsulate(ip_data_d)
            f_e.append(f_data_e)
            c_data_e = self.code.encode(f_data_e)
            self.c_data_e = np.concatenate([self.c_data_e, c_data_e])
            output_data = self.mod.modulate(c_data_e)
            self.output_data = np.concatenate([self.output_data, output_data])
            out.append(output_data)
        self.f_data_d = b''.join(f_d)
        self.f_data_e = b''.join(f_e)
        return out

    def monitor(self):
        input_data = self.input_data
        self.input_data = np.array([])
        for data in input_data:
            self.input_data = np.concatenate([self.input_data, data])
        dic = {
            "input_data": self.input_data,
            "demod_data": self.c_data_d,
            "decode_data": self.f_data_d,
            "f_content": self.f_content,
            "cap_data": self.f_data_e,
            "code_data": self.c_data_e,
            "output_data": self.output_data
        }
        return dic
