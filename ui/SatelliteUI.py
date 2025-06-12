import ipaddress
import tkinter as tk
from collections.abc import Iterable
from tkinter import ttk, scrolledtext, messagebox
import struct
from component.Satellite import Satellite
from ipaddress import ip_network, ip_address
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import json


class SatelliteUI(tk.Toplevel):
    def __init__(self, parent, sat_id, callback, config_data=None, show_data=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.id = sat_id
        self.title(f"同步卫星-{sat_id}")
        self.geometry("900x750")
        self.resizable(True, True)
        self.config_dict = config_data or {}
        self.logic = None
        self.data_dict = show_data
        self.callback = callback

        self.top_frame = tk.Frame(self, padx=15, pady=8)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_config = ttk.Button(self.top_frame, text="配置界面", command=self.show_config)
        self.btn_config.pack(side=tk.LEFT, padx=10)
        self.btn_data = ttk.Button(self.top_frame, text="数据显示", command=self.show_data)
        self.btn_data.pack(side=tk.LEFT, padx=10)

        self.main_container = ttk.Frame(self)
        self.main_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._create_config_interface()
        self._create_data_interface()
        self.current_interface = self.config_frame
        self.current_interface.pack(fill=tk.BOTH, expand=True)

    def _create_config_interface(self):

        self.config_frame = ttk.Frame(self.main_container)

        ip_frame = ttk.LabelFrame(self.config_frame, text="IP配置", padding=10)
        ip_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ip_frame, text="当前卫星IP(格式 1.0.0.1):").pack(side=tk.LEFT, padx=5)
        self.sat_ip_entry = ttk.Entry(ip_frame, width=20)
        self.sat_ip_entry.pack(side=tk.LEFT, padx=5)
        self.sat_ip_entry.insert(tk.END,
                                 "1.0.0.1" if not self.config_dict else self.config_dict.get('ip_text', "1.0.0.1")
                                 )

        id_frame = ttk.LabelFrame(self.config_frame, text="卫星ID配置", padding=10)
        id_frame.pack(fill=tk.X, pady=5)
        ttk.Label(id_frame, text="当前卫星ID(格式 1:2:0):").pack(side=tk.LEFT, padx=5)
        self.sat_id_entry = ttk.Entry(id_frame, width=20)
        self.sat_id_entry.pack(side=tk.LEFT, padx=5)
        self.sat_id_entry.insert(tk.END,
                                 "7:8:9" if not self.config_dict else self.config_dict.get('id_text', "7:8:9")
                                 )

        routing_frame = ttk.LabelFrame(self.config_frame, text="IP路由表", padding=10)
        routing_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.routing_text = scrolledtext.ScrolledText(routing_frame, wrap=tk.WORD, height=6)
        self.routing_text.pack(fill=tk.BOTH, expand=True)
        if not self.config_dict:
            self.routing_text.insert(tk.END, "输入格式（每行一个条目）:\n192.168.1.2 10.0.0.3\n1.0.0.3 1.0.0.1\n")
        else:
            for dst_ip, next_ip in self.config_dict['routing_text'].items():
                self.routing_text.insert(tk.END, f"{dst_ip} {next_ip}\n")

        mapping_frame = ttk.LabelFrame(self.config_frame, text="IP与ID对应表", padding=10)
        mapping_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.mapping_text = scrolledtext.ScrolledText(mapping_frame, wrap=tk.WORD, height=6)
        self.mapping_text.pack(fill=tk.BOTH, expand=True)
        if not self.config_dict:
            self.mapping_text.insert(tk.END, "输入格式（每行一个条目）:\n192.168.1.2 4:5:6\n1.0.0.1 3:2:1\n")
        else:
            for ip, id in self.config_dict['mapping_text'].items():
                self.mapping_text.insert(tk.END, f"{ip} {id}\n")

        self.save_config(False)

        ttk.Button(self.config_frame, text="保存配置", command=self.save_config).pack(pady=5)

    def _create_data_interface(self):

        self.data_frame = ttk.Frame(self.main_container)

        control_frame = ttk.Frame(self.data_frame, padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        ttk.Label(control_frame, text="选择数据类型:").pack(side=tk.LEFT, padx=5)
        self.data_combobox = ttk.Combobox(control_frame, values=[
            "输入信号（调制）", "解调数据（编码）", "解码数据（帧）",
            "帧解析数据", "数据帧", "编码", "输出信号（调制）"
        ], width=20)
        self.data_combobox.current(0)
        self.data_combobox.pack(side=tk.LEFT, padx=5)
        self.data_combobox.bind("<<ComboboxSelected>>", self.on_data_type_selected)

        self.display_area = ttk.Frame(self.data_frame)
        self.display_area.pack(fill=tk.BOTH, expand=True, pady=5)

        self.text_display = scrolledtext.ScrolledText(
            self.display_area,
            wrap=tk.WORD,
            height=20,
            state=tk.DISABLED
        )

        self.figure = Figure(figsize=(8, 4), dpi=100, facecolor="#f5f5f5")
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_area)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.display_area)

        self.current_display = "text"
        self.text_display.pack(fill=tk.BOTH, expand=True)

    def on_data_type_selected(self, event):

        selected_type = self.data_combobox.get()
        self.hide_current_display()
        type_dic = {
            "输入信号（调制）": "input_data",
            "解调数据（编码）": "demod_data",
            "解码数据（帧）": "decode_data",
            "帧解析数据": "f_content",
            "数据帧": "cap_data",
            "编码": "code_data",
            "输出信号（调制）": "output_data"
        }
        if selected_type in ["输入信号（调制）", "输出信号（调制）"]:
            self.show_plot_display(type_dic[selected_type])
        else:
            self.show_text_display(type_dic[selected_type])

    def hide_current_display(self):

        if self.current_display == "text":
            self.text_display.pack_forget()
        else:
            self.canvas.get_tk_widget().pack_forget()
            self.toolbar.pack_forget()

    def show_text_display(self, data_type):

        self.current_display = "text"
        self.update_text_display(data_type)
        self.text_display.pack(fill=tk.BOTH, expand=True)

    def show_plot_display(self, signal_type):

        self.current_display = "plot"
        self.plot_signal(signal_type)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_signal(self, signal_type):

        self.ax.clear()
        if not self.data_dict or signal_type not in self.data_dict:
            self.ax.text(0.5, 0.5, "no data, run logic please.", ha="center", va="center")
            self.canvas.draw()
            return

        data = self.data_dict[signal_type]
        x = np.arange(len(data))
        self.ax.plot(x, data)
        self.ax.set_xlim(0, 100)
        self.ax.set_title(f"{signal_type}")
        self.ax.set_xlabel("t (s)")
        self.ax.set_ylabel("A (V)")

        self.ax.grid(True, linestyle="--", alpha=0.7)
        self.canvas.draw()

    def format_data(self, data):

        if isinstance(data, bytearray):
            return self.format_data(bytes(data))


        elif isinstance(data, bytes):
            fdata = "b'" + ''.join([f'\\x{b:02x}' for b in data]) + "'"
            return fdata


        elif isinstance(data, np.ndarray):
            with np.printoptions(threshold=np.inf, linewidth=80):
                return str(data)


        elif isinstance(data, dict):
            cleaned_dict = {}
            for key, value in data.items():
                cleaned_dict[key] = self.format_data(value)
            return json.dumps(cleaned_dict, indent=4, ensure_ascii=False)


        elif isinstance(data, Iterable) and not isinstance(data, (str, bytes, bytearray)):

            return "[\n" + ",\n".join(self.format_data(item) for item in data) + "\n]"


        else:
            return str(data)

    def update_text_display(self, data_type):

        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete(1.0, tk.END)

        if self.data_dict:
            text = self.data_dict.get(data_type, "无相关数据")
            formatted_text = self.format_data(text)
            self.text_display.insert(tk.END, formatted_text)
        else:
            self.text_display.insert(tk.END, "无有效数据，请先运行逻辑")

        self.text_display.config(state=tk.DISABLED)

    def show_config(self):

        if self.current_interface != self.config_frame:
            self.current_interface.pack_forget()
            self.config_frame.pack(fill=tk.BOTH, expand=True)
            self.current_interface = self.config_frame

    def show_data(self):

        if self.current_interface != self.data_frame:
            self.current_interface.pack_forget()
            self.data_frame.pack(fill=tk.BOTH, expand=True)
            self.current_interface = self.data_frame

    def ip_to_int(self, ip_str):

        try:
            return int(ip_address(ip_str))
        except ValueError:
            return None

    def id_to_bytes(self, id_str):

        try:
            parts = list(map(lambda x: int(x, 16), id_str.split(':')))
            if len(parts) != 3:
                return None
            return struct.pack('>HHH', *parts)
        except:
            return None

    def save_config(self, show=True):
        ip_text = self.sat_ip_entry.get()
        id_text = self.sat_id_entry.get()

        sat_ip = self.ip_to_int(ip_text)
        if not sat_ip:
            messagebox.showerror("错误", "IP格式不正确（需为10进制数.分隔）")
            return

        sat_id = self.id_to_bytes(id_text)
        if not sat_id:
            messagebox.showerror("错误", "ID格式不正确（需为16进制数:分隔）")
            return

        routing_dict = {}
        routing_ori = {}
        for line in self.routing_text.get(1.0, tk.END).splitlines():
            line = line.strip()
            if line and ' ' in line:
                dst_ip_str, next_ip_str = line.split(' ', 1)
                dst_ip_int = self.ip_to_int(dst_ip_str)
                next_ip_int = self.ip_to_int(next_ip_str)
                if dst_ip_int and next_ip_int:
                    routing_dict[dst_ip_int] = next_ip_int
                if dst_ip_str and next_ip_str:
                    routing_ori[dst_ip_str] = next_ip_str

        mapping_dict = {}
        mapping_ori = {}
        for line in self.mapping_text.get(1.0, tk.END).splitlines():
            line = line.strip()
            if line and ' ' in line:
                ip_str, id_str = line.split(' ', 1)
                ip_int = self.ip_to_int(ip_str)
                id_bytes = self.id_to_bytes(id_str)
                if ip_int and id_bytes:
                    mapping_dict[ip_int] = id_bytes
                if ip_str and id_str:
                    mapping_ori[ip_str] = id_str

        self.config_dict = {
            "sat_ip": sat_ip,
            "sat_id": sat_id,
            "ip_text": ip_text,
            "id_text": id_text,
            "routing": routing_dict,
            "routing_text": routing_ori,
            "mapping_text": mapping_ori,
            "ip_id_mapping": mapping_dict
        }
        if show:
            messagebox.showinfo("提示", "配置保存成功！")
        self.callback(self.config_dict, self.id)


