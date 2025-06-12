import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import struct
from ipaddress import ip_address
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from component.Terminal import *
import json
from collections.abc import Iterable


class VoiceTerminalUI_IN(tk.Toplevel):
    def __init__(self, parent, terminal_id, callback, config_data=None, show_data=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.id = terminal_id
        self.title(f"话音终端-{terminal_id}")
        self.geometry("900x700")
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

        adc_frame = ttk.LabelFrame(self.config_frame, text="采样率配置", padding=10)
        adc_frame.pack(fill=tk.X, pady=5)
        self.adc_combobox = ttk.Combobox(adc_frame, values=["8000", "16000", "32000", "44100"], width=15)
        self.adc_combobox.set(8000 if not self.config_dict else self.config_dict.get('adc_samplerate', 8000))
        ttk.Label(adc_frame, text="采样率(Hz):").pack(side=tk.LEFT, padx=5)
        self.adc_combobox.pack(side=tk.LEFT, padx=5)

        wav_frame = ttk.LabelFrame(self.config_frame, text="音频文件配置", padding=10)
        wav_frame.pack(fill=tk.X, pady=5)
        self.wav_path_entry = ttk.Entry(wav_frame, width=50)
        self.wav_path_entry.pack(side=tk.LEFT, padx=5)
        self.wav_path_entry.delete(0, tk.END)
        self.wav_path_entry.insert(tk.END, "" if not self.config_dict else self.config_dict.get('wav_path', ""))
        ttk.Button(wav_frame, text="选择文件", command=self.choose_wav_file).pack(side=tk.LEFT, padx=5)

        ip_frame = ttk.LabelFrame(self.config_frame, text="IP地址配置", padding=10)
        ip_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ip_frame, text="源IP:").grid(row=0, column=0, padx=5, pady=3)
        self.src_ip_entry = ttk.Entry(ip_frame, width=20)
        self.src_ip_entry.grid(row=0, column=1, padx=5, pady=3)
        self.src_ip_entry.insert(tk.END,
                                 "192.168.1.1" if not self.config_dict
                                 else self.config_dict.get('src_ip_text', "192.168.1.1")
                                 )

        ttk.Label(ip_frame, text="目的IP:").grid(row=1, column=0, padx=5, pady=3)
        self.dst_ip_entry = ttk.Entry(ip_frame, width=20)
        self.dst_ip_entry.grid(row=1, column=1, padx=5, pady=3)
        self.dst_ip_entry.insert(tk.END,
                                 "1.0.0.3" if not self.config_dict else self.config_dict.get('dst_ip_text', "1.0.0.3")
                                 )

        id_frame = ttk.LabelFrame(self.config_frame, text="终端ID配置", padding=10)
        id_frame.pack(fill=tk.X, pady=5)
        ttk.Label(id_frame, text="当前终端ID(格式 1:2:3):").pack(side=tk.LEFT, padx=5)
        self.terminal_id_entry = ttk.Entry(id_frame, width=20)
        self.terminal_id_entry.pack(side=tk.LEFT, padx=5)
        self.terminal_id_entry.insert(tk.END,
                                      "1:2:3" if not self.config_dict else self.config_dict.get('terminal_id_text',
                                                                                                "1:2:3")
                                      )

        mapping_frame = ttk.LabelFrame(self.config_frame, text="IP与ID对应表", padding=10)
        mapping_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.mapping_text = scrolledtext.ScrolledText(mapping_frame, wrap=tk.WORD, height=6)
        self.mapping_text.pack(fill=tk.BOTH, expand=True)
        if not self.config_dict:
            self.mapping_text.insert(tk.END, "输入格式（每行一个条目）:\n192.168.1.2 4:5:6\n10.2.3.0 7:8:9\n")
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
            "输入信号", "量化信号", "ip数据报",
            "数据帧", "编码", "调制信号"
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
            "输入信号": "input_data",
            "量化信号": "adc_data",
            "调制信号": "output_data",
            "ip数据报": "ip_data",
            "数据帧": "f_data",
            "编码": "c_data"
        }
        if selected_type in ["输入信号", "量化信号", "调制信号"]:
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

    def choose_wav_file(self):

        file_path = filedialog.askopenfilename(filetypes=[("WAV文件", "*.wav")])
        if file_path:
            self.wav_path_entry.delete(0, tk.END)
            self.wav_path_entry.insert(tk.END, file_path)

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

        adc_rate = self.adc_combobox.get()
        if not adc_rate.isdigit():
            messagebox.showerror("错误", "采样率必须为数字")
            return

        src_ip_text = self.src_ip_entry.get()
        dst_ip_text = self.dst_ip_entry.get()
        terminal_id_text = self.terminal_id_entry.get()

        src_ip = self.ip_to_int(self.src_ip_entry.get())
        dst_ip = self.ip_to_int(self.dst_ip_entry.get())
        if not src_ip or not dst_ip:
            messagebox.showerror("错误", "IP地址格式不正确")
            return

        terminal_id = self.id_to_bytes(self.terminal_id_entry.get())
        if not terminal_id:
            messagebox.showerror("错误", "ID格式不正确（需为16进制数:分隔）")
            return

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
            "adc_samplerate": int(adc_rate),
            "wav_path": self.wav_path_entry.get(),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "terminal_id": terminal_id,
            "src_ip_text": src_ip_text,
            "dst_ip_text": dst_ip_text,
            "terminal_id_text": terminal_id_text,
            "ip_id_mapping": mapping_dict,
            "mapping_text": mapping_ori
        }

        if show:
            messagebox.showinfo("提示", "配置保存成功！")
        self.callback(self.config_dict, self.id)


class SignalTerminalUI_IN(tk.Toplevel):
    def __init__(self, parent, terminal_id, callback, config_data=None, show_data=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.id = terminal_id
        self.title(f"信号终端-{terminal_id}")
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

        adc_frame = ttk.LabelFrame(self.config_frame, text="采样率配置", padding=10)
        adc_frame.pack(fill=tk.X, pady=5)
        self.adc_combobox = ttk.Combobox(adc_frame,
                                         values=["8000", "16000", "32000", "44100", "48000"],
                                         width=15)
        self.adc_combobox.set(4000 if not self.config_dict else self.config_dict.get('adc_samplerate', 4000))
        ttk.Label(adc_frame, text="采样率(Hz):").pack(side=tk.LEFT, padx=5)
        self.adc_combobox.pack(side=tk.LEFT, padx=5)

        signal_frame = ttk.LabelFrame(self.config_frame, text="信号生成参数", padding=10)
        signal_frame.pack(fill=tk.X, pady=5)

        ttk.Label(signal_frame, text="信号采样率(Hz):").grid(row=0, column=0, padx=5, pady=3)
        self.samplerate_entry = ttk.Entry(signal_frame, width=15)
        self.samplerate_entry.grid(row=0, column=1, padx=5, pady=3)
        self.samplerate_entry.insert(tk.END,
                                     "4000" if not self.config_dict
                                     else self.config_dict.get('signal_params', {"samplerate": 4000}).get('samplerate')
                                     )

        ttk.Label(signal_frame, text="持续时间(s):").grid(row=0, column=2, padx=5, pady=3)
        self.duration_entry = ttk.Entry(signal_frame, width=15)
        self.duration_entry.grid(row=0, column=3, padx=5, pady=3)
        self.duration_entry.insert(tk.END,
                                   "1" if not self.config_dict
                                   else self.config_dict.get('signal_params', {"duration": 1}).get('duration')
                                   )

        ttk.Label(signal_frame, text="信号频率(Hz):").grid(row=1, column=0, padx=5, pady=3)
        self.freq_entry = ttk.Entry(signal_frame, width=15)
        self.freq_entry.grid(row=1, column=1, padx=5, pady=3)
        self.freq_entry.insert(tk.END,
                               "100" if not self.config_dict
                               else self.config_dict.get('signal_params', {"frequency": 100}).get('frequency')
                               )

        ip_frame = ttk.LabelFrame(self.config_frame, text="IP地址配置", padding=10)
        ip_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ip_frame, text="源IP:").grid(row=0, column=0, padx=5, pady=3)
        self.src_ip_entry = ttk.Entry(ip_frame, width=20)
        self.src_ip_entry.grid(row=0, column=1, padx=5, pady=3)
        self.src_ip_entry.insert(tk.END,
                                 "192.168.1.1" if not self.config_dict else self.config_dict.get('src_ip_text',
                                                                                                 "192.168.1.1")
                                 )

        ttk.Label(ip_frame, text="目的IP:").grid(row=1, column=0, padx=5, pady=3)
        self.dst_ip_entry = ttk.Entry(ip_frame, width=20)
        self.dst_ip_entry.grid(row=1, column=1, padx=5, pady=3)
        self.dst_ip_entry.insert(tk.END,
                                 "1.0.0.3" if not self.config_dict else self.config_dict.get('dst_ip_text', "1.0.0.3")
                                 )

        id_frame = ttk.LabelFrame(self.config_frame, text="终端ID配置", padding=10)
        id_frame.pack(fill=tk.X, pady=5)
        ttk.Label(id_frame, text="当前终端ID(格式 1:2:3):").pack(side=tk.LEFT, padx=5)
        self.terminal_id_entry = ttk.Entry(id_frame, width=20)
        self.terminal_id_entry.pack(side=tk.LEFT, padx=5)
        self.terminal_id_entry.insert(tk.END,
                                      "1:2:3" if not self.config_dict else self.config_dict.get('terminal_id_text',
                                                                                                "1:2:3")
                                      )

        mapping_frame = ttk.LabelFrame(self.config_frame, text="IP与ID对应表", padding=10)
        mapping_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.mapping_text = scrolledtext.ScrolledText(mapping_frame, wrap=tk.WORD, height=6)
        self.mapping_text.pack(fill=tk.BOTH, expand=True)
        if not self.config_dict:
            self.mapping_text.insert(tk.END, "输入格式（每行一个条目）:\n192.168.1.2 4:5:6\n1.0.0.3 7:8:9\n")
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
            "输入信号", "量化信号", "ip数据报",
            "数据帧", "编码", "调制信号"
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
            "输入信号": "input_data",
            "量化信号": "adc_data",
            "调制信号": "output_data",
            "ip数据报": "ip_data",
            "数据帧": "f_data",
            "编码": "c_data"
        }
        if selected_type in ["输入信号", "量化信号", "调制信号"]:
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

        adc_rate = self.adc_combobox.get()
        if not adc_rate.isdigit():
            messagebox.showerror("错误", "采样率必须为数字")
            return

        src_ip_text = self.src_ip_entry.get()
        dst_ip_text = self.dst_ip_entry.get()
        terminal_id_text = self.terminal_id_entry.get()

        try:
            samplerate = int(self.samplerate_entry.get())
            duration = float(self.duration_entry.get())
            freq = int(self.freq_entry.get())
            if samplerate <= 0 or duration <= 0 or freq <= 0:
                raise ValueError
        except:
            messagebox.showerror("错误", "信号参数必须为正数值")
            return

        src_ip = self.ip_to_int(self.src_ip_entry.get())
        dst_ip = self.ip_to_int(self.dst_ip_entry.get())
        if not src_ip or not dst_ip:
            messagebox.showerror("错误", "IP地址格式不正确")
            return

        terminal_id = self.id_to_bytes(self.terminal_id_entry.get())
        if not terminal_id:
            messagebox.showerror("错误", "ID格式不正确（需为16进制数:分隔）")
            return

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
            "adc_samplerate": adc_rate,
            "signal_params": {
                "samplerate": samplerate,
                "duration": duration,
                "frequency": freq
            },
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "terminal_id": terminal_id,
            "src_ip_text": src_ip_text,
            "dst_ip_text": dst_ip_text,
            "terminal_id_text": terminal_id_text,
            "ip_id_mapping": mapping_dict,
            "mapping_text": mapping_ori
        }

        if show:
            messagebox.showinfo("提示", "配置保存成功！")
        self.callback(self.config_dict, self.id)


class VoiceTerminalUI_OUT(tk.Toplevel):
    def __init__(self, parent, terminal_id, callback, config_data=None, show_data=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.id = terminal_id
        self.title(f"话音终端-{terminal_id}")
        self.geometry("900x700")
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

        adc_frame = ttk.LabelFrame(self.config_frame, text="采样率配置", padding=10)
        adc_frame.pack(fill=tk.X, pady=5)
        self.adc_combobox = ttk.Combobox(adc_frame, values=["8000", "16000", "32000", "44100"], width=15)
        self.adc_combobox.set(8000 if not self.config_dict else self.config_dict.get('adc_samplerate', 8000))
        ttk.Label(adc_frame, text="采样率(Hz):").pack(side=tk.LEFT, padx=5)
        self.adc_combobox.pack(side=tk.LEFT, padx=5)

        wav_frame = ttk.LabelFrame(self.config_frame, text="音频文件输出位置", padding=10)
        wav_frame.pack(fill=tk.X, pady=5)
        self.wav_path_entry = ttk.Entry(wav_frame, width=50)
        self.wav_path_entry.pack(side=tk.LEFT, padx=5)
        self.wav_path_entry.delete(0, tk.END)
        self.wav_path_entry.insert(tk.END, "" if not self.config_dict else self.config_dict.get('file_path', ""))
        ttk.Button(wav_frame, text="选择文件", command=self.choose_wav_file).pack(side=tk.LEFT, padx=5)

        ip_frame = ttk.LabelFrame(self.config_frame, text="IP配置", padding=10)
        ip_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ip_frame, text="当前终端IP(格式 1.0.0.1):").pack(side=tk.LEFT, padx=5)
        self.terminal_ip_entry = ttk.Entry(ip_frame, width=20)
        self.terminal_ip_entry.pack(side=tk.LEFT, padx=5)
        self.terminal_ip_entry.insert(tk.END,
                                      "1.0.0.1" if not self.config_dict
                                      else self.config_dict.get('terminal_ip_text', "1.0.0.1")
                                      )

        id_frame = ttk.LabelFrame(self.config_frame, text="终端ID配置", padding=10)
        id_frame.pack(fill=tk.X, pady=5)
        ttk.Label(id_frame, text="当前终端ID(格式 1:2:3):").pack(side=tk.LEFT, padx=5)
        self.terminal_id_entry = ttk.Entry(id_frame, width=20)
        self.terminal_id_entry.pack(side=tk.LEFT, padx=5)
        self.terminal_id_entry.insert(tk.END,
                                      "1:2:3" if not self.config_dict
                                      else self.config_dict.get('terminal_id_text', "1:2:3")
                                      )
        self.save_config(False)

        ttk.Button(self.config_frame, text="保存配置", command=self.save_config).pack(pady=5)

    def _create_data_interface(self):

        self.data_frame = ttk.Frame(self.main_container)

        control_frame = ttk.Frame(self.data_frame, padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        ttk.Label(control_frame, text="选择数据类型:").pack(side=tk.LEFT, padx=5)
        self.data_combobox = ttk.Combobox(control_frame, values=[
            "输入信号", "编码", "数据帧",
            "ip数据报", "量化信号", "输出信号"
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
            "输入信号": "input_data",
            "量化信号": "adc_data",
            "输出信号": "output_data",
            "ip数据报": "ip_data",
            "数据帧": "f_data",
            "编码": "c_data"
        }
        if selected_type in ["输入信号", "量化信号", "输出信号"]:
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

    def choose_wav_file(self):

        folder_path = filedialog.askdirectory(
            title="请选择文件夹",
            initialdir="/",
            mustexist=False
        )
        if folder_path:
            self.wav_path_entry.delete(0, tk.END)
            self.wav_path_entry.insert(tk.END, folder_path)

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

        adc_rate = self.adc_combobox.get()
        if not adc_rate.isdigit():
            messagebox.showerror("错误", "采样率必须为数字")
            return

        terminal_ip_text = self.terminal_ip_entry.get()
        terminal_id_text = self.terminal_id_entry.get()

        terminal_ip = self.ip_to_int(self.terminal_ip_entry.get())
        if not terminal_ip:
            messagebox.showerror("错误", "IP格式不正确（需为10进制数.分隔）")
            return

        terminal_id = self.id_to_bytes(self.terminal_id_entry.get())
        if not terminal_id:
            messagebox.showerror("错误", "ID格式不正确（需为16进制数:分隔）")
            return

        self.config_dict = {
            "adc_samplerate": int(adc_rate),
            "file_path": self.wav_path_entry.get(),
            "terminal_ip": terminal_ip,
            "terminal_id": terminal_id,
            "terminal_ip_text": terminal_ip_text,
            "terminal_id_text": terminal_id_text
        }

        if show:
            messagebox.showinfo("提示", "配置保存成功！")
        self.callback(self.config_dict, self.id)


class SignalTerminalUI_OUT(tk.Toplevel):
    def __init__(self, parent, terminal_id, callback, config_data=None, show_data=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.id = terminal_id
        self.title(f"信号终端-{terminal_id}")
        self.geometry("900x700")
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

        adc_frame = ttk.LabelFrame(self.config_frame, text="采样率配置", padding=10)
        adc_frame.pack(fill=tk.X, pady=5)
        self.adc_combobox = ttk.Combobox(adc_frame, values=["8000", "16000", "32000", "44100"], width=15)
        self.adc_combobox.set(4000 if not self.config_dict else self.config_dict.get('adc_samplerate', 4000))
        ttk.Label(adc_frame, text="采样率(Hz):").pack(side=tk.LEFT, padx=5)
        self.adc_combobox.pack(side=tk.LEFT, padx=5)

        ip_frame = ttk.LabelFrame(self.config_frame, text="IP配置", padding=10)
        ip_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ip_frame, text="当前终端IP(格式 1.0.0.1):").pack(side=tk.LEFT, padx=5)
        self.terminal_ip_entry = ttk.Entry(ip_frame, width=20)
        self.terminal_ip_entry.pack(side=tk.LEFT, padx=5)
        self.terminal_ip_entry.insert(tk.END,
                                      "1.0.0.3" if not self.config_dict
                                      else self.config_dict.get('terminal_ip_text', "1.0.0.3")
                                      )

        id_frame = ttk.LabelFrame(self.config_frame, text="终端ID配置", padding=10)
        id_frame.pack(fill=tk.X, pady=5)
        ttk.Label(id_frame, text="当前终端ID(格式 1:2:3):").pack(side=tk.LEFT, padx=5)
        self.terminal_id_entry = ttk.Entry(id_frame, width=20)
        self.terminal_id_entry.pack(side=tk.LEFT, padx=5)
        self.terminal_id_entry.insert(tk.END,
                                      "3:2:1" if not self.config_dict
                                      else self.config_dict.get('terminal_id_text', "3:2:1")
                                      )

        self.save_config(False)

        ttk.Button(self.config_frame, text="保存配置", command=self.save_config).pack(pady=5)

    def _create_data_interface(self):

        self.data_frame = ttk.Frame(self.main_container)

        control_frame = ttk.Frame(self.data_frame, padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        ttk.Label(control_frame, text="选择数据类型:").pack(side=tk.LEFT, padx=5)
        self.data_combobox = ttk.Combobox(control_frame, values=[
            "输入信号", "编码", "数据帧",
            "ip数据报", "量化信号", "输出信号"
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
            "输入信号": "input_data",
            "量化信号": "adc_data",
            "输出信号": "output_data",
            "ip数据报": "ip_data",
            "数据帧": "f_data",
            "编码": "c_data"
        }
        if selected_type in ["输入信号", "量化信号", "输出信号"]:
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

        adc_rate = self.adc_combobox.get()
        if not adc_rate.isdigit():
            messagebox.showerror("错误", "采样率必须为数字")
            return

        terminal_ip_text = self.terminal_ip_entry.get()
        terminal_id_text = self.terminal_id_entry.get()

        terminal_ip = self.ip_to_int(self.terminal_ip_entry.get())
        if not terminal_ip:
            messagebox.showerror("错误", "IP格式不正确（需为10进制数.分隔）")
            return

        terminal_id = self.id_to_bytes(self.terminal_id_entry.get())
        if not terminal_id:
            messagebox.showerror("错误", "ID格式不正确（需为16进制数:分隔）")
            return

        self.config_dict = {
            "adc_samplerate": int(adc_rate),
            "terminal_ip": terminal_ip,
            "terminal_id": terminal_id,
            "terminal_ip_text": terminal_ip_text,
            "terminal_id_text": terminal_id_text
        }

        if show:
            messagebox.showinfo("提示", "配置保存成功！")
        self.callback(self.config_dict, self.id)


