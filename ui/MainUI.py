import threading
import tkinter as tk
from tkinter import ttk, Canvas, Frame, Label, Button, Toplevel

from ui.SatelliteUI import SatelliteUI
from ui.StationUI import BaseStationUI
from ui.TerminalUI import *
from component.Satellite import Satellite
from component.Station import BaseStation
from component.Terminal import *


class SatcomSimulationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("卫星通信仿真系统")
        self.root.geometry("1000x700")
        self.component_counter = 0

        self.current_dragging_type = None
        self.current_wire_start = None
        self.components = {}
        self.wires = []
        self.drag_data = None
        self.ui_data = {}
        self.logic_data = {}
        self.rel = {}
        self.is_simu = False

        self.top_settings_bar = Frame(root, height=40, bg="#f5f7fa")
        self.top_settings_bar.pack(fill=tk.X, padx=8, pady=4)
        Label(self.top_settings_bar, text="系统设置", font=("微软雅黑", 11, "bold"), fg="#2d3748").pack(side=tk.LEFT,
                                                                                                        padx=15)

        self.top_simulation_bar = Frame(root, height=40, bg="#edf2f7")
        self.top_simulation_bar.pack(fill=tk.X, padx=8, pady=4)
        Label(self.top_simulation_bar, text="仿真控制", font=("微软雅黑", 11, "bold"), fg="#2d3748").pack(side=tk.LEFT,
                                                                                                          padx=15)

        self.middle_frame = Frame(root)
        self.middle_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.left_component_area = Frame(self.middle_frame, width=350)
        self.left_component_area.pack(side=tk.LEFT, fill=tk.Y, padx=2)

        self.category_bar = Frame(self.left_component_area, width=90, bg="#f0f4f8")
        self.category_bar.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        self._create_category_buttons()

        self.library_bar = Frame(self.left_component_area, bg="#ffffff")
        self.library_bar.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.current_library_frame = None

        self.test_toolbar = Frame(self.middle_frame, width=180, bg="#f0f4f8")
        self.test_toolbar.pack(side=tk.RIGHT, fill=tk.Y, padx=2)
        self._create_test_tools()

        self.canvas_area = Canvas(
            self.middle_frame,
            bg="#e6f1f9",
            highlightthickness=1,
            highlightbackground="#a0b7c8"
        )
        self.canvas_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self._bind_canvas_events()

        self.bottom_debug_bar = Frame(root, height=50, bg="#edf2f7")
        self.bottom_debug_bar.pack(fill=tk.X, padx=8, pady=4)
        self.start_btn = Button(
            self.bottom_debug_bar,
            text="开始仿真 ▶",
            width=12,
            font=("微软雅黑", 10),
            bg="#48bb78",
            fg="white",
            command=self.start_simulation
        )
        self.start_btn.pack(side=tk.LEFT, padx=15)
        self.stop_btn = Button(
            self.bottom_debug_bar,
            text="结束仿真 ■",
            width=12,
            font=("微软雅黑", 10),
            bg="#f56565",
            fg="white",
            command=self.stop_simulation
        )
        self.stop_btn.pack(side=tk.LEFT, padx=15)
        self.stop_btn.config(state=tk.DISABLED)

        self.status_label = tk.Label(
            self.bottom_debug_bar,
            text="仿真状态：未开始",
            font=("微软雅黑", 10),
            bg="#edf2f7",
            fg="#2d3748"
        )
        self.status_label.pack(side=tk.LEFT, padx=20)

    def _create_category_buttons(self):
        """创建卫星/中继站/终端分类按钮"""
        categories = [
            {"name": "卫星", "icon": "📡", "type": "卫星"},
            {"name": "中继站", "icon": "🏢", "type": "中继站"},
            {"name": "终端", "icon": "📱", "type": "终端"}
        ]
        for idx, cat in enumerate(categories):
            btn = Button(
                self.category_bar,
                text=f"{cat['icon']}\n{cat['name']}",
                width=8,
                height=4,
                font=("微软雅黑", 10),
                bg="#e2e8f0",
                relief="flat",
                command=lambda t=cat['type']: self.show_library_components(t)
            )
            btn.grid(row=idx, column=0, pady=10, padx=5)

    def _create_test_tools(self):
        """创建测试工具按钮（信号显示仪）"""
        tools = [
            {"name": "信号显示仪", "icon": "📊", "type": "工具_信号显示仪"}
        ]
        for idx, tool in enumerate(tools):
            btn = Button(
                self.test_toolbar,
                text=f"{tool['icon']}\n{tool['name']}",
                width=14,
                height=4,
                font=("微软雅黑", 10),
                bg="#e2e8f0",
                relief="flat",
                command=lambda t=tool['type']: self.start_dragging(t)
            )
            btn.grid(row=idx, column=0, pady=15, padx=5)

    def show_library_components(self, category):
        """根据分类显示组件库中的卫星通信专用组件"""
        if self.current_library_frame:
            self.current_library_frame.destroy()
        self.current_library_frame = Frame(self.library_bar, bg="white")
        self.current_library_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        components = {
            "卫星": [("同步卫星", "🌍")],
            "中继站": [("地面基站", "📡")],
            "终端": [("话音终端(输入)", "📞"), ("信号终端(输入)", "💻"), ("话音终端(输出)", "📞"), ("信号终端(输出)", "💻")]
        }[category]

        for idx, (name, icon) in enumerate(components):
            btn = Button(
                self.current_library_frame,
                text=f"{icon}\n{name}",
                width=12,
                height=3,
                font=("微软雅黑", 10),
                bg="#f7fafc",
                relief="groove",
                command=lambda n=f"{category}_{name}": self.start_dragging(n)
            )
            btn.grid(row=idx, column=0, pady=8, padx=8)

    def start_dragging(self, component_type):
        """开始拖拽组件（进入预览模式）"""
        self.current_dragging_type = component_type
        self.canvas_area.config(cursor="tcross")

    def _bind_canvas_events(self):
        """绑定画布的所有交互事件"""
        self.canvas_area.bind("<Motion>", self.on_canvas_motion)
        self.canvas_area.bind("<Button-1>", self.on_canvas_click)
        self.canvas_area.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas_area.tag_bind("component", "<Double-Button-1>", self.on_component_double_click)
        self.canvas_area.tag_bind("component", "<ButtonPress-1>", self.on_component_press)
        self.canvas_area.tag_bind("component", "<B1-Motion>", self.on_component_drag)
        self.canvas_area.tag_bind("component", "<ButtonRelease-1>", self.on_component_release)

    def on_canvas_motion(self, event):
        """画布鼠标移动事件（处理组件/连线预览）"""

        if self.current_dragging_type:

            self.canvas_area.delete("preview")
            x, y = event.x, event.y
            color_map = {
                "卫星_同步卫星": "#63b3ed",
                "中继站_地面基站": "#90cdf4",
                "终端_话音终端": "#f6ad55",
                "终端_信号终端": "#f7768e",
                "工具_信号显示仪": "#9ae6b4"
            }
            preview_color = color_map.get(self.current_dragging_type, "#718096")

            self.canvas_area.create_rectangle(
                x - 40, y - 30, x + 40, y + 30,
                outline=preview_color,
                dash=(4, 4),
                tags="preview"
            )
            self.canvas_area.create_text(
                x, y,
                text=self.current_dragging_type.split("_")[1],
                fill=preview_color,
                font=("微软雅黑", 10, "bold"),
                tags="preview"
            )
        elif self.current_wire_start:

            self.canvas_area.delete("wire_preview")
            start_x, start_y = self.current_wire_start["x"], self.current_wire_start["y"]
            self.canvas_area.create_line(
                start_x, start_y, event.x, event.y,
                fill="#718096",
                width=2,
                dash=(4, 4),
                tags="wire_preview"
            )

    def on_canvas_click(self, event):

        """画布左键点击事件（放置组件/开始连线）"""
        if self.current_dragging_type:

            x, y = event.x, event.y
            self._place_component(x, y, self.current_dragging_type)
            self.current_dragging_type = None
            self.canvas_area.config(cursor="arrow")
        else:

            clicked_items = self.canvas_area.find_overlapping(event.x - 1, event.y - 1, event.x + 1, event.y + 1)
            component_tag = None
            for clicked_item in clicked_items:
                if clicked_item and "component" in self.canvas_area.gettags(clicked_item):
                    component_tag = next(
                        (tag for tag in self.canvas_area.gettags(clicked_item) if tag.startswith("component_")), None)
                    if component_tag:
                        break

            if component_tag is None:
                return
            component_info = self.components[component_tag]
            if not self.current_wire_start:

                self.current_wire_start = {
                    "tag": component_tag,
                    "x": component_info["x"],
                    "y": component_info["y"]
                }
                self.canvas_area.config(cursor="plus")
            else:

                if component_tag != self.current_wire_start["tag"]:
                    self._create_wire(self.current_wire_start, component_info)
                self.current_wire_start = None
                self.canvas_area.config(cursor="arrow")
                self.canvas_area.delete("wire_preview")

    def _place_component(self, x, y, component_type):
        """在画布指定位置放置卫星通信组件（记录唯一标签和所有元素）"""
        self.component_counter += 1
        component_tag = f"component_{self.component_counter}"

        style_map = {
            "卫星_同步卫星": {"fill": "#63b3ed", "text": "同步卫星", "icon": "🌍"},
            "中继站_地面基站": {"fill": "#90cdf4", "text": "地面基站", "icon": "📡"},
            "终端_话音终端(输入)": {"fill": "#f6ad55", "text": "话音终端(输入)", "icon": "📞"},
            "终端_信号终端(输入)": {"fill": "#f7768e", "text": "信号终端(输入)", "icon": "💻"},
            "终端_话音终端(输出)": {"fill": "#f6ad55", "text": "话音终端(输出)", "icon": "📞"},
            "终端_信号终端(输出)": {"fill": "#f7768e", "text": "信号终端(输出)", "icon": "💻"},
            "工具_信号显示仪": {"fill": "#9ae6b4", "text": "信号显示仪", "icon": "📊"}
        }
        style = style_map[component_type]

        rect_id = self.canvas_area.create_rectangle(
            x - 40, y - 30, x + 40, y + 30,
            fill=style["fill"],
            outline="#2d3748",
            width=1,
            tags=("component", component_tag)
        )

        icon_id = self.canvas_area.create_text(
            x, y - 10,
            text=style["icon"],
            font=("Segoe UI Emoji", 20),
            tags=("component", component_tag)
        )

        text_id = self.canvas_area.create_text(
            x, y + 18,
            text=style["text"],
            font=("微软雅黑", 10, "bold"),
            tags=("component", component_tag)
        )

        self.components[component_tag] = {
            "tag": component_tag,
            "x": x,
            "y": y,
            "type": component_type,
            "elements": [rect_id, icon_id, text_id],
            "config_class": self._get_config_class(component_type)
        }

    def _get_config_class(self, component_type):
        """根据组件类型获取对应的配置界面类"""
        config_map = {
            "卫星_同步卫星": SatelliteUI,
            "中继站_地面基站": BaseStationUI,
            "终端_话音终端(输入)": VoiceTerminalUI_IN,
            "终端_信号终端(输入)": SignalTerminalUI_IN,
            "终端_话音终端(输出)": VoiceTerminalUI_OUT,
            "终端_信号终端(输出)": SignalTerminalUI_OUT,
            "工具_信号显示仪": None
        }
        return config_map.get(component_type)

    def on_component_double_click(self, event):
        """双击组件打开对应的配置界面"""
        clicked_item = self.canvas_area.find_overlapping(event.x - 1, event.y - 1, event.x + 1, event.y + 1)[0]
        component_tag = next((tag for tag in self.canvas_area.gettags(clicked_item) if tag.startswith("component_")),
                             None)
        if not component_tag:
            return

        component_info = self.components[component_tag]
        config_class = component_info.get("config_class")
        default_data = {"result_data": None}
        result_data = self.logic_data.get(component_tag, default_data)["result_data"]

        if config_class:
            if component_tag not in self.ui_data:
                config_window = config_class(self.root, component_tag, self.update_config)
                self.ui_data[component_tag] = config_window.config_dict
                config_window.transient(self.root)
            else:
                config_window = config_class(self.root, component_tag, self.update_config, self.ui_data[component_tag],
                                             result_data)
                config_window.transient(self.root)

    def on_canvas_right_click(self, event):
        """右键点击取消当前操作"""
        self.current_dragging_type = None
        self.current_wire_start = None
        self.canvas_area.delete("preview")
        self.canvas_area.delete("wire_preview")
        self.canvas_area.config(cursor="arrow")

    def on_component_press(self, event):
        """组件按下事件（记录初始位置）"""
        clicked_item = self.canvas_area.find_overlapping(event.x - 1, event.y - 1, event.x + 1, event.y + 1)[0]
        component_tag = next((tag for tag in self.canvas_area.gettags(clicked_item) if tag.startswith("component_")),
                             None)
        if not component_tag:
            return

        component_info = self.components[component_tag]
        self.drag_data = {
            "component_tag": component_tag,
            "press_x": event.x,
            "press_y": event.y,
            "orig_x": component_info["x"],
            "orig_y": component_info["y"]
        }

    def on_component_drag(self, event):
        """组件拖动事件（使用绝对位置计算）"""
        self.current_wire_start = None
        if not self.drag_data:
            return

        component_tag = self.drag_data["component_tag"]
        component_info = self.components[component_tag]

        delta_x = event.x - self.drag_data["press_x"]
        delta_y = event.y - self.drag_data["press_y"]

        new_x = self.drag_data["orig_x"] + delta_x
        new_y = self.drag_data["orig_y"] + delta_y

        for elem_id in component_info["elements"]:
            elem_type = self.canvas_area.type(elem_id)
            if elem_type == "rectangle":

                self.canvas_area.coords(
                    elem_id,
                    new_x - 40, new_y - 30,
                    new_x + 40, new_y + 30
                )
            elif elem_type == "text":

                current_text = self.canvas_area.itemcget(elem_id, "text")
                if current_text in ["🌍", "📡", "📞", "💻", "📊"]:
                    self.canvas_area.coords(elem_id, new_x, new_y - 10)
                else:
                    self.canvas_area.coords(elem_id, new_x, new_y + 18)

        component_info["x"] = new_x
        component_info["y"] = new_y

        self._update_wires_for_component(component_tag, new_x, new_y)

    def on_component_release(self, event):
        """组件释放事件（清除拖拽数据和预览）"""
        self.drag_data = None
        self.canvas_area.delete("preview")
        self.canvas_area.config(cursor="arrow")

    def _update_wires_for_component(self, component_tag, new_x, new_y):
        """更新与组件关联的所有连线坐标"""
        for wire in self.wires:
            if wire["start_tag"] == component_tag:
                end_component = self.components[wire["end_tag"]]
                self.canvas_area.coords(wire["id"], new_x, new_y, end_component["x"], end_component["y"])
            elif wire["end_tag"] == component_tag:
                start_component = self.components[wire["start_tag"]]
                self.canvas_area.coords(wire["id"], start_component["x"], start_component["y"], new_x, new_y)

    def _create_wire(self, start_info, end_info):
        """创建卫星通信链路连线"""
        wire_id = self.canvas_area.create_line(
            start_info["x"], start_info["y"],
            end_info["x"], end_info["y"],
            fill="#4299e1",
            width=2,
            arrow=tk.LAST,
            arrowshape=(10, 12, 5)
        )
        self.wires.append({
            "id": wire_id,
            "start_tag": start_info["tag"],
            "end_tag": end_info["tag"]
        })

    def relation_check(self):
        rel = {}
        for line in self.wires:
            start = line['start_tag']
            end = line['end_tag']

            if start not in rel:
                rel[start] = {"up": [], "down": []}
            if end not in rel[start]['down']:
                rel[start]['down'].append(end)
            if end not in rel:
                rel[end] = {"up": [], "down": []}
            if start not in rel[end]['up']:
                rel[end]['up'].append(start)

        return rel

    def update_config(self, data, com_id):
        self.ui_data[com_id] = data

    def generate_logic(self):
        self.logic_data = {}
        for component in self.ui_data:
            ui_class = self.components[component].get("config_class").__name__
            logic_class_name = ui_class.replace("UI", "")
            logic_class = globals().get(logic_class_name)
            self.logic_data[component] = {"logic_class": logic_class, "result_data": {}, "run_data": {}}

    def run_logic(self, com_id):
        if com_id is None:
            return
        logic_class = self.logic_data[com_id]['logic_class']
        config = self.ui_data[com_id]
        logic = logic_class(config)
        if not self.rel[com_id]['up']:
            self.logic_data[com_id]["run_data"] = logic.run()
        else:
            up = self.rel[com_id]['up'][0]
            up_data = self.logic_data[up]["run_data"]
            print(f"updata{up_data}")
            self.logic_data[com_id]["run_data"] = logic.run(up_data)
        self.logic_data[com_id]["result_data"] = logic.monitor()
        if not self.rel[com_id]['down']:
            self.run_logic(None)
        else:
            for down in self.rel[com_id]['down']:
                self.run_logic(down)

    def start_simulation(self):
        """开始仿真"""
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.is_simu = True
        self.rel = self.relation_check()
        configed = True
        for com in self.rel:
            if com not in self.ui_data:
                configed = False
                break
        if not configed:
            self.stop_simulation()
            self.status_label.config(text="请先完成配置")
            return

        self.status_label.config(text="卫星通信仿真进行中...")
        threading.Thread(target=self.run_simulation_logic, daemon=True).start()

    def run_simulation_logic(self):
        self.generate_logic()
        start = None
        for com in self.rel:
            if not self.rel[com]['up']:
                start = com
                break
        self.run_logic(start)
        self.root.after(0, self.simulation_completed)

    def simulation_completed(self):
        if self.is_simu:
            self.stop_simulation()

    def stop_simulation(self):
        """结束仿真"""
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.is_simu = False
        self.status_label.config(text="卫星通信仿真结束")


if __name__ == "__main__":
    root = tk.Tk()
    app = SatcomSimulationUI(root)
    root.mainloop()
