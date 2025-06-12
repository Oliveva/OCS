from collections import defaultdict

from . import Protocol  # 假设包含FrameProtocol和IPProtocol的模块
import struct


class Route:
    """
    Routing table.

    Attributes:
        route_table (dict):     The routing table. Structure: {destination IP: next-hop IP}
    """

    def __init__(self):
        self.route_table = {}
        self.ip_protocol = Protocol.IPProtocol(0x00000000, 0x00000000)

    def add_route(self, dst_ip: int, next_ip: int):
        """
        Add route entry.

        Args:
            dst_ip (int):      Destination IP.
            next_ip (int):     Next-hop IP.
        """

        self.route_table[dst_ip] = next_ip

    def route_data(self, raw_ip: bytes):
        """
        Get IP address from IP datagram and find next hop from routing table.

        Args:
            raw_ip (bytes): The IP datagram.

        Retuen:
            int:            next hop IP.
        """

        parsed_ip = self.ip_protocol.unpack(raw_ip)
        dst_ip = parsed_ip['dst_ip']

        if dst_ip in self.route_table:
            return self.route_table[dst_ip]
        else:
            print(f"No route entry for destination address {self.parse_ip(dst_ip)}")
            return None

    def parse_address(self, address: bytes):
        orbit, satellite, terminal = struct.unpack('>HHH', address)
        return {
            'orbit_id': orbit,
            'satellite_id': satellite,
            'terminal_id': terminal
        }

    def parse_ip(self, ip_int: int):
        """
        Convert a 32-bit integer IP address to dotted decimal notation.

        Args:
            ip_int (int):   32-bit integer IP address.

        Return:
            str:            dotted decimal notation.
        """
        # 0 ≤ ip_int ≤ 0xFFFFFFFF
        if not (0 <= ip_int <= 0xFFFFFFFF):
            raise ValueError(f"{ip_int} should be in 0~4294967295.")

        b1 = (ip_int >> 24) & 0xFF
        b2 = (ip_int >> 16) & 0xFF
        b3 = (ip_int >> 8) & 0xFF
        b4 = ip_int & 0xFF

        return f"{b1}.{b2}.{b3}.{b4}"
