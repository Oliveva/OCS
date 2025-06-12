import numpy as np
import struct
from collections import defaultdict, deque
import crcmod.predefined


class IPProtocol:
    """
    Internet Protocol.

    Args:
        src_ip (int):   The source IP address.
        dst_ip (int):   The destination IP address.
    """

    def __init__(self, src_ip: int, dst_ip: int):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.header_length = 20

    def _fragment_adc_data(self, adc_data: np.ndarray, mtu: int = 1500):
        """
        Split ADC data into fragmented payloads that comply with MTU restrictions.

        Args:
            adc_data (ndarray): Original ADC data.
            mtu (int):          Length of MTU.

        Return:
            list :              Fragmented payloads.
        """
        self.header_length = 20
        max_payload_per_frag = mtu - self.header_length
        if max_payload_per_frag <= 0:
            raise ValueError(f"MTU({mtu})must longer than header length ({self.header_length})")

        # Payload must be aligned to 8-byte boundaries (fragment offset unit is 8 bytes)
        max_payload_aligned = (max_payload_per_frag // 8) * 8
        if max_payload_aligned < 8:
            raise ValueError("MTU is too small to accommodate the minimum fragmented payload (8 bytes)")

        total_payload = adc_data.nbytes  # Total bytes of the original payload (16bit => 2bytes/sample) 
        fragments = []
        offset = 0
        is_frag = 0
        if total_payload <= 65515:
            is_frag = 1
            while offset < total_payload:
                current_payload = min(max_payload_aligned, total_payload - offset)
                if current_payload % 2 != 0:
                    current_payload -= 1

                start_idx = offset // 2
                end_idx = start_idx + (current_payload // 2)
                frag_data = adc_data[start_idx:end_idx]

                frag_offset = offset // 8

                more_fragments = 1 if (offset + current_payload) < total_payload else 0

                fragments.append((frag_data, frag_offset, more_fragments, is_frag))
                offset += current_payload
        else:
            fragments.append((adc_data, offset, 0, is_frag))
        return fragments

    def pack(self, adc_data: np.ndarray, fragment_id: int = 0x1234, mtu: int = 1500):
        """
        Pack ADC data into IP datagrams

        Args:
            adc_data (ndarray): Original ADC data.
            fragment_id (int):  Fragment ID.
            mtu (int):          Length of Maximum Transmission Unit.

        Return:
            list[bytes] :       List of IP datagrams.
        """

        # generate fragmented payloads
        try:
            fragments = self._fragment_adc_data(adc_data, mtu)
        except ValueError as e:
            raise RuntimeError(f"Fragment failed: {e}")

        datagrams = []
        for frag_data, frag_offset, more_frags, is_frag in fragments:

            # Payload Data (big-endian 16-bit)
            payload = frag_data.astype('>i2').tobytes()
            payload_length = len(payload)

            head = bytearray()

            # Version & Extend (IPv4=0x4, no extend=0x0 → 0x40)
            head.append(0x40)

            # Total Length (0, will be calculated later)
            head.extend(struct.pack('>H', 0))  # index 1-2

            # Fragmentation Control Field (32bit)
            # 16-bit identifier | 3-bit type (highest bit = MF, remaining bits reserved) | 13-bit offset
            frag_type = (is_frag << 2 | more_frags << 1)
            fragmentation_control = (fragment_id << 16) | (frag_type << 13) | frag_offset
            head.extend(struct.pack('>I', fragmentation_control))  # index 3-6

            # TTL (8bit, fixed at 255)
            head.append(0xFF)  # index 7

            # Header Checksum (0, will be calculated later)
            head.extend(struct.pack('>H', 0))  # index 8-9

            # Reserved Field (16bit=0)
            head.extend(struct.pack('>H', 0))  # index 10-11

            # source/destination IP (each 32bit, big-endian)
            head.extend(struct.pack('>I', self.src_ip))  # index 12-15 (source IP)
            head.extend(struct.pack('>I', self.dst_ip))  # index 16-19 (destination IP)

            # calculate total length (20 bytes header + payload length)
            if is_frag == 1:
                total_length = 20 + payload_length
                struct.pack_into('>H', head, 1, total_length)

            # calculate the header checksum (header only)
            checksum = self._calculate_header_checksum(head)
            struct.pack_into('>H', head, 8, checksum)

            datagram = head + payload
            datagrams.append(datagram)

        return datagrams

    def unpack(self, datagram: bytes):
        """
        Unpack IP datagram into the information it carries.

        Args:
            datagram (bytes):   IP datagram.

        Return:
            dict :              Information of IP datagram.
        """

        # verify the minimum length of the datagram
        if len(datagram) < 20:
            raise ValueError("Datagram length too short (at least 20 bytes for the header) ")

        fields = {}

        # Version & Extend (index 0)
        version_ext = datagram[0]
        fields['version'] = (version_ext >> 4) & 0x0F
        fields['ext_header_type'] = version_ext & 0x0F

        # Fragmentation Control Field (index 3-6)
        frag_control = struct.unpack('>I', datagram[3:7])[0]
        fields['fragment_id'] = (frag_control >> 16) & 0xFFFF
        frag_type = (frag_control >> 13) & 0x07
        fields['is_frag'] = (frag_type >> 2) & 0x01
        fields['mf_flag'] = (frag_type >> 1) & 0x01
        fields['frag_offset'] = frag_control & 0x1FFF

        # Total Length (index 1-2)
        if fields['is_frag'] == 1:
            fields['total_length'] = struct.unpack('>H', datagram[1:3])[0]
        else:
            fields['total_length'] = len(datagram)
        if fields['total_length'] != len(datagram):
            raise ValueError(
                f"Total length mismatch (datagram length {len(datagram)}, declared length {fields['total_length']})"
            )

        # TTL (index 7) 
        fields['ttl'] = datagram[7]

        # Header Checksum (index 8-9) 
        fields['header_checksum'] = struct.unpack('>H', datagram[8:10])[0]

        # Reserved Field (index 10-11)
        fields['reserved'] = struct.unpack('>H', datagram[10:12])[0]

        # source/destination IP (index 12-19)
        fields['src_ip'] = struct.unpack('>I', datagram[12:16])[0]
        fields['dst_ip'] = struct.unpack('>I', datagram[16:20])[0]

        # Payload
        payload_length = fields['total_length'] - 20
        fields['payload'] = datagram[20:20 + payload_length]

        # verify the header checksum (header only)
        if not self._verify_header_checksum(datagram[:20], fields['header_checksum']):
            raise ValueError("Header checksum verification failed.")

        return fields

    def _calculate_header_checksum(self, header_bytes: bytes):
        """
        Calculate the header checksum.

        Args:
            header_bytes (bytes):   The header of IP datagram.

        Return:
            int :                   Header Checksum.
        """

        head_padded = bytearray(header_bytes)

        if len(head_padded) % 2 != 0:
            head_padded.append(0)

        checksum = 0
        for i in range(0, len(head_padded), 2):
            word = (head_padded[i] << 8) | head_padded[i + 1]
            checksum += word

        while checksum >> 16:
            checksum = (checksum >> 16) + (checksum & 0xFFFF)

        return ~checksum & 0xFFFF

    def _verify_header_checksum(self, header_bytes: bytes, reported_checksum: int):
        """
        Verify the correctness of the checksum.

        Args:
            header_bytes (bytes):       The header of IP datagram.
            reported_checksum (int):    Reported checksum.

        Return:
            bool:                       Ture: Verification passed;
                                        False: Verification failed.
        """

        header_for_calculation = bytearray(header_bytes)
        struct.pack_into('>H', header_for_calculation, 8, 0)

        computed_checksum = self._calculate_header_checksum(header_for_calculation)

        return computed_checksum == reported_checksum

    def reassemble(self, fragments: list):
        """
        Reassemble fragmented datagrams of the same group.

        Args:
            fragments(list):  The fragmented datagrams. (List of return value of self.unpack())

        Return:
            np.ndarray: Original payload data.
        """

        frag_group = defaultdict(list)
        for frag in fragments:
            frag_group[frag['fragment_id']].append(frag)

        if not frag_group:
            return None
        datagram_id = next(iter(frag_group.keys()))
        datagram_frags = sorted(frag_group[datagram_id], key=lambda x: x['frag_offset'])

        expected_offset = 0
        total_payload = bytearray()
        for frag in datagram_frags:
            if frag['frag_offset'] != expected_offset:
                print(
                    f"Fragments are not contiguous (expected offset is {expected_offset}, actually is "
                    f"{frag['frag_offset']}). "
                )
                return None

            total_payload += frag['payload']

            if frag['mf_flag'] == 0:
                break
            expected_offset = frag['frag_offset'] + (len(frag['payload']) // 8)

        try:
            adc_data = np.frombuffer(total_payload, dtype='>i2')
        except ValueError:
            return None

        return adc_data


class FrameProtocol:
    def __init__(self, src_address: bytes, dst_address: bytes):
        if len(src_address) != 6 or len(dst_address) != 6:
            raise ValueError("Address length should be 6 bytes")
        self.src_address = src_address
        self.dst_address = dst_address
        self._sequence = 0
        self.frame_header_length = 32
        self.sync_header = b'\xab\xcd\x12\x34'
        self.end_marker = b'\x55\xaa'

    def encapsulate(self, payload: bytes, frame_type: int = 0x01) -> bytes:
        frame_header = bytearray()
        frame_header.extend(self.sync_header)
        frame_header.append(1)
        frame_header.append(frame_type)
        frame_header.extend(self.src_address)
        frame_header.extend(self.dst_address)
        frame_header.extend(struct.pack('>I', self._get_sequence()))
        frame_header.extend(struct.pack('>Q', self._get_timestamp()))
        frame_header.extend(b'\x00\x00')

        crc16 = self._calculate_crc16(frame_header[:self.frame_header_length])
        frame_header.extend(struct.pack('>H', crc16))

        frame_payload = bytearray(payload)
        crc32 = self._calculate_crc32(frame_payload)
        frame_payload.extend(struct.pack('>I', crc32))

        full_frame = bytes(frame_header) + frame_payload + self.end_marker
        return full_frame

    def decapsulate(self, frame: bytes) -> dict:
        if len(frame) < self.frame_header_length + 2 + 2:
            raise ValueError("Frame length is insufficient.")
        if frame[-2:] != self.end_marker:
            raise ValueError("Frame end marker error.")

        header = frame[:self.frame_header_length + 2]
        sync = header[:4]
        if sync != self.sync_header:
            raise ValueError("Sync header error.")

        version = header[4]
        frame_type = header[5]
        src_addr = header[6:12]
        dst_addr = header[12:18]
        sequence = struct.unpack('>I', header[18:22])[0]
        timestamp = struct.unpack('>Q', header[22:30])[0]
        reserved = header[30:32]
        reported_crc16 = struct.unpack('>H', header[32:34])[0]

        actual_crc16 = self._calculate_crc16(header[:self.frame_header_length])
        if actual_crc16 != reported_crc16:
            raise ValueError("Frame header CRC check failed.")

        payload_start = self.frame_header_length + 2
        payload_end = len(frame) - 2
        payload = frame[payload_start:payload_end]
        if len(payload) >= 4:
            reported_crc32 = struct.unpack('>I', payload[-4:])[0]
            payload = payload[:-4]
        else:
            reported_crc32 = 0

        if payload:
            actual_crc32 = self._calculate_crc32(payload)
            if actual_crc32 != reported_crc32:
                raise ValueError("Payload CRC check failed.")

        return {
            'sync_header': sync,
            'version': version,
            'frame_type': frame_type,
            'src_address': src_addr,
            'dst_address': dst_addr,
            'sequence': sequence,
            'timestamp': timestamp,
            'reserved': reserved,
            'payload': payload,
            'end_marker': self.end_marker
        }

    def _get_sequence(self) -> int:
        seq = self._sequence
        self._sequence = (self._sequence + 1) % (1 << 32)
        return seq

    def _get_timestamp(self) -> int:

        return int(np.round((np.datetime64('now') - np.datetime64('1970-01-01')) / np.timedelta64(1, 'ms')))

    @staticmethod
    def _calculate_crc16(data: bytes) -> int:
        crc16 = crcmod.predefined.Crc('crc-16')
        crc16.update(data)
        return crc16.crcValue

    @staticmethod
    def _calculate_crc32(data: bytes) -> int:
        crc32 = crcmod.predefined.Crc('crc-32')
        crc32.update(data)
        return crc32.crcValue


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Converter import *

    ADConverter = ADC(sample_rate=9000)
    signal = GeneratedSignalInput()
    td, sig = ADConverter.convert(signal)

    adc_original = sig

    src_ip = 0xC0A80101  # 192.168.1.1
    dst_ip = 0x0A000002  # 10.0.0.2
    IP = IPProtocol(src_ip, dst_ip)
    datagram_fragments = IP.pack(adc_original, mtu=1500)

    src_addr = struct.pack('>HHH', 0x0001, 0x0002, 0x0003)
    dst_addr = struct.pack('>HHH', 0x0004, 0x0005, 0x0006)
    print(src_addr)
    frame_protocol = FrameProtocol(src_addr, dst_addr)
    frame_fragments = []

    for fragment in datagram_fragments:
        ip_datagram = fragment

        data_frame = frame_protocol.encapsulate(ip_datagram, frame_type=0x01)
        print(f"Data frame length: {len(data_frame)} bytes")

        try:
            parsed_frame = frame_protocol.decapsulate(data_frame)
            print("Decapsulate successfully: ")
            print(f"  type: 0x{parsed_frame['frame_type']:02X}")
            print(f"  sequence: {parsed_frame['sequence']}")
            print(f"  payload length: {len(parsed_frame['payload'])} bytes")
            print(f"  timestamp: {parsed_frame['timestamp']} ms")
            if parsed_frame['payload'] == ip_datagram:
                frame_fragments.append(parsed_frame['payload'])
        except ValueError as e:
            print(f"Fail to decapsulate: {e}")

    parsed_fragments = []
    for frag in frame_fragments:
        try:
            parsed = IP.unpack(frag)
            parsed_fragments.append(parsed)
            print(f"Unpack fragment {len(parsed_fragments)}:")
            print(f"  id: 0x{parsed['fragment_id']:04X}")
            print(f"  MF: {parsed['mf_flag']}")
            print(f"  offset: {parsed['frag_offset']} (8bytes/) ")
            print(f"  payload length: {len(parsed['payload'])} bytes\n")
        except ValueError as e:
            print(f"Fail to unpack: {e}")

    adc_reassembled = IP.reassemble(parsed_fragments)
    if adc_reassembled is not None:
        print("Reassemble successfully.")
        print(f"original ADC data length: {len(adc_original)}")
        print(f"reassembled ADC data length: {len(adc_reassembled)}")
        print("data verification:", np.array_equal(adc_original, adc_reassembled))
    else:
        print("Fail to reassemble.")

    import matplotlib.pyplot as plt

    plt.subplot(211)
    plt.plot(td, adc_original)
    plt.subplot(212)
    plt.plot(td, adc_reassembled)
    plt.show()
