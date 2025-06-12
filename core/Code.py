import numpy as np
from commpy.channelcoding import conv_encode, viterbi_decode
from commpy.utilities import dec2bitarray, bitarray2dec
from commpy.channelcoding.convcode import Trellis


class ConvCode:
    """
    Convolutional code coder.

    Args:
        g_matrix (2D ndarray):          Generator matrix G(D) of the convolutional encoder. Each element of G(D)
                                        represents a polynomial. Default: [[0o1, 0o13]]
        memory (1D ndarray of ints):    Number of memory elements per input of the convolutional encoder. Default: [2]
    """

    def __init__(self, g_matrix: np.ndarray = np.array([[0o1, 0o13]]), memory: np.ndarray = np.array([2])):
        self.trellis = Trellis(
            memory=memory,
            g_matrix=g_matrix
        )

    def bytes_to_bits(self, byte_stream: bytes) -> np.ndarray:
        """
        Convert byte stream to big-endian bit stream.

        Args:
            byte_stream (bytes):    Byte stream.

        Returns:
            ndarray :            The big-endian bit stream.
        """

        bit_list = []
        for byte in byte_stream:
            bits = dec2bitarray(byte, 8)
            bit_list.extend(bits.tolist())
        return np.array(bit_list, dtype=np.int8)

    def bits_to_bytes(self, bit_stream: np.ndarray) -> bytes:
        """
        Convert big-endian bit stream to byte stream.

        Args:
            bit_stream (ndarray):   The big-endian bit stream.

        Returns:
            bytes :                 Byte stream.
        """

        byte_len = len(bit_stream) // 8
        truncated_bits = bit_stream[:byte_len * 8]
        byte_list = [bitarray2dec(truncated_bits[i * 8:(i + 1) * 8]) for i in range(byte_len)]
        return bytes(byte_list)

    def encode(self, byte_stream: bytes) -> np.ndarray:
        """
        Convolutional code encoder.

        Args:
            byte_stream (bytes):    Byte stream.

        Returns:
            ndarray :               The encoded bits.
        """

        original_bits = self.bytes_to_bits(byte_stream)
        encoded_bits = conv_encode(original_bits, self.trellis)
        return encoded_bits

    def decode(self, encoded_bits: np.ndarray) -> bytes:
        """
        Convolutional code decoder.

        Args:
            encoded_bits (ndarray):     The encoded bits.

        Returns:
            bytes :                     The decoded byte stream.
        """
        decode_bits = viterbi_decode(encoded_bits, self.trellis)
        return self.bits_to_bytes(decode_bits.astype(np.int8))

if __name__ == "__main__":
    import Converter, Protocol, struct

    signal = Converter.GeneratedSignalInput()
    adc = Converter.ADC()
    sig, t = adc.convert(signal)
    ip = Protocol.IPProtocol(0xC0A80101, 0x0A000002)
    src_addr = struct.pack('>HHH', 0x0001, 0x0002, 0x0003)
    dst_addr = struct.pack('>HHH', 0x0004, 0x0005, 0x0006)
    fr = Protocol.FrameProtocol(src_addr, dst_addr)
    datagram_fragments = ip.pack(sig, mtu=1500)
    conv = ConvCode(memory=np.array([2]))
    frame_fragments = []
    for fragment in datagram_fragments:

        ip_datagram = fragment

        data_frame = fr.encapsulate(ip_datagram, frame_type=0x01)
        print(f"Generated data frame length: {len(data_frame)} bytes")
        code = conv.encode(data_frame)
        d = conv.decode(code)
        print(code)
        print(d == data_frame)
        try:
            parsed_frame = fr.decapsulate(d)
            print("Frame data unpacked successfully:")
            print(f"  Frame type: 0x{parsed_frame['frame_type']:02X}")
            print(f"  Sequence: {parsed_frame['sequence']}")
            print(f"  Payload length: {len(parsed_frame['payload'])} bytes")
            print(f"  Timestamp: {parsed_frame['timestamp']} ms")

            if parsed_frame['payload'] == ip_datagram:
                print("CRC check passed.")
                frame_fragments.append(parsed_frame['payload'])
        except ValueError as e:
            print(f"Unpacking failed: {e}")
