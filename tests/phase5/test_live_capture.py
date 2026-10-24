import struct
from pathlib import Path

import numpy as np
import pytest

from hyperweave.demo.live_capture import LiveCapture, _decode_packet, _flow_vector


pytestmark = pytest.mark.phase5


def _ethernet_tcp_packet():
    ethernet = bytes.fromhex("00112233445566778899aabb0800")
    ipv4 = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 40, 1, 0, 64, 6, 0,
                       bytes((192, 0, 2, 10)), bytes((198, 51, 100, 20)))
    tcp = struct.pack("!HHIIHHHH", 51000, 443, 0, 0, 0x5002, 4096, 0, 0)
    return ethernet + ipv4 + tcp


def test_live_packet_decoder_and_flow_schema_mapping():
    packet = _decode_packet(_ethernet_tcp_packet(), 1, 1234.5)
    assert packet["src"] == "192.0.2.10"
    assert packet["dst"] == "198.51.100.20"
    assert (packet["sport"], packet["dport"], packet["protocol"]) == (51000, 443, 6)
    assert packet["ip_bytes"] == 40 and packet["flags"] == 2

    features = Path("configs/nf_v2_features.txt").read_text().splitlines()
    flow = {"first": 1234.5, "last": 1234.75, "forward_packets": 1, "reverse_packets": 0,
            "forward_bytes": 40, "reverse_bytes": 0, "lengths": [40], "ttl": [64],
            "flags": 2, "tcp_window": 4096, "protocol": 6, "src": packet["src"],
            "dst": packet["dst"], "sport": packet["sport"], "dport": packet["dport"]}
    vector = _flow_vector(flow, features)
    assert vector.shape == (43,)
    assert vector[features.index("IN_BYTES")] == 40
    assert vector[features.index("FLOW_DURATION_MILLISECONDS")] == 250
    assert vector[features.index("L7_PROTO")] == 0
    assert np.isfinite(vector).all()


def test_live_packet_decoder_ignores_non_ip_and_truncated_frames():
    assert _decode_packet(bytes(13), 1, 0) is None
    assert _decode_packet(bytes.fromhex("00112233445566778899aabb0800") + b"\x45", 1, 0) is None


def test_interface_list_prioritizes_connected_interfaces_and_hides_disconnected(monkeypatch):
    class Result:
        stdout = ("1.ap1 [Up, Running, Wireless, Not associated]\n"
                  "2.lo0 [Up, Running, Loopback]\n"
              "3.awdl0 [Up, Running, Wireless, Associated]\n"
              "4.en0 [Up, Running, Wireless, Associated]\n"
              "5.en1 [Up, Disconnected]\n")

    monkeypatch.setattr("hyperweave.demo.live_capture.shutil.which", lambda _: "/usr/sbin/tcpdump")
    monkeypatch.setattr("hyperweave.demo.live_capture.subprocess.run", lambda *args, **kwargs: Result())
    interfaces = LiveCapture.interfaces()["interfaces"]
    assert [item["name"] for item in interfaces] == ["en0", "awdl0", "lo0"]
