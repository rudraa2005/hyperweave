"""Passive tcpdump capture, approximate NF-v2 flow aggregation, and live scoring."""
import ipaddress
import json
import re
import shutil
import struct
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np


class LiveCapture:
    def __init__(self, model_state):
        self.state = model_state
        self.lock = threading.RLock()
        self.process = None
        self.interface = None
        self.worker = None
        self.startup_event = threading.Event()
        self.ready = False
        self.started_at = None
        self.error = None
        self.packet_count = 0
        self.byte_count = 0
        self.flow_count = 0
        self.alert_count = 0
        self.last_rate_at = None
        self.last_packets = 0
        self.last_bytes = 0
        self.packet_rate = 0.0
        self.bit_rate = 0.0
        self.flows = {}
        self.events = deque(maxlen=400)
        self.agent_events = deque(maxlen=160)
        self.sequence = 0
        self.agent_sequence = 0

    @staticmethod
    def interfaces():
        binary = shutil.which("tcpdump")
        if not binary:
            return {"available": False, "error": "tcpdump is not installed", "interfaces": []}
        try:
            result = subprocess.run([binary, "-D"], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError) as exc:
            return {"available": False, "error": str(exc), "interfaces": []}
        interfaces = []
        for line in result.stdout.splitlines():
            match = re.match(r"\s*\d+\.([^ ]+)\s+\[(.*)\]", line)
            if match:
                status = match.group(2)
                normalized = status.lower()
                if any(state in normalized for state in ("disconnected", "not associated", "[none]", "none")):
                    continue
                interfaces.append({"name": match.group(1), "status": status})
        def priority(item):
            name, status = item["name"].lower(), item["status"].lower()
            associated = "associated" in status and "not associated" not in status
            if name.startswith("en") and associated:
                return 0
            if associated and not name.startswith("awdl"):
                return 1
            if "loopback" in status:
                return 3
            return 2

        interfaces.sort(key=lambda item: (priority(item), item["name"]))
        return {"available": True, "interfaces": interfaces}

    def start(self, interface):
        if not isinstance(interface, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,32}", interface):
            raise ValueError("invalid network interface name")
        available = {item["name"] for item in self.interfaces()["interfaces"]}
        if interface not in available:
            raise ValueError("interface is not available to tcpdump")
        binary = shutil.which("tcpdump")
        if not binary:
            raise RuntimeError("tcpdump is not installed")
        with self.lock:
            if self.process and self.process.poll() is None:
                raise RuntimeError("packet capture is already running")
            self.error = None
            self.ready = False
            self.startup_event.clear()
            self.interface = interface
            self.started_at = time.time()
            self.packet_count = self.byte_count = self.flow_count = self.alert_count = 0
            self.packet_rate = self.bit_rate = 0.0
            self.last_rate_at = self.started_at
            self.last_packets = self.last_bytes = 0
            self.flows.clear()
            self.events.clear()
            self.agent_events.clear()
            self.process = subprocess.Popen([binary, "-i", interface, "-n", "-s", "0", "-U", "-w", "-"],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            self.worker = threading.Thread(target=self._read_capture, args=(self.process,), daemon=True)
            self.worker.start()
            process = self.process
        if not self.startup_event.wait(timeout=5):
            self.stop()
            raise RuntimeError("tcpdump did not become ready within five seconds")
        with self.lock:
            if self.error:
                message = self.error
            elif process.poll() is not None:
                message = f"tcpdump exited during startup with status {process.returncode}"
            else:
                return {"status": "capturing", "ready": True, "interface": interface,
                        "source": "live_passive_packet_capture"}
        self.stop()
        raise RuntimeError(_capture_error_hint(message))

    def stop(self):
        with self.lock:
            process = self.process
            worker = self.worker
            if process and process.poll() is None:
                process.terminate()
        if process:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        if worker and worker is not threading.current_thread():
            worker.join(timeout=1)
        with self.lock:
            self.process = None
            self.interface = None
            self.ready = False
            return self.status()

    def status(self):
        running = bool(self.process and self.process.poll() is None)
        return {"running": running, "ready": bool(running and self.ready),
                "interface": self.interface if running else None,
                "error": self.error, "packets": self.packet_count, "bytes": self.byte_count,
                "flows": self.flow_count, "alerts": self.alert_count,
                "packets_per_second": self.packet_rate, "bits_per_second": self.bit_rate,
                "uptime_seconds": max(0.0, time.time() - self.started_at) if self.started_at else 0.0,
                "classification_mode": "trained_int8_checkpoint_on_approximate_nf_v2_flow_features",
                "ground_truth_available": False,
                "feature_note": "Packet-derived flow features are approximate; application-specific NF-v2 fields are unavailable."}

    def snapshot(self, after=0):
        with self.lock:
            return {**self.status(), "events": [event for event in self.events if event["id"] > after],
                    "agents": list(self.agent_events)}

    def _read_capture(self, process):
        try:
            header = _read_exact(process.stdout, 24)
            if len(header) != 24:
                details = process.stderr.read(4096).decode(errors="replace").strip()
                raise RuntimeError(details or "capture ended before pcap header arrived")
            magic = header[:4]
            endian = {b"\xd4\xc3\xb2\xa1": "<", b"\xa1\xb2\xc3\xd4": ">",
                      b"\x4d\x3c\xb2\xa1": "<", b"\xa1\xb2\x3c\x4d": ">"}.get(magic)
            if endian is None:
                raise RuntimeError("unsupported pcap stream format")
            linktype = struct.unpack(endian + "I", header[20:24])[0]
            with self.lock:
                self.ready = True
            self.startup_event.set()
            while process.poll() is None:
                record = _read_exact(process.stdout, 16)
                if not record:
                    break
                if len(record) != 16:
                    break
                seconds, fraction, captured, original = struct.unpack(endian + "IIII", record)
                frame = _read_exact(process.stdout, captured)
                if len(frame) != captured:
                    break
                packet = _decode_packet(frame, linktype, seconds + fraction / (1e9 if b"\x4d" in magic else 1e6))
                if packet:
                    self._observe(packet, original)
            if process.poll() not in (None, 0, -15):
                message = process.stderr.read(4096).decode(errors="replace").strip()
                raise RuntimeError(message or f"tcpdump exited with status {process.returncode}")
        except Exception as exc:
            with self.lock:
                self.error = _capture_error_hint(str(exc))
            self.startup_event.set()

    def _observe(self, packet, frame_bytes):
        now = packet["time"]
        flow_bytes = packet["ip_bytes"]
        key_parts = ((packet["src"], packet["sport"]), (packet["dst"], packet["dport"]))
        key = (packet["protocol"], *sorted(key_parts))
        with self.lock:
            flow = self.flows.get(key)
            if flow and now - flow["last"] > 120:
                del self.flows[key]
                flow = None
            if flow is None:
                flow = {"key": key, "first_src": key_parts[0], "first": now, "last": now,
                        "forward_bytes": 0, "reverse_bytes": 0, "forward_packets": 0, "reverse_packets": 0,
                        "lengths": [], "ttl": [], "flags": 0, "tcp_window": 0, "protocol": packet["protocol"],
                        "src": packet["src"], "dst": packet["dst"], "sport": packet["sport"], "dport": packet["dport"]}
                self.flows[key] = flow
                self.flow_count += 1
            forward = key_parts[0] == flow["first_src"]
            direction = "forward" if forward else "reverse"
            flow[f"{direction}_bytes"] += flow_bytes
            flow[f"{direction}_packets"] += 1
            flow["last"] = now
            flow["lengths"].append(flow_bytes)
            if len(flow["lengths"]) > 256:
                flow["lengths"].pop(0)
            flow["ttl"].append(packet["ttl"])
            flow["ttl"] = flow["ttl"][-256:]
            flow["flags"] |= packet["flags"]
            flow["tcp_window"] = max(flow["tcp_window"], packet["window"])
            self.packet_count += 1
            self.byte_count += frame_bytes
            vector = _flow_vector(flow, self.state.normalizer.features)
            normalized = self.state.normalizer.transform(vector.reshape(1, -1))
            model = self.state.current["int8"]
            logit = float(model.logits(normalized)[0])
            alert = logit > self.state.current["threshold"]
            if alert:
                self.alert_count += 1
            self.sequence += 1
            event = {"id": self.sequence, "timestamp": now, "src": packet["src"], "dst": packet["dst"],
                     "src_port": packet["sport"], "dst_port": packet["dport"],
                     "protocol": packet["protocol"], "packet_bytes": frame_bytes,
                     "flow_packets": flow["forward_packets"] + flow["reverse_packets"],
                     "flow_bytes": flow["forward_bytes"] + flow["reverse_bytes"],
                     "duration_ms": max(0.0, (flow["last"] - flow["first"]) * 1000),
                     "score": logit, "alert": bool(alert), "features_approximate": True}
            self.events.append(event)
            self._update_rates(now)
            if self.sequence % 500 == 0:
                stale = [flow_key for flow_key, value in self.flows.items() if now - value["last"] > 120]
                for flow_key in stale:
                    del self.flows[flow_key]
                while len(self.flows) > 5000:
                    self.flows.pop(next(iter(self.flows)))
            if self.sequence % 10 == 1 or alert:
                self._run_agents(event, normalized[0], model)

    def _update_rates(self, now):
        elapsed = now - (self.last_rate_at or now)
        if elapsed >= .5:
            self.packet_rate = (self.packet_count - self.last_packets) / elapsed
            self.bit_rate = (self.byte_count - self.last_bytes) * 8 / elapsed
            self.last_packets, self.last_bytes, self.last_rate_at = self.packet_count, self.byte_count, now

    def _run_agents(self, event, normalized, model):
        abs_features = np.abs(normalized)
        outlier_count = int(np.count_nonzero(abs_features > 5))
        threshold = self.state.current["threshold"]
        score = event["score"]
        active = int(np.count_nonzero(self.state.current["baked"].gate_mask))
        results = [
            ("Neuron Architect", f"Scored the current flow with {active} active generated neurons; {model.w1.shape[2]} input features."),
            ("Training Strategist", f"Checked normalized feature drift: {outlier_count} of 43 values exceed five training standard deviations."),
            ("Optimization Scientist", f"Live Python int8 inference completed; generated model costs {self.state.current['baked'].macs:,} MACs per flow."),
            ("Experimental Scientist", f"Compared score {score:.4f} with fixed A-val threshold {threshold:.4f}; {'candidate alert' if event['alert'] else 'no alert'} for this flow."),
            ("Adversarial Validator", f"Marked result {'for analyst review' if event['alert'] or outlier_count > 5 else 'within current threshold'}; packet labels are unavailable, so this is not a verified attack verdict."),
        ]
        priority = "high" if event["alert"] else "review" if outlier_count > 5 else "normal"
        results.append(("Coordinator", f"Set live triage priority to {priority}; no blocking or response action was taken."))
        with self.lock:
            for role, message in results:
                self.agent_sequence += 1
                self.agent_events.append({"id": self.agent_sequence, "time": time.time(), "role": role,
                                          "message": message, "flow_id": event["id"], "priority": priority,
                                          "mode": "local_rule_based_agent"})


def _read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def _capture_error_hint(message):
    if "Grant this user capture permission" in message:
        return message
    normalized = message.lower()
    if "bpf" in normalized or "permission denied" in normalized or "operation not permitted" in normalized:
        return (f"{message}. tcpdump cannot access the packet-capture device. Grant this user capture "
                "permission for the selected interface, then restart the dashboard; do not run the web "
                "server as root. On macOS, system or device-policy restrictions may prevent BPF access "
                "even when the account belongs to the access_bpf group.")
    return message


def _decode_packet(frame, linktype, timestamp):
    offset = 0
    ethertype = None
    if linktype == 1:
        if len(frame) < 14:
            return None
        ethertype = struct.unpack("!H", frame[12:14])[0]
        offset = 14
        while ethertype in (0x8100, 0x88a8, 0x9100):
            if len(frame) < offset + 4:
                return None
            ethertype = struct.unpack("!H", frame[offset + 2:offset + 4])[0]
            offset += 4
    elif linktype in (101, 12):
        ethertype = 0x0800 if frame and frame[0] >> 4 == 4 else 0x86dd
    elif linktype in (0, 108):
        if len(frame) < 5:
            return None
        family = struct.unpack("<I", frame[:4])[0]
        ethertype = 0x0800 if family in (2, 24, 28, 30) else 0x86dd
        offset = 4
    else:
        return None
    if ethertype == 0x0800:
        if len(frame) < offset + 20:
            return None
        version_ihl = frame[offset]
        header_length = (version_ihl & 15) * 4
        if version_ihl >> 4 != 4 or header_length < 20 or len(frame) < offset + header_length:
            return None
        proto = frame[offset + 9]
        ip_bytes = struct.unpack("!H", frame[offset + 2:offset + 4])[0]
        src = str(ipaddress.IPv4Address(frame[offset + 12:offset + 16]))
        dst = str(ipaddress.IPv4Address(frame[offset + 16:offset + 20]))
        ttl = frame[offset + 8]
        transport = offset + header_length
    elif ethertype == 0x86dd:
        if len(frame) < offset + 40 or frame[offset] >> 4 != 6:
            return None
        proto = frame[offset + 6]
        ip_bytes = 40 + struct.unpack("!H", frame[offset + 4:offset + 6])[0]
        src = str(ipaddress.IPv6Address(frame[offset + 8:offset + 24]))
        dst = str(ipaddress.IPv6Address(frame[offset + 24:offset + 40]))
        ttl = frame[offset + 7]
        transport = offset + 40
    else:
        return None
    if proto == 6:
        if len(frame) < transport + 20:
            return None
        sport, dport = struct.unpack("!HH", frame[transport:transport + 4])
        tcp_length = max(20, (frame[transport + 12] >> 4) * 4)
        flags = frame[transport + 13]
        window = struct.unpack("!H", frame[transport + 14:transport + 16])[0]
        l7 = 6
    elif proto == 17:
        if len(frame) < transport + 8:
            return None
        sport, dport = struct.unpack("!HH", frame[transport:transport + 4])
        flags = window = 0
        tcp_length = 8
        l7 = 17
    else:
        sport = dport = 0
        flags = window = 0
        tcp_length = 0
        l7 = proto
    return {"time": timestamp, "src": src, "dst": dst, "sport": sport, "dport": dport,
            "protocol": proto, "l7": l7, "ttl": ttl, "flags": flags, "window": window,
            "packet_length": len(frame), "ip_bytes": ip_bytes, "transport_length": tcp_length}


def _flow_vector(flow, features):
    duration = max(0.0, flow["last"] - flow["first"])
    forward_packets, reverse_packets = flow["forward_packets"], flow["reverse_packets"]
    forward_bytes, reverse_bytes = flow["forward_bytes"], flow["reverse_bytes"]
    lengths = flow["lengths"]
    values = {"IPV4_SRC_ADDR": _ip_model_value(flow["src"]), "IPV4_DST_ADDR": _ip_model_value(flow["dst"]),
              "L4_SRC_PORT": flow["sport"], "L4_DST_PORT": flow["dport"], "PROTOCOL": flow["protocol"],
              "L7_PROTO": 0, "IN_BYTES": forward_bytes, "OUT_BYTES": reverse_bytes,
              "IN_PKTS": forward_packets, "OUT_PKTS": reverse_packets,
              "FLOW_DURATION_MILLISECONDS": duration * 1000, "TCP_FLAGS": flow["flags"],
              "CLIENT_TCP_FLAGS": flow["flags"], "SERVER_TCP_FLAGS": 0,
              "DURATION_IN": duration * 1000, "DURATION_OUT": duration * 1000,
              "MIN_TTL": min(flow["ttl"]), "MAX_TTL": max(flow["ttl"]),
              "LONGEST_FLOW_PKT": max(lengths), "SHORTEST_FLOW_PKT": min(lengths),
              "MIN_IP_PKT_LEN": min(lengths), "MAX_IP_PKT_LEN": max(lengths),
              "SRC_TO_DST_SECOND_BYTES": forward_bytes / max(duration, 1e-6),
              "DST_TO_SRC_SECOND_BYTES": reverse_bytes / max(duration, 1e-6),
              "SRC_TO_DST_AVG_THROUGHPUT": forward_bytes * 8 / max(duration, 1e-6),
              "DST_TO_SRC_AVG_THROUGHPUT": reverse_bytes * 8 / max(duration, 1e-6),
              "TCP_WIN_MAX_IN": flow["tcp_window"], "TCP_WIN_MAX_OUT": flow["tcp_window"]}
    buckets = ((128, "NUM_PKTS_UP_TO_128_BYTES"), (256, "NUM_PKTS_128_TO_256_BYTES"),
               (512, "NUM_PKTS_256_TO_512_BYTES"), (1024, "NUM_PKTS_512_TO_1024_BYTES"),
               (1514, "NUM_PKTS_1024_TO_1514_BYTES"))
    for size, name in buckets:
        lower = {"NUM_PKTS_UP_TO_128_BYTES": 0, "NUM_PKTS_128_TO_256_BYTES": 128,
                 "NUM_PKTS_256_TO_512_BYTES": 256, "NUM_PKTS_512_TO_1024_BYTES": 512,
                 "NUM_PKTS_1024_TO_1514_BYTES": 1024}[name]
        values[name] = sum(lower < length <= size for length in lengths)
    for name in features:
        if name not in values:
            values[name] = 0
    return np.asarray([values[name] for name in features], dtype=float)


def _ip_model_value(address):
    parsed = ipaddress.ip_address(address)
    return int(parsed) if parsed.version == 4 else 0
