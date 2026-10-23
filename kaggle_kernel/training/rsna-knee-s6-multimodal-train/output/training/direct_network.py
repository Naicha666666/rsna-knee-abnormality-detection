"""Windows downloader transport: physical-interface sockets, no proxy fallback.

DNS is also sent through the chosen interface. No system-wide routes or VPN
settings are changed. TLS certificate verification stays enabled.
"""
from __future__ import annotations

import ipaddress
import os
import socket
import struct
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool


def _dns_name(data, position):
    names, next_position, seen = [], None, set()
    while True:
        if position in seen or position >= len(data):
            raise ValueError("invalid DNS response")
        seen.add(position)
        size = data[position]
        if size & 0xc0 == 0xc0:
            if next_position is None:
                next_position = position + 2
            position = ((size & 63) << 8) | data[position + 1]
        elif size == 0:
            return ".".join(names), next_position or position + 1
        else:
            names.append(data[position + 1:position + 1 + size].decode("ascii"))
            position += size + 1


class DirectNetwork:
    def __init__(self, source_ip, interface_index, dns_servers=("223.5.5.5", "119.29.29.29")):
        if os.name != "nt":
            raise RuntimeError("this direct-interface transport is Windows-specific")
        self.source_ip, self.interface_index = source_ip, int(interface_index)
        self.dns_servers = tuple(dns_servers)
        self.cache, self.connections = {}, []
        self.lock = threading.Lock()

    def socket(self, kind):
        s = socket.socket(socket.AF_INET, kind)
        try:
            s.setsockopt(socket.IPPROTO_IP, 31, struct.pack("!I", self.interface_index))  # IP_UNICAST_IF
            # Windows sets this DWORD in network order, but returns the index
            # in host order from getsockopt.
            actual = struct.unpack("I", s.getsockopt(socket.IPPROTO_IP, 31, 4))[0]
            if actual != self.interface_index:
                raise RuntimeError("physical interface binding failed")
            s.bind((self.source_ip, 0))
            return s
        except Exception:
            s.close()
            raise

    def resolve(self, host, depth=0):
        try:
            return [str(ipaddress.IPv4Address(host))]
        except ValueError:
            pass
        if depth > 5:
            raise RuntimeError("DNS CNAME chain too long")
        with self.lock:
            cached = self.cache.get(host)
        if cached and time.monotonic() < cached[0]:
            return cached[1]
        ident = os.urandom(2)
        query = ident + b"\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
        query += b"".join(bytes([len(x)]) + x.encode("ascii") for x in host.split(".")) + b"\0\x00\x01\x00\x01"
        for server in self.dns_servers:
            s = self.socket(socket.SOCK_DGRAM)
            try:
                s.settimeout(4)
                s.sendto(query, (server, 53))
                response, remote = s.recvfrom(65535)
                if response[:2] != ident or remote[0] != server:
                    continue
                _, flags, qd, answers, _, _ = struct.unpack("!6H", response[:12])
                if flags & 15:
                    continue
                pos = 12
                for _ in range(qd):
                    _, pos = _dns_name(response, pos); pos += 4
                found, cnames = [], []
                for _ in range(answers):
                    _, pos = _dns_name(response, pos)
                    kind, cls, ttl, size = struct.unpack("!HHIH", response[pos:pos+10]); pos += 10
                    if kind == 1 and size == 4:
                        address = socket.inet_ntoa(response[pos:pos+size])
                        if not ipaddress.ip_address(address).is_global:
                            raise RuntimeError("DNS returned a private/fake IP; refusing VPN-routed DNS")
                        found.append(address)
                    elif kind == 5:
                        cname, _ = _dns_name(response, pos); cnames.append(cname)
                    pos += size
                if not found and cnames:
                    found = self.resolve(cnames[-1], depth + 1)
                if found:
                    with self.lock:
                        self.cache[host] = (time.monotonic() + 300, found)
                    return found
            except (OSError, ValueError):
                pass
            finally:
                s.close()
        raise RuntimeError(f"direct DNS failed for {host}; no proxy fallback")

    def connect(self, host, port, timeout):
        ips = self.resolve(host)
        last = None
        for ip in ips:
            s = self.socket(socket.SOCK_STREAM)
            try:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(float(timeout) if isinstance(timeout, (float, int)) else 15.)
                s.connect((ip, port))
                if s.getsockname()[0] != self.source_ip:
                    raise RuntimeError("wrong source interface")
                with self.lock:
                    self.connections.append({"time": time.time(), "host": host, "remote_ip": ip,
                                             "port": port, "local_ip": s.getsockname()[0],
                                             "local_port": s.getsockname()[1], "interface_index": self.interface_index})
                return s
            except Exception as exc:
                last = exc; s.close()
        raise OSError(f"direct connection failed for {host}: {type(last).__name__}; no proxy fallback")

    def session(self):
        network = self
        class DirectHTTP(HTTPConnection):
            def _new_conn(self):
                return network.connect(self._dns_host, self.port, self.timeout)
        class DirectHTTPS(HTTPSConnection):
            def _new_conn(self):
                return network.connect(self._dns_host, self.port, self.timeout)
        class HTTPPool(HTTPConnectionPool):
            ConnectionCls = DirectHTTP
        class HTTPSPool(HTTPSConnectionPool):
            ConnectionCls = DirectHTTPS
        class Adapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                super().init_poolmanager(*args, **kwargs)
                self.poolmanager.pool_classes_by_scheme = {"http": HTTPPool, "https": HTTPSPool}
            def proxy_manager_for(self, *args, **kwargs):
                raise RuntimeError("proxy use forbidden for this download")
        class Session(requests.sessions.Session):
            def __init__(self):
                super().__init__()
                self.trust_env = False
                self.proxies.clear()
                self.mount("http://", Adapter())
                self.mount("https://", Adapter())
            def send(self, request, **kwargs):
                if kwargs.get("proxies") or kwargs.get("verify") is False:
                    raise RuntimeError("proxy or disabled TLS verification forbidden")
                kwargs["proxies"] = {}
                if kwargs.get('timeout') is None:kwargs['timeout']=(15,45)
                return super().send(request, **kwargs)
        return Session()

    def receipt(self):
        with self.lock:
            return {"mode": "physical_ethernet_IP_UNICAST_IF", "proxy_fallback": False,
                    "source_ip": self.source_ip, "interface_index": self.interface_index,
                    "dns_servers": self.dns_servers, "connections": list(self.connections)}


if __name__ == "__main__":
    import json
    net = DirectNetwork("192.168.1.31", 11)
    for host in ("www.kaggle.com", "storage.googleapis.com"):
        try:
            ips = net.resolve(host)
            r = net.session().get("https://" + host + "/", timeout=(10, 15), stream=True)
            print(json.dumps({"host": host, "ips": ips, "status": r.status_code}), flush=True)
            r.close()
        except Exception as exc:
            print(json.dumps({"host": host, "error_type": type(exc).__name__}), flush=True)
    print(json.dumps(net.receipt()), flush=True)
