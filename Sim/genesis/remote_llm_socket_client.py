import socket
import struct
import json
import numpy as np
from typing import Any, Dict, Optional


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and dtypes."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class RemoteLLMSocketClient:
    """4-byte big-endian length-prefix JSON persistent connection client for Genesis."""

    def __init__(
        self,
        host: str,
        port: int,
        connect_timeout: float = 120.0,
        recv_timeout: Optional[float] = 600.0,  # 10 minutes for LLM generation
        tcp_nodelay: bool = True,
    ):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.recv_timeout = recv_timeout
        self.tcp_nodelay = tcp_nodelay
        self._sock: Optional[socket.socket] = None

    def connect(self) -> None:
        if self._sock is not None:
            return
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
            print(f"[DEBUG] TCP connection established to {self.host}:{self.port}")
        except socket.timeout:
            raise TimeoutError(f"Connection to {self.host}:{self.port} timed out after {self.connect_timeout}s. Is the server running?")
        except ConnectionRefusedError:
            raise ConnectionRefusedError(f"Connection to {self.host}:{self.port} refused. Is the server running on port {self.port}?")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}")

        if self.tcp_nodelay:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if self.recv_timeout is not None:
            s.settimeout(self.recv_timeout)
        self._sock = s

        # handshake
        self.send({"type": "hello", "client": "genesis_loop"})
        resp = self.recv()
        if not isinstance(resp, dict) or resp.get("type") != "hello_ack":
            raise RuntimeError(f"Handshake failed: {resp}")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None

    def send(self, obj: Dict[str, Any]) -> None:
        if self._sock is None:
            raise RuntimeError("Socket is not connected.")
        data = json.dumps(obj, ensure_ascii=False, cls=NumpyEncoder).encode("utf-8")
        header = struct.pack(">I", len(data))
        self._sock.sendall(header + data)

    def recv(self) -> Dict[str, Any]:
        if self._sock is None:
            raise RuntimeError("Socket is not connected.")
        header = self._recvall(4)
        (length,) = struct.unpack(">I", header)
        payload = self._recvall(length)
        return json.loads(payload.decode("utf-8"))

    def request(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        self.send(obj)
        return self.recv()

    def _recvall(self, n: int) -> bytes:
        assert self._sock is not None
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Socket closed by peer.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
