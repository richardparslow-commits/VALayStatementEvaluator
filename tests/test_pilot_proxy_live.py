"""Opt-in Linux/Docker probes of the actual pilot proxy, using synthetic bodies."""
from __future__ import annotations

import http.client
import http.server
import json
import os
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path

from tests import hermetic  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
IMAGE = os.environ.get("VA_LSE_TEST_PILOT_PROXY_IMAGE", "")
CANARY = b"SYNTHETIC_ONLY_PROXY_PRIVACY_CANARY_92714"


@unittest.skipUnless(IMAGE, "Opt-in Linux Docker pilot proxy image is unset.")
class PilotProxyTests(unittest.TestCase):
    @classmethod
    def run_cli(cls, *args, check=True):
        return subprocess.run(list(args), text=True, capture_output=True, timeout=30, check=check)

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.work = Path(cls.directory.name)
        cert, key = cls.work/"cert.pem", cls.work/"key.pem"
        cls.run_cli("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-keyout", str(key), "-out", str(cert), "-subj", "/CN=synthetic.invalid")
        # Only a throwaway fixture key. Actual-host keys must be private to UID 10001.
        key.chmod(0o644)
        cls.requests = []
        class Backend(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *args):
                pass
            def do_POST(self):
                if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                    body = bytearray()
                    while True:
                        size = int(self.rfile.readline().split(b";")[0], 16)
                        if size == 0:
                            self.rfile.readline(); break
                        body.extend(self.rfile.read(size)); self.rfile.read(2)
                else:
                    body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                cls.requests.append(bytes(body))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.send_header("X-Accel-Buffering", "yes")
                self.end_headers()
                try:
                    self.wfile.write(b"OK")
                except OSError:
                    pass
            def do_GET(self):
                self.send_response(200); self.send_header("Content-Length", "2")
                self.end_headers(); self.wfile.write(b"OK")
        cls.backend = http.server.ThreadingHTTPServer(("0.0.0.0", 0), Backend)
        cls.backend.daemon_threads = True
        cls.thread = threading.Thread(target=cls.backend.serve_forever, daemon=True)
        cls.thread.start()
        cls.addClassCleanup(cls.backend.server_close)
        cls.addClassCleanup(cls.backend.shutdown)
        configuration = (ROOT/"nginx/pilot.conf").read_text().replace("streamlit-web:8501", f"streamlit-web:{cls.backend.server_port}")
        cls.configuration = cls.work/"nginx.conf"; cls.configuration.write_text(configuration)
        empty = cls.work/"empty.env"; empty.write_text("")
        environment = {**os.environ, "VA_LSE_BUILD_SHA": "0"*40,
            "VA_LSE_PILOT_ENV_FILE": str(empty), "VA_LSE_PILOT_APPROVAL_HOST_FILE": str(empty),
            "VA_LSE_OIDC_SECRETS_FILE": str(empty), "VA_LSE_PARSER_IMAGE": "sha256:"+"0"*64,
            "VA_LSE_PILOT_LOG_RETENTION_DAYS": "1", "VA_LSE_DOCKER_GID": "0",
            "VA_LSE_BIND_IP": "127.0.0.1", "VA_LSE_TLS_CERT": str(cert), "VA_LSE_TLS_KEY": str(key)}
        result = subprocess.run(["docker", "compose", "-f", str(ROOT/"docker-compose.pilot.yml"), "config", "--format", "json"],
                                env=environment, text=True, capture_output=True, timeout=30, check=True)
        cls.policy = json.loads(result.stdout)["services"]["nginx"]

    def setUp(self):
        self.name = "va-lse-proxy-test-"+uuid.uuid4().hex[:12]
        self.addCleanup(lambda: self.run_cli("docker", "rm", "-f", self.name, check=False))
        p = self.policy
        args = ["docker", "run", "-d", "--name", self.name, "--add-host", "streamlit-web:host-gateway",
                "-p", "127.0.0.1::8443", "--memory", str(p["mem_limit"]), "--memory-swap", str(p["memswap_limit"]),
                "--pids-limit", str(p["pids_limit"]), "--cpus", str(p["cpus"]), "--log-driver", p["logging"]["driver"]]
        if p["read_only"]: args.append("--read-only")
        for capability in p["cap_drop"]: args.extend(["--cap-drop", capability])
        for option in p["security_opt"]: args.extend(["--security-opt", option])
        for mount in p["tmpfs"]: args.extend(["--tmpfs", mount])
        for name, limits in p["ulimits"].items():
            # Compose JSON can omit zero values, or retain the scalar as single.
            soft = limits.get("soft", limits.get("single", 0))
            hard = limits.get("hard", limits.get("single", 0))
            args.extend(["--ulimit", f"{name}={soft}:{hard}"])
        for source, target in ((self.configuration, "/etc/nginx/nginx.conf"), (self.work/"cert.pem", "/run/tls/cert.pem"), (self.work/"key.pem", "/run/tls/key.pem")):
            args.extend(["--mount", f"type=bind,source={source},target={target},readonly"])
        args.append(IMAGE)
        self.run_cli(*args)
        self.wait_ready()

    def wait_ready(self):
        mapping = self.run_cli("docker", "port", self.name, "8443/tcp").stdout.strip()
        self.port = int(mapping.rsplit(":", 1)[1])
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            try:
                connection=self.connect(); connection.request("GET", "/")
                response=connection.getresponse(); response.read(); connection.close()
                if response.status == 200: return
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(.1)
        self.fail("Synthetic proxy did not become ready.")

    def connect(self):
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT); context.check_hostname=False; context.verify_mode=ssl.CERT_NONE
        return http.client.HTTPSConnection("127.0.0.1", self.port, context=context, timeout=10)

    def test_nonroot_readonly_private_tmpfs_swap_and_core_limits(self):
        inspection=json.loads(self.run_cli("docker", "inspect", self.name).stdout)[0]
        host=inspection["HostConfig"]
        self.assertTrue(host["ReadonlyRootfs"])
        self.assertEqual(inspection["Config"]["User"], "10001:10001")
        self.assertEqual(host["Memory"], host["MemorySwap"])
        self.assertEqual(host["LogConfig"]["Type"], "none")
        self.assertIn("ALL", host["CapDrop"])
        for mount in ("/var/cache/nginx", "/run/nginx"):
            self.assertIn("mode=0700", host["Tmpfs"][mount])
        result=self.run_cli("docker", "exec", self.name, "sh", "-c", "id -u; stat -c '%a:%u:%g' /var/cache/nginx /run/nginx; ulimit -c; cat /proc/1/status; cat /proc/mounts").stdout
        self.assertIn("10001\n700:10001:10001\n700:10001:10001\n0\n", result)
        self.assertIn("CapEff:\t0000000000000000", result)
        self.assertRegex(result, r"tmpfs /var/cache/nginx tmpfs .*nosuid.*nodev.*noexec")
        swap=self.run_cli("docker", "exec", self.name, "sh", "-c",
            "if [ -f /sys/fs/cgroup/memory.swap.max ]; then test \"$(cat /sys/fs/cgroup/memory.swap.max)\" = 0; "
            "else test \"$(cat /sys/fs/cgroup/memory/memory.memsw.limit_in_bytes)\" = \"$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)\"; fi",
            check=False)
        self.assertEqual(swap.returncode, 0, "Kernel cgroup does not enforce the requested no-swap policy.")
        denied=self.run_cli("docker", "exec", self.name, "sh", "-c", "touch /persistent-canary", check=False)
        self.assertNotEqual(denied.returncode, 0)

    def test_content_length_and_chunked_uploads_leave_no_persistent_body(self):
        body=CANARY+b"x"*(2*1024*1024)
        for chunked in (False, True):
            connection=self.connect()
            connection.request("POST", "/upload?synthetic=1", body=[body[:100], body[100:]] if chunked else body, encode_chunked=chunked)
            response=connection.getresponse(); response.read(); connection.close()
            self.assertEqual(response.status, 200)
            self.assertIn(body, self.requests)
        self.assertEqual(self.run_cli("docker", "diff", self.name).stdout.strip(), "")
        files=self.run_cli("docker", "exec", self.name, "find", "/var/cache/nginx", "-type", "f").stdout.strip()
        self.assertEqual(files, "")

    def test_rejected_and_aborted_uploads_do_not_survive_or_log_canaries(self):
        connection=self.connect(); connection.putrequest("POST", "/upload?"+CANARY.decode())
        connection.putheader("Content-Length", str(52*1024*1024)); connection.endheaders()
        response=connection.getresponse(); content=response.read(); connection.close()
        self.assertEqual(response.status, 413); self.assertNotIn(CANARY, content)
        connection=self.connect(); connection.putrequest("POST", "/upload")
        connection.putheader("Content-Length", "2000000"); connection.endheaders(); connection.send(CANARY)
        connection.sock.shutdown(socket.SHUT_RDWR); connection.close()
        time.sleep(.3)
        self.assertEqual(self.run_cli("docker", "diff", self.name).stdout.strip(), "")
        files=self.run_cli("docker", "exec", self.name, "find", "/var/cache/nginx", "-type", "f").stdout.strip()
        self.assertEqual(files, "")
        active=self.run_cli("docker", "exec", self.name, "nginx", "-T").stdout
        self.assertIn("access_log off;", active)
        self.assertIn("error_log /dev/null crit;", active)

    def test_kill_and_restart_remove_volatile_copies(self):
        self.run_cli("docker", "exec", self.name, "sh", "-c", "echo SYNTHETIC_ONLY > /var/cache/nginx/crash-canary")
        self.run_cli("docker", "kill", "--signal", "KILL", self.name)
        self.run_cli("docker", "start", self.name); self.wait_ready()
        absent=self.run_cli("docker", "exec", self.name, "test", "!", "-e", "/var/cache/nginx/crash-canary", check=False)
        self.assertEqual(absent.returncode, 0)
        self.assertEqual(self.run_cli("docker", "diff", self.name).stdout.strip(), "")

    def test_no_store_response_headers_and_private_operational_routes(self):
        for route, expected in (("/", 200), ("/health", 404), ("/ready", 404), ("/metrics", 404)):
            connection=self.connect(); connection.request("GET", route)
            response=connection.getresponse(); response.read(); connection.close()
            self.assertEqual(response.status, expected)
            self.assertEqual(response.getheader("Cache-Control"), "no-store")
            self.assertEqual(response.getheader("Referrer-Policy"), "no-referrer")


if __name__ == "__main__":
    unittest.main()
