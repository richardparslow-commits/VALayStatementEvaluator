"""Dedicated parser engine identity and private mutual-TLS client configuration.

Credentials authorize the dedicated parser VM, never the application daemon.
Host separation and absence of secrets require independent host acceptance;
an engine ID/label alone cannot prove either property.
"""
from __future__ import annotations

import ipaddress
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from .parser_protocol import ParserRefused

TLS_DIRECTORY = Path("/run/parser-engine/tls")
_PRIVATE_NETWORKS = tuple(ipaddress.ip_network(value) for value in
                          ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))


def engine_identity(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:-]{0,127}", value):
        raise ParserRefused("A reviewed parser engine identity is required.")
    return value


def parser_engine_id() -> str:
    return engine_identity(os.environ.get("VA_LSE_PARSER_ENGINE_ID", ""))


@dataclass(frozen=True)
class ParserEngine:
    endpoint: str
    identity: str
    application_identity: str
    tls_directory: Path = TLS_DIRECTORY

    def __post_init__(self) -> None:
        # Literal RFC1918 addresses avoid DNS rebinding, public destinations,
        # local sockets and Docker contexts. No optional plaintext fallback.
        match = re.fullmatch(r"tcp://([0-9.]+):2376", self.endpoint)
        try:
            address = ipaddress.IPv4Address(match[1] if match else "")
        except ipaddress.AddressValueError as exc:
            raise ParserRefused("A private dedicated parser TLS endpoint is required.") from exc
        if not any(address in network for network in _PRIVATE_NETWORKS):
            raise ParserRefused("A private dedicated parser TLS endpoint is required.")
        engine_identity(self.identity)
        engine_identity(self.application_identity)
        if self.identity == self.application_identity:
            raise ParserRefused("The parser must use a different engine from the application.")
        if not self.tls_directory.is_absolute():
            raise ParserRefused("Parser TLS credentials require an absolute private directory.")

    @classmethod
    def from_environment(cls) -> ParserEngine:
        return cls(os.environ.get("VA_LSE_PARSER_ENGINE_ENDPOINT", ""), parser_engine_id(),
                   os.environ.get("VA_LSE_APPLICATION_ENGINE_ID", ""))

    def verify_credentials(self) -> None:
        # Do not print, copy, or load key contents into Python. Docker validates
        # both server and client certificates; the server cert needs the IP SAN.
        try:
            directory = self.tls_directory.lstat()
            if (not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.geteuid()
                    or directory.st_mode & 0o077):
                raise ParserRefused("Parser TLS credentials require a private owned directory.")
        except OSError as exc:
            raise ParserRefused("Parser TLS credentials require a private owned directory.") from exc
        for name in ("ca.pem", "cert.pem", "key.pem"):
            path = self.tls_directory / name
            try:
                info = path.lstat()
                if (not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 128 * 1024
                        or info.st_mode & 0o022 or not os.access(path, os.R_OK)
                        or (name == "key.pem" and (info.st_uid != os.geteuid() or info.st_mode & 0o077))):
                    raise ParserRefused("Parser TLS credentials are absent or unsafe.")
            except OSError as exc:
                raise ParserRefused("Parser TLS credentials are absent or unsafe.") from exc

    def environment(self) -> dict[str, str]:
        return {"PATH": os.defpath, "HOME": "/nonexistent", "DOCKER_HOST": self.endpoint,
                "DOCKER_TLS_VERIFY": "1", "DOCKER_CERT_PATH": str(self.tls_directory)}
