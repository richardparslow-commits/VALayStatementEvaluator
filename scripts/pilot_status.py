#!/usr/bin/env python3
"""Private host diagnostics: fixed service state/counts, never raw logs or env."""
from __future__ import annotations

import json
import re
import subprocess
from typing import Any

SERVICES = ('streamlit-web', 'parser-launcher', 'nginx')
_FORMAT = ('{"status":{{json .State.Status}},"exit_code":{{json .State.ExitCode}},'
           '"oom_killed":{{json .State.OOMKilled}},"restarts":{{json .RestartCount}},'
           '"health":{{if .State.Health}}{{json .State.Health.Status}}{{else}}"unavailable"{{end}}}')


def _read(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=True)
    if len(result.stdout) > 4096:
        raise ValueError('Service metadata exceeded its bound.')
    return result.stdout.strip()


def status() -> dict[str, Any]:
    report: dict[str, Any] = {}
    for service in SERVICES:
        try:
            identifier = _read(['docker', 'compose', '-f', 'docker-compose.pilot.yml',
                                'ps', '-a', '-q', service])
            if not identifier:
                report[service] = {'status': 'absent'}
                continue
            if not re.fullmatch(r'[a-f0-9]{12,64}', identifier):
                raise ValueError('Only one container per pilot service is supported.')
            data = json.loads(_read(['docker', 'inspect', '--format', _FORMAT, identifier]))
            if (not isinstance(data, dict) or set(data) != {'status', 'exit_code', 'oom_killed', 'restarts', 'health'}
                    or data['status'] not in {'created', 'running', 'paused', 'restarting', 'removing', 'exited', 'dead'}
                    or data['health'] not in {'starting', 'healthy', 'unhealthy', 'unavailable'}
                    or type(data['exit_code']) is not int or not 0 <= data['exit_code'] <= 255
                    or type(data['restarts']) is not int or not 0 <= data['restarts'] <= 1_000_000_000
                    or type(data['oom_killed']) is not bool):
                raise ValueError('Unexpected service metadata.')
            report[service] = data
        except (OSError, ValueError, TypeError, subprocess.SubprocessError):
            report[service] = {'status': 'unavailable'}  # No daemon/CLI exception text.
    return report


def main() -> int:
    report = status()
    print(json.dumps(report))
    return 2 if any(v['status'] != 'running' or v.get('health') == 'unhealthy'
                    or (service != 'nginx' and v.get('health') != 'healthy')
                    for service, v in report.items()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
