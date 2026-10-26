"""Ephemeral synthetic-only second Docker daemon with mutual TLS for Linux CI.

Both daemons share a disposable CI host. This verifies transport/daemon identity,
not the required production separation into a dedicated parser host or VM.
No provider credentials, deployment credentials or veteran information are used.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import ssl
import subprocess
import time
from pathlib import Path


def call(args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, timeout=120, **kwargs)


def wait_for_distinct_engine(env, application_id, *, timeout=60, clock=time.monotonic,
                             sleep=time.sleep, run=subprocess.run):
    """Return the dedicated daemon's engine id, once it differs from the app's.

    A zero exit from ``docker info`` is not enough to trust a reply: while the
    second daemon is still coming up, the CLI can answer for the application's
    daemon (or fall back to a default socket), and that identity must never be
    recorded as the parser engine's. Both an unreachable endpoint and a reply
    that still names the application engine are retried until ``timeout``, then
    reported with both identities instead of asserting on the first answer.
    """
    deadline = clock() + timeout
    attempts = 0
    reported = set()
    last_reply = ''
    last_error = ''
    while clock() < deadline:
        attempts += 1
        try:
            result = run(['docker', 'info', '--format', '{{.ID}}'], env=env,
                         capture_output=True, timeout=10)
        except subprocess.TimeoutExpired:
            result = None
            last_error = 'docker info timed out'
        if result is not None and result.returncode == 0:
            candidate = result.stdout.decode(errors='replace').strip()
            if candidate and candidate != application_id:
                return candidate
            last_reply = candidate
            if candidate not in reported:
                reported.add(candidate)
                print(f'Synthetic parser daemon is not distinct yet (attempt {attempts}): '
                      f'{candidate or "<empty engine id>"} is the application engine.')
        elif result is not None:
            last_error = result.stderr.decode(errors='replace').strip()[-500:]
        sleep(.5)
    details = (f'after {attempts} attempt(s) in {timeout}s on {env.get("DOCKER_HOST")}; '
               f'application engine id {application_id!r}, last reply {last_reply or "<none>"!r}')
    if last_error:
        details += f', last error {last_error!r}'
    raise RuntimeError(
        'Synthetic dedicated TLS engine did not become a distinct engine ' + details
        + '; inspect the private CI daemon log above.'
    )


def _report_startup_failure(directory, tls, address):
    """Print bounded diagnostics for a dedicated daemon that never came up.

    Daemon setup uses only synthetic credentials/data. These tails contain no key
    contents; the reason is exposed instead of leaving a bare assertion.
    """
    print((directory / 'daemon.log').read_text(errors='replace')[-8000:])
    print((directory / 'containerd.log').read_text(errors='replace')[-3000:])
    context = ssl.create_default_context(cafile=str(tls / 'ca.pem'))
    context.load_cert_chain(str(tls / 'cert.pem'), str(tls / 'key.pem'))
    connection = http.client.HTTPSConnection(address, 2376, timeout=5, context=context)
    try:
        connection.request('GET', '/_ping')
        reply = connection.getresponse()
        print('Synthetic direct verified TLS ping:', reply.status, reply.read(64).decode(errors='replace'))
    except (OSError, ssl.SSLError) as exc:
        print('Synthetic direct verified TLS ping refused:', type(exc).__name__, str(exc))
    finally:
        connection.close()


def setup(directory):
    directory.mkdir(mode=0o700)
    tls = directory / 'client'
    server = directory / 'server'
    tls.mkdir(mode=0o700)
    server.mkdir(mode=0o700)
    networks = json.loads(call(['docker', 'network', 'inspect', 'bridge']).stdout)
    address = '10.73.0.1'
    application_id = call(['docker', 'info', '--format', '{{.ID}}']).stdout.decode().strip()
    # Do not borrow docker0's gateway as a listener address. Provision an
    # explicit private test endpoint and only its TLS port on this disposable VM.
    call(['sudo', 'ip', 'address', 'add', address + '/32', 'dev', 'lo'])
    for chain in ('INPUT', 'OUTPUT'):
        call(['sudo', 'iptables', '-I', chain, '-d', address + '/32', '-p', 'tcp', '--dport', '2376', '-j', 'ACCEPT'])
    # A second daemon's bridge=none setup can remove the application's docker0.
    # Give this daemon its own bridge instead; parser containers still use none.
    call(['sudo', 'ip', 'link', 'add', 'va-parser-ci', 'type', 'bridge'])
    call(['sudo', 'ip', 'address', 'add', '10.74.0.1/24', 'dev', 'va-parser-ci'])
    call(['sudo', 'ip', 'link', 'set', 'va-parser-ci', 'up'])
    call(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
          '-subj', '/CN=SyntheticParserCI', '-keyout', str(server / 'ca-key.pem'), '-out', str(server / 'ca.pem')])
    for name, common, extensions in (('server', 'SyntheticParserServer', f'subjectAltName=IP:{address}\nextendedKeyUsage=serverAuth\n'),
                                     ('client', 'SyntheticParserClient', 'extendedKeyUsage=clientAuth\n')):
        (server / f'{name}.ext').write_text(extensions)
        call(['openssl', 'req', '-newkey', 'rsa:2048', '-nodes', '-subj', f'/CN={common}',
              '-keyout', str(server / f'{name}-key.pem'), '-out', str(server / f'{name}.csr')])
        call(['openssl', 'x509', '-req', '-days', '1', '-in', str(server / f'{name}.csr'),
              '-CA', str(server / 'ca.pem'), '-CAkey', str(server / 'ca-key.pem'), '-CAcreateserial',
              '-extfile', str(server / f'{name}.ext'), '-out', str(server / f'{name}.pem')])
    for source, target in (('ca.pem', 'ca.pem'), ('client.pem', 'cert.pem'), ('client-key.pem', 'key.pem')):
        shutil.copyfile(server / source, tls / target)
        (tls / target).chmod(0o400)
    launcher_tls = directory / 'launcher-client'
    shutil.copytree(tls, launcher_tls)
    call(['sudo', 'chown', '-R', '65534:65534', str(launcher_tls)])
    env = {'PATH': os.environ['PATH'], 'HOME': '/nonexistent', 'DOCKER_HOST': f'tcp://{address}:2376',
           'DOCKER_TLS_VERIFY': '1', 'DOCKER_CERT_PATH': str(tls)}
    # Do not inherit the runner's daemon.json or its general containerd socket.
    configuration = directory / 'daemon.json'
    configuration.write_text('{}\n')
    with (directory / 'containerd.log').open('wb') as log:
        containerd = subprocess.Popen(['sudo', 'containerd', '--root=' + str(directory / 'containerd-data'),
                                      '--state=' + str(directory / 'containerd-state'),
                                      '--address=' + str(directory / 'containerd.sock')],
                                     stdout=log, stderr=log, env={'PATH': os.environ['PATH']}, start_new_session=True)
        (directory / 'containerd-process.pid').write_text(str(containerd.pid))
    with (directory / 'daemon.log').open('wb') as log:
        subprocess.Popen(['sudo', 'dockerd', '--config-file=' + str(configuration),
                          '--containerd=' + str(directory / 'containerd.sock'),
                          '--containerd-namespace=va-lse-parser-ci', '--containerd-plugins-namespace=va-lse-parser-ci-plugins',
                          '--host=' + env['DOCKER_HOST'], '--tlsverify',
                          '--tlscacert=' + str(server / 'ca.pem'), '--tlscert=' + str(server / 'server.pem'),
                          '--tlskey=' + str(server / 'server-key.pem'), '--data-root=' + str(directory / 'data'),
                          '--exec-root=' + str(directory / 'exec'), '--pidfile=' + str(directory / 'daemon.pid'),
                          '--bridge=va-parser-ci', '--iptables=false', '--ip-masq=false', '--ip-forward=false',
                          '--label=va-lse-purpose=parser-only'], stdout=log, stderr=log,
                         env={'PATH': os.environ['PATH']}, start_new_session=True)
    try:
        engine_id = wait_for_distinct_engine(env, application_id)
    except RuntimeError:
        _report_startup_failure(directory, tls, address)
        raise
    # Fail setup if the independent application bridge was changed/removed.
    call(['ip', 'link', 'show', networks[0]['Options'].get('com.docker.network.bridge.name', 'docker0')])
    image = call(['docker', 'image', 'inspect', 'va-lse-parser:ci', '--format', '{{.Id}}']).stdout.decode().strip()
    archive = directory / 'parser-image.tar'
    call(['docker', 'save', '-o', str(archive), image])
    call(['docker', 'load', '-i', str(archive)], env=env)
    archive.unlink()
    values = {'VA_LSE_TEST_PARSER_ENGINE_ENDPOINT': env['DOCKER_HOST'],
              'VA_LSE_TEST_PARSER_ENGINE_ID': engine_id, 'VA_LSE_TEST_APPLICATION_ENGINE_ID': application_id,
              'VA_LSE_TEST_PARSER_ENGINE_TLS_DIRECTORY': str(tls),
              'VA_LSE_TEST_PARSER_LAUNCHER_TLS_DIRECTORY': str(launcher_tls)}
    with Path(os.environ['GITHUB_ENV']).open('a') as output:
        for key, value in values.items():
            output.write(f'{key}={value}\n')
    print('Synthetic mutual-TLS parser daemon ready; engine identity differs from application daemon.')


def cleanup(directory):
    pid = directory / 'daemon.pid'
    if pid.exists():
        value = pid.read_text().strip()
        if not value.isdecimal():
            raise RuntimeError('Invalid synthetic CI daemon PID.')
        subprocess.run(['sudo', 'kill', value], check=False, timeout=10, capture_output=True)
    containerd = directory / 'containerd-process.pid'
    if containerd.exists():
        value = containerd.read_text().strip()
        if not value.isdecimal():
            raise RuntimeError('Invalid synthetic CI containerd PID.')
        subprocess.run(['sudo', 'kill', '--', '-' + value], check=False, timeout=10, capture_output=True)
    # Disposable runner destroys data-root. Remove private client/server keys now.
    for name in ('client', 'server', 'launcher-client'):
        path = directory / name
        if path.exists():
            call(['sudo', 'rm', '-rf', '--', str(path)])
    for chain in ('INPUT', 'OUTPUT'):
        subprocess.run(['sudo', 'iptables', '-D', chain, '-d', '10.73.0.1/32', '-p', 'tcp',
                        '--dport', '2376', '-j', 'ACCEPT'], capture_output=True, timeout=10, check=False)
    subprocess.run(['sudo', 'ip', 'address', 'del', '10.73.0.1/32', 'dev', 'lo'],
                   capture_output=True, timeout=10, check=False)
    subprocess.run(['sudo', 'ip', 'link', 'delete', 'va-parser-ci'], capture_output=True, timeout=10, check=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleanup', action='store_true')
    args = parser.parse_args()
    directory = (Path(os.environ['RUNNER_TEMP']) / 'va-lse-parser-engine-ci').resolve()
    cleanup(directory) if args.cleanup else setup(directory)
