"""Ephemeral synthetic-only second Docker daemon with mutual TLS for Linux CI.

Both daemons share a disposable CI host. This verifies transport/daemon identity,
not the required production separation into a dedicated parser host or VM.
No provider credentials, deployment credentials or veteran information are used.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def call(args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, timeout=120, **kwargs)


def setup(directory):
    directory.mkdir(mode=0o700)
    tls = directory / 'client'
    server = directory / 'server'
    tls.mkdir(mode=0o700)
    server.mkdir(mode=0o700)
    networks = json.loads(call(['docker', 'network', 'inspect', 'bridge']).stdout)
    address = networks[0]['IPAM']['Config'][0]['Gateway']
    application_id = call(['docker', 'info', '--format', '{{.ID}}']).stdout.decode().strip()
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
    with (directory / 'daemon.log').open('wb') as log:
        subprocess.Popen(['sudo', 'dockerd', '--host=' + env['DOCKER_HOST'], '--tlsverify',
                          '--tlscacert=' + str(server / 'ca.pem'), '--tlscert=' + str(server / 'server.pem'),
                          '--tlskey=' + str(server / 'server-key.pem'), '--data-root=' + str(directory / 'data'),
                          '--exec-root=' + str(directory / 'exec'), '--pidfile=' + str(directory / 'daemon.pid'),
                          '--bridge=none', '--iptables=false', '--ip-masq=false', '--ip-forward=false',
                          '--label=va-lse-purpose=parser-only'], stdout=log, stderr=log,
                         env={'PATH': os.environ['PATH']}, start_new_session=True)
    for attempt in range(60):
        result = subprocess.run(['docker', 'info', '--format', '{{.ID}}'], env=env, capture_output=True, timeout=3)
        if result.returncode == 0:
            engine_id = result.stdout.decode().strip()
            break
        time.sleep(.5)
    else:
        raise RuntimeError('Synthetic dedicated TLS engine did not start; inspect private CI daemon log.')
    assert engine_id and engine_id != application_id
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
    # Disposable runner destroys data-root. Remove private client/server keys now.
    for name in ('client', 'server', 'launcher-client'):
        path = directory / name
        if path.exists():
            call(['sudo', 'rm', '-rf', '--', str(path)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleanup', action='store_true')
    args = parser.parse_args()
    directory = (Path(os.environ['RUNNER_TEMP']) / 'va-lse-parser-engine-ci').resolve()
    cleanup(directory) if args.cleanup else setup(directory)
