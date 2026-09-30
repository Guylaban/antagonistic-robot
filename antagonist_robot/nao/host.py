"""Resolve the NAO robot's current IPv4 address.

On a direct Ethernet cable there is no DHCP, so the robot self-assigns a
link-local 169.254.x.x address that changes between sessions (sometimes
within minutes). Configure nao.ip as "nao.local" and resolve it on every
connection instead of hardcoding an address.
"""

import socket


def resolve_ipv4(host: str, port: int) -> str:
    """Return an IPv4 address for host.

    Forces the IPv4 record: nao.local also answers over IPv6, but
    nao_speaker_server.py only listens on IPv4. An IP literal is
    returned unchanged.

    Raises:
        OSError: if the name cannot be resolved (robot off or not cabled).
    """
    info = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    return info[0][4][0]
