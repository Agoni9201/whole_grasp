#!/usr/bin/env python3
"""Read Linux sysfs USB link speeds before handover grasp; no device SDK/CAN."""
from __future__ import annotations

import argparse
from pathlib import Path


def usb_cameras(root=Path('/sys/bus/usb/devices')):
    result = []
    for path in sorted(Path(root).glob('*')):
        try:
            if not (path/'product').is_file():
                continue
            product = (path/'product').read_text().strip()
            if not any(token in product.lower() for token in ('orbbec', 'gemini', 'realsense')):
                continue
            serial = (path/'serial').read_text().strip() if (path/'serial').is_file() else None
            speed = float((path/'speed').read_text().strip())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f'cannot inspect USB camera at {path}: {exc}') from exc
        result.append(dict(usb_path=path.name, product=product, serial=serial, speed_mbps=speed))
    return result


def require_wrist_superspeed(devices, serial):
    selected = [d for d in devices if d['serial'] == serial]
    if len(selected) != 1:
        raise RuntimeError(f'expected one connected wrist camera {serial}; found {len(selected)}')
    wrist = selected[0]
    if wrist['speed_mbps'] < 5000:
        raise RuntimeError(
            f"wrist {serial} at {wrist['usb_path']} is linked at {wrist['speed_mbps']:g}M; "
            'this 1280x720@30 collection setup requires USB 3 SuperSpeed (>=5000M). '
            'Reconnect the camera using a USB 3 data cable and USB 3 port; '
            'a USB 2 hub or upstream cable also limits the link. '
            'Do not lower the calibrated image resolution as an automatic fallback.')
    return wrist


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wrist-serial', default='CP0T963000S4')
    args = p.parse_args(argv)
    print('[preflight][usb] sysfs only; no camera streams, CAN or arm/gripper commands')
    try:
        devices = usb_cameras()
        for device in devices:
            print(f"[preflight][usb] {device['usb_path']} {device['product']} "
                  f"serial={device['serial'] or '(not exposed in sysfs)'} "
                  f"link={device['speed_mbps']:g}M")
        require_wrist_superspeed(devices, args.wrist_serial)
    except RuntimeError as exc:
        print(f'[preflight][usb][blocked] {exc}')
        return 2
    print('[preflight][usb] wrist SuperSpeed link OK; stream profile/frames still need verification')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
