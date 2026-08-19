"""Electrode tap test for BrainSync BS8A.

Streams all 8 hardware channels in Normal mode, Gain24, 250 Hz and prints the
per-channel peak-to-peak amplitude every second.  Tap/shake one electrode at a
time: the responding hardware channel index tells you the real wire order.

Usage:
    python scripts/electrode_tap_test.py --seconds 20
"""
from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np


def print_stats(buffers, label: str) -> None:
    print(f"\n[{label}] per-channel amplitude (uV):")
    print(f"{'CH':>4} {'min':>10} {'max':>10} {'ptp':>10} {'std':>10}")
    for i, buf in enumerate(buffers):
        if len(buf) == 0:
            print(f"{i:>4} {'--':>10} {'--':>10} {'--':>10} {'--':>10}")
            continue
        arr = np.asarray(buf)
        print(f"{i:>4} {arr.min():>10.2f} {arr.max():>10.2f} "
              f"{np.ptp(arr):>10.2f} {arr.std():>10.2f}")


async def main(seconds: int, gain: str) -> None:
    import brainsync_sdk as sdk

    gain_map = {
        "Gain1": sdk.EegGain.Gain1, "Gain2": sdk.EegGain.Gain2,
        "Gain4": sdk.EegGain.Gain4, "Gain6": sdk.EegGain.Gain6,
        "Gain8": sdk.EegGain.Gain8, "Gain12": sdk.EegGain.Gain12,
        "Gain24": sdk.EegGain.Gain24,
    }
    gain_enum = gain_map[gain]
    buffers = [[] for _ in range(8)]
    last_print = time.monotonic()

    def on_packets(packets):
        nonlocal last_print
        for p in packets:
            mv = p.to_microvolts(gain_enum)
            for ch in range(8):
                buffers[ch].append(float(mv[ch]))
        if time.monotonic() - last_print >= 1.0:
            print_stats(buffers, "live")
            last_print = time.monotonic()

    handle = await sdk.open_brainsync_serial()
    await sdk.set_eeg_sample_rate(handle, sdk.EegSampleRate.Hz250)
    await sdk.set_eeg_gain(handle, gain_enum)
    await sdk.set_eeg_signal_type(handle, sdk.EegSignalType.Normal)
    try:
        await sdk.set_eeg_signal_types(handle, [sdk.EegSignalType.Normal] * 8)
    except Exception:
        pass
    try:
        await sdk.set_eeg_gains(handle, [gain_enum] * 8)
    except Exception:
        pass
    await sdk.subscribe_eeg_data(handle, 250, on_packets)
    await sdk.set_eeg_transfer(handle, True)
    print(f"Streaming for {seconds}s. Tap/shake one electrode at a time...")
    await asyncio.sleep(seconds)
    await sdk.set_eeg_transfer(handle, False)
    await sdk.unsubscribe_eeg_data(handle)
    print_stats(buffers, "final")
    try:
        await sdk.close_device(handle)
    except Exception:
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=int, default=20)
    p.add_argument("--gain", default="Gain24")
    args = p.parse_args()
    asyncio.run(main(args.seconds, args.gain))
