"""EEG acquisition backends: BrainSync SDK (real device) and deterministic mock."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable

import numpy as np

from .mock_eeg import generate_session_data

logger = logging.getLogger("frontend.acquisition")

ChunkCallback = Callable[[np.ndarray, int], None]  # samples (ch, n), start_sample


class MockAcquirer:
    """Streams pre-generated synthetic EEG in near-real time."""

    def __init__(self, sfreq: int, channels: list[str], stimuli: list, target_number: int,
                 seed: int = 0, chunk_size: int = 10) -> None:
        self.sfreq = int(sfreq)
        self.channels = list(channels)
        self.chunk_size = int(chunk_size)
        self.raw = generate_session_data(self.sfreq, self.channels, stimuli,
                                         target_number, seed=seed)
        self._pos = 0
        self._last_chunk_mono: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.callback: ChunkCallback | None = None

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._pos

    def estimated_sample_count(self, now_monotonic: float) -> int:
        """Interpolate sample index between chunks to reduce marker timing jitter."""
        with self._lock:
            if self._last_chunk_mono is None:
                return self._pos
            est = self._pos + max(0, int((now_monotonic - self._last_chunk_mono) * self.sfreq))
            return min(est, self.raw.shape[1])

    def start(self, callback: ChunkCallback) -> None:
        self.callback = callback
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mock-eeg", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                start = self._pos
                if start >= self.raw.shape[1]:
                    break
                stop = min(start + self.chunk_size, self.raw.shape[1])
                chunk = self.raw[:, start:stop]
                self._pos = stop
            if self.callback is not None:
                try:
                    self.callback(chunk, start)
                except Exception:
                    logger.exception("mock acquisition callback failed")
            with self._lock:
                self._last_chunk_mono = time.monotonic()
            next_t += self.chunk_size / self.sfreq
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None


class BrainSyncAcquirer:
    """Real BrainSync BS8A acquirer. Runs the async SDK API in a worker thread."""

    def __init__(self, sfreq: int = 500, gain: str = "Gain24", batch_size: int = 10,
                 channels: list[str] | None = None) -> None:
        self.sfreq = int(sfreq)
        self.gain = gain
        self.batch_size = int(batch_size)
        self.channels = channels or []
        self._pos = 0
        self._last_chunk_mono: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.callback: ChunkCallback | None = None
        self.device_handle = None
        self.loss_stats: dict = {}
        self.last_error: Exception | None = None

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._pos

    def estimated_sample_count(self, now_monotonic: float) -> int:
        """Interpolate sample index between SDK callbacks.

        Batch size is intentionally small (10 samples at 500 Hz = 20 ms) so this
        estimate is close even when the SDK batches packets.
        """
        with self._lock:
            if self._last_chunk_mono is None:
                return self._pos
            est = self._pos + max(0, int((now_monotonic - self._last_chunk_mono) * self.sfreq))
            return min(est, self._pos + self.batch_size)

    def start(self, callback: ChunkCallback) -> None:
        self.callback = callback
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="brainsync-eeg", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            asyncio.run(self._async_run())
        except Exception as exc:
            self.last_error = exc
            logger.error("BrainSync acquisition failed: %s", exc)

    async def _async_run(self) -> None:
        import brainsync_sdk

        gain_map = {
            "Gain1": brainsync_sdk.EegGain.Gain1,
            "Gain2": brainsync_sdk.EegGain.Gain2,
            "Gain4": brainsync_sdk.EegGain.Gain4,
            "Gain6": brainsync_sdk.EegGain.Gain6,
            "Gain8": brainsync_sdk.EegGain.Gain8,
            "Gain12": brainsync_sdk.EegGain.Gain12,
            "Gain24": brainsync_sdk.EegGain.Gain24,
        }
        rate_map = {
            250: brainsync_sdk.EegSampleRate.Hz250,
            500: brainsync_sdk.EegSampleRate.Hz500,
            1000: brainsync_sdk.EegSampleRate.Hz1000,
            2000: brainsync_sdk.EegSampleRate.Hz2000,
            4000: brainsync_sdk.EegSampleRate.Hz4000,
            8000: brainsync_sdk.EegSampleRate.Hz8000,
        }
        gain_enum = gain_map.get(self.gain, brainsync_sdk.EegGain.Gain24)
        rate_enum = rate_map.get(self.sfreq, brainsync_sdk.EegSampleRate.Hz500)

        self.device_handle = await brainsync_sdk.open_brainsync_serial()
        await brainsync_sdk.set_eeg_sample_rate(self.device_handle, rate_enum)
        await brainsync_sdk.set_eeg_gain(self.device_handle, gain_enum)
        await brainsync_sdk.set_eeg_signal_type(self.device_handle, brainsync_sdk.EegSignalType.Normal)
        await brainsync_sdk.clear_receive_buffer(self.device_handle)

        def on_packets(packets) -> None:
            try:
                n = len(packets)
                data = np.stack([np.asarray(p.to_microvolts(gain_enum), dtype=np.float32)
                                 for p in packets], axis=1)
                if data.shape[0] != len(self.channels) and self.channels:
                    data = data[:len(self.channels), :]
                with self._lock:
                    start = self._pos
                    self._pos += n
                    self._last_chunk_mono = time.monotonic()
                if self.callback is not None:
                    self.callback(data, start)
            except Exception:
                logger.exception("BrainSync packet callback failed")

        await brainsync_sdk.subscribe_eeg_data(self.device_handle, self.batch_size, on_packets)
        await brainsync_sdk.set_eeg_transfer(self.device_handle, True)
        await asyncio.sleep(0.3)
        try:
            await brainsync_sdk.reset_eeg_loss_stats(self.device_handle)
        except Exception:
            pass

        try:
            while not self._stop.is_set():
                await asyncio.sleep(0.1)
        finally:
            try:
                await brainsync_sdk.set_eeg_transfer(self.device_handle, False)
            except Exception:
                pass
            try:
                await brainsync_sdk.unsubscribe_eeg_data(self.device_handle)
            except Exception:
                pass
            try:
                self.loss_stats = await brainsync_sdk.get_eeg_loss_stats(self.device_handle)
            except Exception:
                pass
            try:
                await brainsync_sdk.close_device(self.device_handle)
            except Exception:
                pass

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None


def create_acquirer(mode: str, sfreq: int, channels: list[str], stimuli: list,
                    target_number: int, seed: int) -> object:
    if mode == "mock":
        return MockAcquirer(sfreq, channels, stimuli, target_number, seed=seed)
    if mode == "device":
        return BrainSyncAcquirer(sfreq=sfreq, channels=channels)
    raise ValueError(f"Unknown acquisition mode: {mode!r}")
