"""Paradigm timeline and balanced randomisation for the guess-number P300 task."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ParadigmConfig:
    blocks: int = 6
    repetitions: int = 5
    fixation_s: float = 0.5
    stimulus_s: float = 0.2
    blank_s: float = 1.3
    inter_block_s: float = 2.0
    start_delay_s: float = 1.0
    end_delay_s: float = 1.0
    seed: int = 0


def make_block_sequence(block: int, repetitions: int, rng: np.random.Generator) -> list[int]:
    """Balanced random order of 1..9, each repeated `repetitions` times."""
    numbers = []
    for _ in range(repetitions):
        order = list(range(1, 10))
        rng.shuffle(order)
        numbers.extend(order)
    # Avoid immediate exact repeats when possible.
    for _ in range(20):
        swapped = False
        for i in range(1, len(numbers)):
            if numbers[i] == numbers[i - 1]:
                candidates = [j for j in range(i + 1, len(numbers)) if numbers[j] != numbers[i]]
                if candidates:
                    j = int(rng.choice(candidates))
                    numbers[i], numbers[j] = numbers[j], numbers[i]
                    swapped = True
        if not swapped:
            break
    return numbers


@dataclass
class TimelineEvent:
    time_sec: float
    kind: str
    duration_sec: float = 0.0
    number: int | None = None
    block: int | None = None
    trial: int | None = None


class Paradigm:
    """Builds the full event timeline and exposes a small state machine."""

    def __init__(self, cfg: ParadigmConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.events: list[TimelineEvent] = []
        self._build()

    def _build(self) -> None:
        cfg = self.cfg
        t = cfg.start_delay_s
        for block in range(cfg.blocks):
            self.events.append(TimelineEvent(t, "block_start", block=block))
            self.events.append(TimelineEvent(t, "fixation_on", cfg.fixation_s, block=block))
            t += cfg.fixation_s
            self.events.append(TimelineEvent(t, "fixation_off", block=block))
            seq = make_block_sequence(block, cfg.repetitions, self.rng)
            for trial, number in enumerate(seq):
                self.events.append(TimelineEvent(t, "stim_on", cfg.stimulus_s,
                                                 number=number, block=block, trial=trial))
                t += cfg.stimulus_s
                self.events.append(TimelineEvent(t, "stim_off", block=block, number=number, trial=trial))
                t += cfg.blank_s
            self.events.append(TimelineEvent(t, "block_end", block=block))
            t += cfg.inter_block_s
        self.events.append(TimelineEvent(t, "session_end"))
        self.events.append(TimelineEvent(t + cfg.end_delay_s, "session_stop"))

    @property
    def duration_sec(self) -> float:
        return self.events[-1].time_sec

    @property
    def stim_on_events(self) -> list[TimelineEvent]:
        return [e for e in self.events if e.kind == "stim_on"]

    def schedule_records(self) -> list[dict[str, Any]]:
        """JSON-friendly timeline (used by mock generator and session metadata)."""
        return [{
            "time_sec": e.time_sec,
            "kind": e.kind,
            "duration_sec": e.duration_sec,
            "number": e.number,
            "block": e.block,
            "trial": e.trial,
        } for e in self.events]


class TimelineRunner:
    """Consume due timeline events with a monotonic clock."""

    def __init__(self, paradigm: Paradigm, on_event: Callable[[TimelineEvent, float], None]) -> None:
        self.paradigm = paradigm
        self.on_event = on_event
        self.queue = list(paradigm.events)
        self.finished = False

    def tick(self, now_sec: float) -> None:
        while self.queue and now_sec >= self.queue[0].time_sec:
            event = self.queue.pop(0)
            self.on_event(event, now_sec)
        if not self.queue:
            self.finished = True

    @property
    def next_deadline(self) -> float | None:
        return self.queue[0].time_sec if self.queue else None
