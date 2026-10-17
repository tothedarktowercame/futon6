"""Run a per-paper CPU stage over its papers in as many processes as FUTON6_WORKERS asks.

Papers are independent, so a stage that takes them one at a time on a many-core
host leaves every core but one idle. `map_papers` runs a function over the papers
in config.workers() processes (FUTON6_WORKERS: a count, or `all` for every CPU the
job may use; unset, one, which is the serial loop) and yields the results in input
order, so what a stage prints and records is what its serial loop printed and
recorded. `map_in_order` does the same for any picklable items (a paper's graph
files, chunks of windows).

The function must be defined at module level (worker processes import it) and
should return its failures rather than raise them, as a serial loop that catches
per paper would.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

import futon6_config as config

Item = TypeVar("Item")
Result = TypeVar("Result")


def map_in_order(work: Callable[[Item], Result], items: Iterable[Item],
                 workers: int | None = None) -> Iterator[tuple[Item, Result]]:
    items = list(items)
    count = max(1, min(workers or config.workers(), len(items)))
    if count == 1:
        for item in items:
            yield item, work(item)
        return
    with ProcessPoolExecutor(max_workers=count) as pool:
        yield from zip(items, pool.map(work, items))


def map_papers(work: Callable[[str], Result], papers: Iterable[str],
               workers: int | None = None) -> Iterator[tuple[str, Result]]:
    yield from map_in_order(work, papers, workers)
