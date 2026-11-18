"""Own-process Linux memory observations. No system settings or other PIDs are touched."""

import mmap
import sys
from pathlib import Path


def unavailable(reason):
    return dict(status="unavailable", reason=reason)


def parse_status(text):
    result = {}
    for field in ("VmRSS", "VmHWM"):
        rows = [line.split() for line in text.splitlines() if line.startswith(field + ":")]
        if (
            len(rows) != 1
            or len(rows[0]) != 3
            or rows[0][2] != "kB"
            or not rows[0][1].isdigit()
        ):
            result[field] = unavailable("field missing, duplicated or malformed")
        else:
            result[field] = dict(status="available", value_kib=int(rows[0][1]))
    return result


def observe():
    if sys.platform != "linux":
        return {
            field: unavailable("Linux-only metric") for field in ("VmRSS", "VmHWM", "ru_maxrss")
        }
    try:
        result = parse_status(Path("/proc/self/status").read_text())
    except (OSError, UnicodeError) as exc:
        result = {field: unavailable(type(exc).__name__) for field in ("VmRSS", "VmHWM")}
    try:
        import resource

        result["ru_maxrss"] = dict(
            status="available", value_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
    except (ImportError, OSError) as exc:
        result["ru_maxrss"] = unavailable(type(exc).__name__)
    return result


def reset_peak():
    if sys.platform != "linux":
        return unavailable("Linux-only own-process phase peak reset")
    try:
        # Mode5 resets this mm's RSS peak accounting, not memory protections or data.
        Path("/proc/self/clear_refs").write_text("5\n")
    except OSError as exc:
        return unavailable(type(exc).__name__)
    return dict(status="available", method="own /proc/self/clear_refs mode 5")


def difference(later, earlier, field):
    if later[field]["status"] != "available" or earlier[field]["status"] != "available":
        return unavailable("both samples must expose the metric")
    return dict(
        status="available", value_kib=later[field]["value_kib"] - earlier[field]["value_kib"]
    )


def window_status(reset, baseline, after, alignment_kib=2048):
    if reset["status"] != "available":
        return unavailable("phase reset unavailable; retain absolute observations only")
    if any(
        sample[field]["status"] != "available"
        for sample in (baseline, after)
        for field in ("VmRSS", "VmHWM")
    ):
        return unavailable("VmRSS/VmHWM unavailable")
    if abs(baseline["VmHWM"]["value_kib"] - baseline["VmRSS"]["value_kib"]) > alignment_kib:
        return unavailable("post-reset peak is not close to current RSS")
    return dict(
        status="observed",
        scope="process RSS/HWM, not tensor allocation or exclusive model memory",
    )


def positive_control(size_bytes=16 * 1024 * 1024, minimum_growth_kib=12 * 1024):
    if sys.platform != "linux":
        return unavailable("Linux-only private mapping positive control")
    reset = reset_peak()
    baseline = observe()
    try:
        allocation = mmap.mmap(
            -1, size_bytes, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE
        )
        try:
            for offset in range(0, size_bytes, mmap.PAGESIZE):
                allocation[offset] = 1
            allocation[-1] = 1
            live = observe()
        finally:
            allocation.close()
    except (OSError, ValueError, MemoryError) as exc:
        return dict(
            status="unavailable", reason=type(exc).__name__, reset=reset, baseline=baseline
        )
    released = observe()
    checked = window_status(reset, baseline, live)
    rss = difference(live, baseline, "VmRSS")
    hwm = difference(live, baseline, "VmHWM")
    if checked["status"] == "unavailable":
        status = "unavailable"
    elif min(rss["value_kib"], hwm["value_kib"]) < minimum_growth_kib:
        status = "failed"
    else:
        status = "passed"
    return dict(
        status=status,
        allocation_bytes=size_bytes,
        minimum_growth_kib=minimum_growth_kib,
        reset=reset,
        baseline=baseline,
        while_live=live,
        after_release=released,
        rss_growth_kib=rss,
        hwm_growth_kib=hwm,
        scope="coarse resident-memory response; does NOT validate KiB-sized KV resolution",
    )
