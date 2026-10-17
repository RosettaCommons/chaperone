#!/usr/bin/env python3
"""Compatibility CLI for launching the Apptainer-backed Harbor task."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _gpu_selection(
    value: str, environment: dict[str, str]
) -> tuple[int, str, list[str]]:
    if value != "auto":
        count = int(value)
        if not 0 <= count < 5:
            raise argparse.ArgumentTypeError(
                "--gpus must be auto or an integer from 0 through 4"
            )
        if count == 0:
            return 0, "--gpus", []
        for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
            raw = environment.get(name, "").strip()
            if not raw or raw.lower() in {"none", "void", "-1", "all"}:
                continue
            devices = [device.strip() for device in raw.split(",") if device.strip()]
            if len(devices) < count:
                raise argparse.ArgumentTypeError(
                    f"--gpus={count} exceeds the {len(devices)} devices exposed by {name}"
                )
            return count, f"--gpus constrained by {name}", devices[:count]
        return count, "--gpus", [str(index) for index in range(count)]

    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        raw = environment.get(name, "").strip()
        if not raw or raw.lower() in {"none", "void", "-1"}:
            continue
        devices = [device.strip() for device in raw.split(",") if device.strip()]
        if devices == ["all"]:
            raise argparse.ArgumentTypeError(
                f"{name}=all is ambiguous; pass an explicit --gpus count"
            )
        if len(devices) >= 5:
            raise argparse.ArgumentTypeError(
                f"{name} exposes {len(devices)} devices, but this task permits at most 4"
            )
        return len(devices), name, devices
    return 0, "no visible-device environment", []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a design evaluation with Harbor and Apptainer"
    )
    parser.add_argument("mode", choices=("local", "job"))
    parser.add_argument("--agent", choices=("codex",), default="codex")
    parser.add_argument(
        "--arm", choices=("baseline", "harness", "both"), default="harness"
    )
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--attempts", type=_positive_integer, default=1)
    parser.add_argument("--concurrency", type=_positive_integer, default=1)
    parser.add_argument("--model")
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        help="Codex model reasoning effort (default: medium)",
    )
    parser.add_argument("--prompt", type=Path)
    args = parser.parse_args()

    environment = os.environ.copy()
    try:
        gpu_count, source, devices = _gpu_selection(args.gpus, environment)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))

    environment["GPUS"] = str(gpu_count)
    if args.gpus != "auto":
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        environment["NVIDIA_VISIBLE_DEVICES"] = ",".join(devices) if devices else "none"
    environment["ATTEMPTS"] = str(args.attempts)
    environment["CONCURRENCY"] = str(args.concurrency)
    environment["REASONING_EFFORT"] = args.reasoning_effort
    if args.model:
        environment["MODEL"] = args.model

    print(
        f"Local GPU assignment: count={gpu_count} devices={devices} source={source}",
        flush=True,
    )
    runner = Path(__file__).resolve().with_name("run.sh")
    command = [str(runner), args.mode, args.arm]
    if args.prompt:
        command.append(str(args.prompt.resolve()))
    os.execve(runner, command, environment)


if __name__ == "__main__":
    main()
