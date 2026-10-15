import argparse


def positive_integer(value: str) -> int:
    """Parse a strictly positive integer for an ``argparse`` option."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_integer(value: str) -> int:
    """Parse a non-negative integer for an ``argparse`` option."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed
