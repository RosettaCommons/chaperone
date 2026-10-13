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


def parse_steps(argv):
    """
    Extracts a list of step names from `argv` until the first flag is encountered.
    Example argv:
        ["pipeline", "stepA", "stepB", "--stepA.k=3", "--help"]
    Returns:
        ["stepA", "stepB"]
    """
    steps = []
    for arg in argv[1:]:  # skip argv[0], the program name
        if arg.startswith("-"):
            break
        if arg.strip():  # Filter out empty strings
            steps.append(arg)
    return steps


def get_global_config():
    # TODO: Pull from CHAPERONE_CONFIG_SEARCH_PATHS
    return {}


def get_flag_value(block_flag_name, global_value):
    from absl import flags

    f = flags.FLAGS
    """Get block-specific flag value if set, otherwise use global value."""
    if hasattr(f, block_flag_name):
        flag_obj = f[block_flag_name]
        # Check if flag was explicitly set by comparing to default
        # ABSL sets .present for flags that were actually parsed
        if flag_obj.present:
            return flag_obj.value
    return global_value
