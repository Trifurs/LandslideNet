#!/usr/bin/env python3
"""Step 3: generate full-domain susceptibility maps."""

from utils.prediction import parse_args, run


def main(argv=None):
    """Parse command-line arguments and run susceptibility mapping."""
    run(parse_args(argv))


if __name__ == "__main__":
    main()
