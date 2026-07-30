#!/usr/bin/env python3
"""Step 2: run leakage-safe regional experiments and manuscript comparisons."""

from utils.experiment import parse_args, run


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
