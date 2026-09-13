"""One managed native run, only from a freshly approved exact UI selection."""

import sys

from yamkit.ui.native_inference import execute_request

if __name__ == "__main__":
    raise SystemExit(execute_request(sys.argv[1], motion=True) if len(sys.argv) == 2 else 2)
