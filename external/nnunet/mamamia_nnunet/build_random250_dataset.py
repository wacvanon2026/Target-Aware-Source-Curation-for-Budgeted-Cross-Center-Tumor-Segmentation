#!/usr/bin/env python3
import sys
from build_datasets import main
if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('random250')
    main()
