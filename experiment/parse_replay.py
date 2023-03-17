#!/usr/bin/env python3

import csv
import glob
import os
import re

_filename_re = re.compile(r'log_([0-9]+)x([0-9]+)_f([0-9]+)_replay([0-9]+)_0[.]log')
def parse_basename(filename):
    match = re.match(_filename_re, filename)
    assert match is not None
    return match.groups()

_replay_re = re.compile(r'^\[[0-9]+ - [0-9a-f]+\] +[0-9.]+ \{3\}\{resilience\}: Checkpoint replay finished in ([0-9.]+) seconds$', re.MULTILINE)
def parse_content(path):
    with open(path, 'r') as f:
        content = f.read()
        replay_match = re.search(_replay_re, content)
        replay = replay_match.group(1) if replay_match is not None else 'ERROR'
        return (replay,)

def main():
    paths = glob.glob('checkpoint/*_replay*_0.log')
    content = [(os.path.dirname(path),) + parse_basename(os.path.basename(path)) + parse_content(path) for path in paths]
    content.sort(key=lambda row: (row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4])))

    import sys
    # with open(out_filename, 'w') as f:
    out = csv.writer(sys.stdout, dialect='excel-tab') # f)
    out.writerow(['system', 'nodes', 'procs_per_node', 'freq', 'replay', 'replay_time'])
    out.writerows(content)

if __name__ == '__main__':
    main()
