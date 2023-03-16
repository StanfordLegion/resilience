#!/usr/bin/env python3

import csv
import glob
import os
import re

_filename_re = re.compile(r'log_([0-9]+)x([0-9]+)_f([0-9]+)_r([0-9]+)(?:_prof)?_([0-9]+)[.]log')
def parse_basename(filename):
    match = re.match(_filename_re, filename)
    assert match is not None
    return match.groups()

_resource_re = re.compile(r'^\[[0-9]+ - [0-9a-f]+\] +[0-9.]+ \{3\}\{resilience\}: Serialized checkpoint ([0-9]+) in ([0-9.]+) seconds \(primary: ([0-9]+) bytes, ([0-9]+) futures, ([0-9]+) regions, ([0-9]+) ispaces\),  \(sharded: ([0-9]+) bytes, ([0-9]+) future_maps, ([0-9]+) ipartitions\), RSS = ([0-9]+) KiB$', re.MULTILINE)
def parse_content(path):
    with open(path, 'r') as f:
        content = f.read()
        return re.findall(_resource_re, content)

def main():
    paths = glob.glob('checkpoint/*.log')
    content = [(os.path.dirname(path),) + parse_basename(os.path.basename(path)) + content for path in paths for content in parse_content(path)]
    content.sort(key=lambda row: (row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])))

    import sys
    # with open(out_filename, 'w') as f:
    out = csv.writer(sys.stdout, dialect='excel-tab') # f)
    out.writerow(['system', 'nodes', 'procs_per_node', 'freq', 'rep', 'rank', 'checkpoint', 'serialization_seconds', 'primary_bytes', 'futures', 'regions', 'ispaces', 'shard_bytes', 'future_maps', 'partitions', 'rss_kib'])
    out.writerows(content)

if __name__ == '__main__':
    main()
