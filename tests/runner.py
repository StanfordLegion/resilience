#!/usr/bin/env python3

# Copyright 2023 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import collections, re, os, shutil, subprocess, sys, tempfile

def run_cmd(cmd, test_dir):
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=test_dir,
        encoding='utf-8')
    if proc.returncode != 0:
        print(proc.stdout)
        sys.exit(proc.returncode)

_checkpoint_pattern = re.compile(r'checkpoint\.([0-9]*)\.dat')
_subcheckpoint_pattern = re.compile(r'checkpoint\.([0-9]*)\..*\.dat')
def list_checkpoints(test_dir):
    checkpoints = []
    subcheckpoints = collections.defaultdict(list)
    for filename in os.listdir(test_dir):
        match = re.fullmatch(_checkpoint_pattern, filename)
        if match is not None:
            checkpoints.append((int(match.group(1)), filename))
        match = re.fullmatch(_subcheckpoint_pattern, filename)
        if match is not None:
            subcheckpoints[int(match.group(1))].append(filename)
    return [(x, [y] + sorted(subcheckpoints[x]))
            for (x, y) in sorted(checkpoints)]

def compare_checkpoints(orig_dir, orig_checkpoints,
                        replay_dir, replay_checkpoints):
    # We flip things around because only the suffix matches
    zipped = reversed(list(zip(reversed(orig_checkpoints),
                               reversed(replay_checkpoints))))
    for ((orig_idx, orig_files), (replay_idx, replay_files)) in zipped:
        assert orig_idx == replay_idx
        assert orig_files == replay_files
        for f in orig_files:
            subprocess.run(
                ['diff',
                 os.path.join(orig_dir, f),
                 os.path.join(replay_dir, f)],
                check=True)

def run_test_config(cmd, root_dir):
    orig_dir = os.path.join(root_dir, 'orig')
    os.mkdir(orig_dir)
    print('Running original config...')
    run_cmd(cmd, orig_dir)
    checkpoints = list_checkpoints(orig_dir)
    print(f'Got {len(checkpoints)} checkpoints')
    for (i, checkpoint_files) in checkpoints:
        print(f'Testing replay from checkpoint {i}...')
        replay_dir = os.path.join(orig_dir, f'replay{i}')
        os.mkdir(replay_dir)
        for checkpoint_file in checkpoint_files:
            shutil.copyfile(
                os.path.join(orig_dir, checkpoint_file),
                os.path.join(replay_dir, checkpoint_file))
        run_cmd(cmd + ['-replay', '-cpt', str(i)], replay_dir)
        replay_checkpoints = list_checkpoints(replay_dir)
        # If execution is deterministic, replay should produce
        # identical checkpoint files to the original run.
        compare_checkpoints(orig_dir, checkpoints,
                            replay_dir, replay_checkpoints)

def run_all_tests(cmd):
    root_dir = tempfile.mkdtemp()
    print(f'Test directory: {root_dir}')
    run_test_config(cmd, root_dir)
    shutil.rmtree(root_dir)

if __name__ == '__main__':
    cmd = sys.argv[1:]
    run_all_tests(cmd)
