#!/usr/bin/env python3

# Copyright 2024 Stanford University
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

import argparse, collections, re, os, shutil, subprocess, sys, tempfile

def run_cmd(cmd, test_dir, verbose):
    if not verbose:
        cmd = ['stdbuf', '-o0'] + cmd
    cmd_str = ' '.join([('"%s"' % x if ' ' in x else x) for x in cmd])
    if verbose:
        print(f'Command: {cmd_str}')
    proc = subprocess.run(
        cmd,
        stdout=(None if verbose else subprocess.PIPE),
        stderr=(None if verbose else subprocess.STDOUT),
        cwd=test_dir,
        encoding='utf-8')
    if proc.returncode != 0:
        if not verbose:
            print(f'Failed command: {cmd_str}')
            print(proc.stdout)
        print(f'Command exited with code: {proc.returncode}', flush=True)
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

def run_test_config(cmd, root_dir, diff, verbose):
    orig_dir = os.path.join(root_dir, 'orig')
    os.mkdir(orig_dir)
    print('Running original config...')
    run_cmd(cmd, orig_dir, verbose)
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
        run_cmd(cmd + ['-checkpoint:replay', str(i)], replay_dir, verbose)
        replay_checkpoints = list_checkpoints(replay_dir)
        if diff:
            # If execution is deterministic, replay should produce
            # identical checkpoint files to the original run.
            compare_checkpoints(orig_dir, checkpoints,
                                replay_dir, replay_checkpoints)

def run_all_tests(cmd, diff, verbose):
    root_dir = tempfile.mkdtemp()
    print(f'Test directory: {root_dir}')
    run_test_config(cmd, root_dir, diff, verbose)
    shutil.rmtree(root_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test runner')
    parser.add_argument('--no-diff', action='store_true', help='do not diff resulting checkpoints')
    parser.add_argument('--verbose', '-v', action='store_true', help='verbose output')
    parser.add_argument(dest='command', nargs='+', help='command to execute')
    args = parser.parse_args()
    run_all_tests(args.command, not args.no_diff, args.verbose)
