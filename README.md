# Automatic, Distributed Checkpointing for Legion

This library supports drop-in, distributed checkpointing for Legion
applications (currently C++ only, but with Regent support planned).

## Quickstart

The fastest way to build from scratch is to run:

```bash
./test.sh
```

This will download and build Legion, build the checkpointing library,
build a suite of test applications and examples and run their test
suites. Assuming the test suite passes, everything is working at this
point.

If you need to install by hand, see the instructions below.

## Getting Started

To use it, change the following lines in your application:

```c++
#include "legion.h"
using namespace Legion;
```

To:

```c++
#include "resilience.h"
using namespace ResilientLegion;
```

Add the following line to the start of your top-level task (assuming
that's what you want to checkpoint):

```c++
runtime->enable_checkpointing(ctx);
```

Then, wherever you want to add checkpoints, call:

```c++
runtime->checkpoint(ctx);
```

That's it! Run your application and checkpoints will be dumped at
every call to `checkpoint()`. When you replay, add the flags `-replay
-cpt 0` (to replay from checkpoint `0`) and your application will
restart at that checkpoint.

```console
$ ./tests/region
Done!
Data: 1 2 3 4 5 6 7 8 9 10 11

$ ls *.dat
checkpoint.0.dat  checkpoint.0.lr.0..dat

$ ./tests/region -replay -cpt 0 -level resilience=2
[0 - 7fd18129a800]    0.062628 {2}{resilience}: In enable_checkpointing: replay 1 load_checkpoint_tag 0
[0 - 7fd18129a800]    0.063989 {2}{resilience}: After loading checkpoint, max: api 0 future 1 future_map 0 index_space 1 region_tag 1 partition 0 checkpoint 1
[0 - 7fd18129a800]    0.067304 {2}{resilience}: execute_task: no-op for replay, tag 0
[0 - 7fd18129a800]    0.067363 {2}{resilience}: In checkpoint: restoring regions from tag 0
[0 - 7fd18129a800]    0.067370 {2}{resilience}: restore_region_content: region from checkpoint, tag 0
[0 - 7fd18129a800]    0.067503 {2}{resilience}: restore_region: file_name checkpoint.0.lr.0..dat
[0 - 7fd18129a800]    0.068975 {2}{resilience}: In checkpoint: skipping tag 0 max 1
[0 - 7fd18129a800]    0.069044 {2}{resilience}: execute_task: launching task_id 1
Done!
Data: 1 2 3 4 5 6 7 8 9 10 11
```

## Installing

Build Legion with CMake, and install it to some path. The
checkpointing framework **ONLY** works with CMake.

To build the checkpointing framework:

```bash
mkdir resilience/build
cd resilience/build
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH=$PWD/../legion/install # or whever Legion is installed
```

Change `Debug` to `Release` when your application is working and you
start to do performance tests.
