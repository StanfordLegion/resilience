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
