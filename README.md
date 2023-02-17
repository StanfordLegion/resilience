# Automatic, Distributed Checkpointing for Legion

This library supports drop-in, distributed checkpointing for Legion
applications (currently C++ only, but with Regent support planned).

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
runtime->checkpoint(ctx, task);
```

That's it! Run your application and checkpoints will be dumped at
every call to `checkpoint()`.
