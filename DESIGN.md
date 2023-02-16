# What Do Need To Interpose On?

  * Runtime methods
  * Data structures where we care about lifetimes or escaped data (Future, FutureMap)

We do NOT need to interpose on:

  * Handle data types (unless we care about lifetimes or escape)
      * IndexSpace, LogicalRegion
      * We can track our state separately
      * We can still interpose all associated Runtime methods
      * We can track lifetimes through Runtime destroy methods
      * There is no way to mutate these objects without a corresponding runtime call
