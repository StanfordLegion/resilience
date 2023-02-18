 * Initial fields allocated after creating a region
 * Track values escaping from futures
 * More flexible selection of checkpoint location (directory prefix, file patterns, etc.)
 * Double allocate regions to maintain consistent tree IDs on replay?
   * Needed because otherwise I don't think subtask allocation of partitions can be safe---if we no-op out the tasks, we only have the Future, which inherently capture handle IDs
   * (In other words, we need stable handle IDs to make nested partition creation work without instrumenting nested tasks)
   * Bonus is that it avoids runtime creation of new region trees
 * Nested partitions
 * Distributed region checkpoints
 * Add timing code on checkpoints so I can start tracking that
