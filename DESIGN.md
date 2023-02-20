# Design Decisions

This document attempts to capture some of the salient design decisions
in the resilience wrapper.

## What APIs To Interpose On?

 * Runtime methods
 * Data structures where we care about lifetimes or escaped data (Future, FutureMap)

We do NOT need to interpose on:

 * Handle data types (unless we care about lifetimes or escape)
   * IndexSpace, LogicalRegion
   * We can track our state separately
   * We can still interpose all associated Runtime methods
   * We can track lifetimes through Runtime destroy methods
   * There is no way to mutate these objects without a corresponding runtime call

## Uninitialized Regions

 * Problem: Legion throws an error if you read an uninitialized region
 * Solution 1: Track dirty regions
   * Requires tracking region state on every execute_task, etc.
   * Difficult to model precisely due to region tree hierarchy
     * E.g., if a subregion is uninitialized, you cannot save the whole region tree
   * Approximation DOES NOT work:
     * If you assume writes are complete, you hit errors when a subregion is uninitialized (bad)
     * If you assume writes are incomplete, you lose data (worse!)
 * Solution 2: Initialize all regions
   * No need to track region tree state
   * Allows approximation by ensuring that all data is initialized
     * (Now you can err on the side of assuming regions are dirty)

## Restoring Region Data

 * Problem: the partitions required to do a distributed load of region data from disk may not exist at the point where a region is created
 * Solution 1: Restore partitions eagerly
   * If you do this, you need to restore index spaces eagerly too
   * This is fine if we capture 100% of partitioning operations in the checkpointed task, but definitely causes reordering of operations if not
   * Requires tracking the identities of index spaces/partitions/regions and their relationships. Currently, don't need to know that region 2 depends on index space 1, because the user provides us with the index space via the API. If we do eager restore now we need to track this internally.
 * Solution 2: Defer region restore to the original checkpoint
   * Now all the partitions exist at the point where we do the restore
   * Regions will be uninitialized (or filled?) earlier in the execution, but that should be fine since we no-op every important operation
   * This also has the benefit that we automatically pick up any fields created after the region was created but before the checkpoint; i.e., we restore the same set of fields that we saved (which otherwise we'd need additional tracking for)

## Handle IDs

 * In general, handle IDs (e.g., region tree IDs) do NOT match from run to run
 * This is because we need to e.g., create additional regions on replay for use with attach launchers
   * This also happens in partition construction, see: https://github.com/StanfordLegion/legion/issues/1404
 * Therefore, handle IDs CANNOT be serialized; we need to use tags
 * Corollary: partition colors are not stable (because they are often auto-selected by the runtime) and cannot be replied upon

## Object Relationships

 * Some objects reference others (e.g., a region refers to an index space and field space, a partition refers to a color space)
 * Right not we DO NOT need to track these, because the necessary values are presented through the API (i.e., because execution is deterministic, we can rely on the user to pass us the right value at the right time to reconstruct the ith value)

## Region Lifetime

 * Right now we create regions even if we know they will later be destroyed
 * This is because certain API calls assume they exist and would error on a NO_REGION (e.g., attach_name)
 * An alternative would be to wrap these API calls better (e.g., to turn them into no-ops when they get NO_REGIONs). Or else we could wrap LogicalRegion but that would make the wrapper larger/more complicated
