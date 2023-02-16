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
