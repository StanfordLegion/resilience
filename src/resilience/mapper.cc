/* Copyright 2023 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "resilience.h"

using namespace ResilientLegion;
using namespace ResilientLegion::Mapping;

static LegionRuntime::Logger::Category log_mapper("resilient_mapper");

ResilientMapper::ResilientMapper(MapperRuntime *rt, Machine machine, Processor local,
                                 const char *mapper_name)
    : DefaultMapper(rt, machine, local, mapper_name) {}

void ResilientMapper::default_policy_rank_processor_kinds(
    MapperContext ctx, const Task &task, std::vector<Processor::Kind> &ranking) {
  log_mapper.debug() << "ResilientMapper::default_policy_rank_processor_kinds";
  ranking.resize(5);
  ranking[0] = Processor::TOC_PROC;
  ranking[1] = Processor::PROC_SET;
  ranking[2] = Processor::IO_PROC;
  ranking[3] = Processor::LOC_PROC;
  ranking[4] = Processor::PY_PROC;
}

Memory ResilientMapper::default_policy_select_target_memory(MapperContext ctx,
                                                            Processor target_proc,
                                                            const RegionRequirement &req,
                                                            MemoryConstraint mc) {
  log_mapper.debug() << "ResilientMapper::default_policy_select_target_memory";

  // This is like the DefaultMapper version except we always prefer RDMA (and only
  // consider zero-copy memories to fit that criterion).

  assert(!mc.is_valid());

  const bool prefer_rdma = true;

  // Consult the processor-memory mapping cache.
  {
    // TODO: deal with the updates in machine model which will
    //       invalidate this cache
    std::map<Processor, Memory>::iterator it;
    if (prefer_rdma) {
      it = cached_rdma_target_memory.find(target_proc);
      if (it != cached_rdma_target_memory.end()) return it->second;
    } else {
      it = cached_target_memory.find(target_proc);
      if (it != cached_target_memory.end()) return it->second;
    }
  }

  // Find the visible memories from the processor for the given kind
  Machine::MemoryQuery visible_memories(machine);
  visible_memories.has_affinity_to(target_proc);
  if (visible_memories.count() == 0) {
    log_mapper.error("No visible memories from processor " IDFMT
                     "! "
                     "This machine is really messed up!",
                     target_proc.id);
    assert(false);
  }
  // Figure out the memory with the highest-bandwidth
  Memory best_memory = Memory::NO_MEMORY;
  unsigned best_bandwidth = 0;
  Memory best_rdma_memory = Memory::NO_MEMORY;
  unsigned best_rdma_bandwidth = 0;
  std::vector<Machine::ProcessorMemoryAffinity> affinity(1);
  for (Machine::MemoryQuery::iterator it = visible_memories.begin();
       it != visible_memories.end(); it++) {
    affinity.clear();
    machine.get_proc_mem_affinity(affinity, target_proc, *it,
                                  false /*not just local affinities*/);
    assert(affinity.size() == 1);
    if (!best_memory.exists() || (affinity[0].bandwidth > best_bandwidth)) {
      best_memory = *it;
      best_bandwidth = affinity[0].bandwidth;
    }
    // The only RDMA memory we care about is zero-copy. Pinned memory
    // provides no benefits for CPU disk copies.
    if (it->kind() == Memory::Z_COPY_MEM &&
        (!best_rdma_memory.exists() || (affinity[0].bandwidth > best_rdma_bandwidth))) {
      best_rdma_memory = *it;
      best_rdma_bandwidth = affinity[0].bandwidth;
    }
  }
  if (!best_memory.exists()) {
    log_mapper.error() << "Default mapper error: Failed to find a memory of kind "
                       << mc.get_kind() << " connected to processor " << target_proc;
    assert(false);
  }
  if (!best_rdma_memory.exists()) best_rdma_memory = best_memory;

  // Cache best memory for target processor.
  if (prefer_rdma)
    cached_rdma_target_memory[target_proc] = best_rdma_memory;
  else
    cached_target_memory[target_proc] = best_memory;

  return prefer_rdma ? best_rdma_memory : best_memory;
}

LogicalRegion ResilientMapper::default_policy_select_instance_region(
    MapperContext ctx, Memory target_memory, const RegionRequirement &req,
    const LayoutConstraintSet &constraints, bool force_new_instances,
    bool meets_constraints) {
  return req.region;
}

void ResilientMapper::map_copy(const MapperContext ctx, const Copy &copy,
                               const MapCopyInput &input, MapCopyOutput &output) {
  log_mapper.debug() << "ResilientMapper::map_copy";

  // Unlike the default mapper, we do NOT want to reuse source instances for these copies,
  // unless the instances are restricted.

  for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++) {
    // If the instance is restricted, just take that.
    if (copy.src_requirements[idx].is_restricted()) {
      output.src_instances[idx] = input.src_instances[idx];
      if (!output.src_instances[idx].empty())
        runtime->acquire_and_filter_instances(ctx, output.src_instances[idx]);
    } else {
      // Otherwise look in the cache for the instance (rotating our index to minimize
      // reuse), or create a fresh copy if needed.
      auto &req = copy.src_requirements[idx];
      auto it = cached_copy_mapping.find(req.region);
      auto cache_idx = cached_copy_index.find(req.region);
      if (it != cached_copy_mapping.end() && cache_idx != cached_copy_index.end() &&
          cache_idx->second < it->second.size()) {
        auto &next_cache_idx = cache_idx->second;
        auto &instances = it->second.at(next_cache_idx);
        next_cache_idx = (next_cache_idx + 1) % Runtime::config_max_instances;
        runtime->acquire_and_filter_instances(ctx, instances);
        if (instances.empty()) {
          log_mapper.error() << "instances got garbage collected, this is bad";
          abort();
        }
        output.src_instances[idx] = instances;
      } else {
        auto it = cached_copy_index.emplace(req.region, 0);
        auto &next_cache_idx = it.first->second;
        size_t cache_idx = next_cache_idx;
        next_cache_idx = (next_cache_idx + 1) % Runtime::config_max_instances;
        auto &cache = cached_copy_mapping[req.region];
        assert(cache.size() == cache_idx);
        cache.resize(cache_idx + 1);
        auto &instances = cache[cache_idx];
        resilient_create_copy_instance<true /*is src*/>(
            ctx, copy, copy.src_requirements[idx], idx, instances);
        // Make sure we don't GC instances we're caching.
        for (auto &instance : instances) {
          runtime->set_garbage_collection_priority(ctx, instance,
                                                   LEGION_GC_NEVER_PRIORITY + 1);
        }
        output.src_instances[idx] = instances;
      }
    }

    // Ok to take destination instances, if they exist.
    output.dst_instances[idx] = input.dst_instances[idx];
    if (!output.dst_instances[idx].empty()) {
      runtime->acquire_and_filter_instances(ctx, output.dst_instances[idx]);
    }

    if (!copy.dst_requirements[idx].is_restricted()) {
      resilient_create_copy_instance<false /*is src*/>(
          ctx, copy, copy.dst_requirements[idx], idx, output.dst_instances[idx]);
    }
  }
  // Shouldn't ever see this in the resilient mapper.
  assert(copy.src_indirect_requirements.empty());
  assert(copy.dst_indirect_requirements.empty());
}

template <bool IS_SRC>
void ResilientMapper::resilient_create_copy_instance(
    MapperContext ctx, const Copy &copy, const RegionRequirement &req, unsigned idx,
    std::vector<PhysicalInstance> &instances) {
  // This is identical to the DefaultMapper version except we always create new instances

  // See if we have all the fields covered
  std::set<FieldID> missing_fields = req.privilege_fields;
  for (std::vector<PhysicalInstance>::const_iterator it = instances.begin();
       it != instances.end(); it++) {
    it->remove_space_fields(missing_fields);
    if (missing_fields.empty()) break;
  }
  if (missing_fields.empty()) return;
  // If we still have fields, we need to make an instance
  // We clearly need to take a guess, let's see if we can find
  // one of our instances to use.
  Memory target_memory =
      default_policy_select_target_memory(ctx, copy.parent_task->current_proc, req);
  bool force_new_instances = true;  // Elliott: always force new instances
  LayoutConstraintID our_layout_id = default_policy_select_layout_constraints(
      ctx, target_memory, req, COPY_MAPPING, true /*needs check*/, force_new_instances);
  LayoutConstraintSet creation_constraints =
      runtime->find_layout_constraints(ctx, our_layout_id);
  creation_constraints.add_constraint(
      FieldConstraint(missing_fields, false /*contig*/, false /*inorder*/));
  instances.resize(instances.size() + 1);
  if (!default_make_instance(ctx, target_memory, creation_constraints, instances.back(),
                             COPY_MAPPING, force_new_instances, true /*meets*/, req)) {
    // If we failed to make it that is bad
    log_mapper.error()
        << "Resilient mapper failed allocation for "
        << (IS_SRC ? "source" : "destination") << " region requirement " << idx
        << " of explicit region-to-region copy operation in task "
        << copy.parent_task->get_task_name() << " (ID "
        << copy.parent_task->get_unique_id() << ") in memory " << target_memory
        << " for processor " << copy.parent_task->current_proc
        << ". This means the working set of your application is too big for the allotted "
           "capacity of the given memory under the default mapper's mapping scheme. You "
           "have three choices: ask Realm to allocate more memory, write a custom mapper "
           "to better manage working sets, or find a bigger machine. Good luck!";
    abort();
  }
}
