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

ResilientMapper::ResilientMapper(MapperRuntime *rt, Machine machine, Processor local,
                                 const char *mapper_name)
    : DefaultMapper(rt, machine, local, mapper_name) {}

void ResilientMapper::default_policy_rank_processor_kinds(
    MapperContext ctx, const Task &task, std::vector<Processor::Kind> &ranking) {
  ranking.resize(5);
  ranking[0] = Processor::TOC_PROC;
  ranking[1] = Processor::PROC_SET;
  ranking[2] = Processor::IO_PROC;
  ranking[3] = Processor::LOC_PROC;
  ranking[4] = Processor::PY_PROC;
}

LogicalRegion ResilientMapper::default_policy_select_instance_region(
    MapperContext ctx, Memory target_memory, const RegionRequirement &req,
    const LayoutConstraintSet &constraints, bool force_new_instances,
    bool meets_constraints) {
  return req.region;
}

void ResilientMapper::map_copy(const MapperContext ctx, const Copy &copy,
                               const MapCopyInput &input, MapCopyOutput &output) {
  // Unlike the default mapper, we do NOT want to reuse source instances for these copies,
  // unless the instances are restricted.

  bool has_unrestricted = false;
  for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++) {
    // Always make new source instances unless restricted.
    if (!copy.src_requirements[idx].is_restricted())
      default_create_copy_instance<true /*is src*/>(ctx, copy, copy.src_requirements[idx],
                                                    idx, output.src_instances[idx]);

    // Ok to take destination instances, if they exist.
    output.dst_instances[idx] = input.dst_instances[idx];
    if (!output.dst_instances[idx].empty())
      runtime->acquire_and_filter_instances(ctx, output.dst_instances[idx]);
    if (!copy.dst_requirements[idx].is_restricted()) has_unrestricted = true;
  }
  // If the destinations were all restricted we know we got everything
  if (has_unrestricted) {
    for (unsigned idx = 0; idx < copy.dst_requirements.size(); idx++) {
      output.dst_instances[idx] = input.dst_instances[idx];
      if (!copy.dst_requirements[idx].is_restricted())
        default_create_copy_instance<false /*is src*/>(
            ctx, copy, copy.dst_requirements[idx], idx, output.dst_instances[idx]);
    }
  }
  // Shouldn't ever see this in the resilient mapper.
  assert(!copy.src_indirect_requirements.empty());
  assert(!copy.dst_indirect_requirements.empty());
}
