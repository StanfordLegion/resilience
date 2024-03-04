/* Copyright 2024 Stanford University
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

#ifndef RESILIENCE_MAPPER_H
#define RESILIENCE_MAPPER_H

#include "default_mapper.h"
#include "legion.h"
#include "resilience/types.h"

namespace ResilientLegion {
namespace Mapping {

class ResilientMapper : public DefaultMapper {
public:
  ResilientMapper(MapperRuntime *rt, Machine machine, Processor local,
                  const char *mapper_name);
  void default_policy_rank_processor_kinds(
      MapperContext ctx, const Task &task,
      std::vector<Processor::Kind> &ranking) override;
  Memory default_policy_select_target_memory(
      MapperContext ctx, Processor target_proc, const RegionRequirement &req,
      MemoryConstraint mc = MemoryConstraint()) override;
  LayoutConstraintID default_policy_select_layout_constraints(
      MapperContext ctx, Memory target_memory, const RegionRequirement &req,
      MappingKind mapping_kind, bool needs_field_constraint_check,
      bool &force_new_instances) override;
  LogicalRegion default_policy_select_instance_region(
      MapperContext ctx, Memory target_memory, const RegionRequirement &req,
      const LayoutConstraintSet &constraints, bool force_new_instances,
      bool meets_constraints) override;
  void map_copy(const MapperContext ctx, const Copy &copy, const MapCopyInput &input,
                MapCopyOutput &output) override;

private:
  std::map<LogicalRegion, std::vector<std::vector<PhysicalInstance>>> cached_copy_mapping;
  std::map<LogicalRegion, size_t> cached_copy_index;
};

}  // namespace Mapping

}  // namespace ResilientLegion

#endif  // RESILIENCE_MAPPER_H
