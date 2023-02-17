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

#include <signal.h>

#include <iostream>

#include "resilience.h"

using namespace ResilientLegion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  SUM_TASK_ID,
  WRITE_TASK_ID,
};

enum FieldIDs {
  POINT_FIELD_ID,
  VALUE_FIELD_ID,
};

int sum(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
        Runtime *runtime) {
  const FieldAccessor<READ_ONLY, int, 1> acc(regions[0], VALUE_FIELD_ID);
  DomainT<1> domain = runtime->get_index_space_domain(
      ctx, IndexSpaceT<1>(task->regions[0].region.get_index_space()));

  int total = 0;
  std::cout << "Data:";
  for (PointInDomainIterator<1> pir(domain); pir(); pir++) {
    std::cout << " (" << coord_t(*pir) << ", " << acc[*pir] << ")";
    total += acc[*pir];
  }
  std::cout << std::endl;
  return total;
}

void abort(InputArgs args) {
  for (int i = 1; i < args.argc; i++) {
    if (strstr(args.argv[i], "-abort")) raise(SIGSEGV);
  }
}

void top_level(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Runtime *runtime) {
  runtime->enable_checkpointing(ctx);

  int N = 10;
  const Rect<1> domain(0, N - 1);
  IndexSpaceT<1> ispace = runtime->create_index_space(ctx, domain);
  FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator fal = runtime->create_field_allocator(ctx, fspace);
    fal.allocate_field(sizeof(int), VALUE_FIELD_ID);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, ispace, fspace);

  int n = 2;
  Rect<1> color_bounds(0, n - 1);
  IndexSpace cspace = runtime->create_index_space(ctx, color_bounds);

  IndexPartition ip = runtime->create_equal_partition(ctx, ispace, cspace);
  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);

  LogicalRegion lsr = runtime->get_logical_subregion_by_color(ctx, lp, 0);

  runtime->fill_field<int>(ctx, lr, lr, VALUE_FIELD_ID, 1);
  runtime->fill_field<int>(ctx, lsr, lr, VALUE_FIELD_ID, 2);

  runtime->checkpoint(ctx, task);

  // Invalid, actually
  abort(Runtime::get_input_args());

  TaskLauncher sum_launcher(SUM_TASK_ID, TaskArgument());
  sum_launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
  sum_launcher.add_field(0, VALUE_FIELD_ID);
  Future sum_future = runtime->execute_task(ctx, sum_launcher);
  int sum = sum_future.get_result<int>();
  assert(sum == 15);
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<top_level>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(SUM_TASK_ID, "sum");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int, sum>(registrar, "sum");
  }

  return Runtime::start(argc, argv);
}
