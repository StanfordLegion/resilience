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

#include "legion_resilience.h"

using namespace ResilientLegion;

int sum(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
        Runtime *runtime) {
  PhysicalRegion pr = regions[0];
  const FieldAccessor<READ_ONLY, int, 1> acc(pr, 0);
  auto domain =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  int total = 0;
  for (PointInRectIterator<1> pir(domain); pir(); pir++) {
    total += acc[*pir];
  }
  return total;
}

void write(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
           Runtime *runtime) {
  PhysicalRegion pr = regions[0];
  const FieldAccessor<READ_WRITE, Point<1>, 1> acc(pr, 1);
  auto domain =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(domain); pir(); pir++) {
    acc[*pir] = *pir;
  }
}

void abort(InputArgs args) {
  for (int i = 1; i < args.argc; i++) {
    if (strstr(args.argv[i], "-abort")) raise(SIGSEGV);
  }
}

void top_level(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Runtime *runtime) {
  runtime->enable_checkpointing();

  int N = 10;
  const Rect<1> domain(0, N - 1);
  IndexSpaceT<1> ispace = runtime->create_index_space(ctx, domain);
  FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator fal = runtime->create_field_allocator(ctx, fspace);
    fal.allocate_field(sizeof(int), 0);
    fal.allocate_field(sizeof(Point<1>), 1);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, ispace, fspace);

  int n = 10;
  Rect<1> color_bounds(0, n - 1);
  IndexSpace cspace = runtime->create_index_space(ctx, color_bounds);

  TaskLauncher write(2, TaskArgument());
  write.add_region_requirement(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
  write.add_field(0, 1);
  runtime->execute_task(ctx, write);

  IndexPartition ip = runtime->create_partition_by_field(ctx, lr, lr, 1, cspace);
  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);

  for (int i = 0; i < 10; i++) {
    LogicalRegion lsr = runtime->get_logical_subregion_by_color(ctx, lp, i);
    runtime->fill_field<int>(ctx, lsr, lr, 0, 2 * i);
  }

  runtime->checkpoint(ctx, task);

  // Invalid, actually
  abort(Legion::Runtime::get_input_args());

  TaskLauncher sum_launcher(1, TaskArgument());
  sum_launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
  sum_launcher.add_field(0, 0);
  Future sum_future = runtime->execute_task(ctx, sum_launcher);
  std::cout << "Got here!\n";
  int sum = sum_future.get_result<int>();
  assert(sum == 90);
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(0);
  {
    TaskVariantRegistrar registrar(0, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(1, "sum");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<int, sum>(registrar, "sum");
  }
  {
    TaskVariantRegistrar registrar(2, "write");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<write>(registrar, "write");
  }

  return Runtime::start(argc, argv);
}
