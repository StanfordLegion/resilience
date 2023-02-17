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

void read_region(const Task *task, const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime) {
  PhysicalRegion pr = regions[0];
  const FieldAccessor<READ_ONLY, int, 1> acc(pr, 0);
  DomainT<1> domain = runtime->get_index_space_domain(
      ctx, IndexSpaceT<1>(task->regions[0].region.get_index_space()));
  std::cout << "Data:";
  for (PointInDomainIterator<1> pir(domain); pir(); pir++) {
    std::cout << " " << acc[*pir];
    assert(acc[*pir] == *pir + 1);
  }
  std::cout << std::endl;
}

void write_region(const Task *task, const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime) {
  PhysicalRegion pr = regions[0];
  const FieldAccessor<READ_WRITE, int, 1> acc(pr, 0);
  DomainT<1> domain = runtime->get_index_space_domain(
      ctx, IndexSpaceT<1>(task->regions[0].region.get_index_space()));
  for (PointInDomainIterator<1> pir(domain); pir(); pir++) {
    acc[*pir] = *pir + 1;
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

  // Checkpoint between each of the major API calls below to make sure we can restore the
  // application at each state
  runtime->checkpoint(ctx, task);

  for (int trial = 0; trial < 2; ++trial) {
    int N = 10 * (trial + 1);
    std::cout << "Running trial " << trial << " with " << N << " elements" << std::endl;
    const Rect<1> domain(0, N - 1);
    IndexSpaceT<1> ispace = runtime->create_index_space(ctx, domain);
    runtime->attach_name(ispace, "test ispace");
    runtime->checkpoint(ctx, task);
    FieldSpace fspace = runtime->create_field_space(ctx);
    runtime->attach_name(fspace, "test fspace");
    runtime->checkpoint(ctx, task);
    std::cout << "Index space " << ispace << " domain "
              << runtime->get_index_space_domain(ispace) << std::endl;
    {
      FieldAllocator fal = runtime->create_field_allocator(ctx, fspace);
      fal.allocate_field(sizeof(int), 0);
    }
    runtime->checkpoint(ctx, task);
    LogicalRegion lr = runtime->create_logical_region(ctx, ispace, fspace);
    runtime->attach_name(lr, "test lr");
    runtime->checkpoint(ctx, task);

    TaskLauncher write_launcher(2, TaskArgument());
    write_launcher.add_region_requirement(
        RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
    write_launcher.add_field(0, 0);
    runtime->execute_task(ctx, write_launcher);
    runtime->checkpoint(ctx, task);

    TaskLauncher read_launcher(1, TaskArgument());
    read_launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
    read_launcher.add_field(0, 0);
    runtime->execute_task(ctx, read_launcher);
    runtime->checkpoint(ctx, task);

    runtime->destroy_logical_region(ctx, lr);
    runtime->checkpoint(ctx, task);
    runtime->destroy_field_space(ctx, fspace);
    runtime->checkpoint(ctx, task);
    runtime->destroy_index_space(ctx, ispace);
    runtime->checkpoint(ctx, task);
  }
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(0);
  {
    TaskVariantRegistrar registrar(0, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(1, "read_region");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<read_region>(registrar, "read_region");
  }
  {
    TaskVariantRegistrar registrar(2, "write_region");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<write_region>(registrar, "write_region");
  }

  return Runtime::start(argc, argv);
}
