#include <signal.h>

#include <iostream>

#include "legion.h"
#include "resilience.h"

using namespace Legion;

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

void write_region(const Task *task, const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime) {
  PhysicalRegion pr = regions[0];
  const FieldAccessor<READ_WRITE, int, 1> acc(pr, 0);
  auto domain =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(domain); pir(); pir++) {
    acc[*pir] = *pir + 1;
  }
}

void abort(InputArgs args) {
  for (int i = 1; i < args.argc; i++) {
    if (strstr(args.argv[i], "-abort")) raise(SIGSEGV);
  }
}

void top_level(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Runtime *runtime_) {
  using namespace ResilientLegion;
  using ResilientLegion::Future;
  using ResilientLegion::LogicalRegion;
  using ResilientLegion::Runtime;
  Runtime runtime__(runtime_);
  Runtime *runtime = &runtime__;

  int N = 10;
  const Rect<1> domain(0, N - 1);
  IndexSpaceT<1> ispace = runtime->create_index_space(ctx, domain);
  FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator fal = runtime->create_field_allocator(ctx, fspace);
    fal.allocate_field(sizeof(int), 0);
  }
  LogicalRegion lr0 = runtime->create_logical_region(ctx, ispace, fspace);
  LogicalRegion lr1 = runtime->create_logical_region(ctx, ispace, fspace);

  int n = 2;
  Rect<1> color_bounds(0, n - 1);
  IndexSpace cspace = runtime->create_index_space(ctx, color_bounds);

  IndexPartition ip0 = runtime->create_equal_partition(ctx, ispace, cspace);
  IndexPartition ip1 = runtime->create_equal_partition(ctx, ispace, cspace);

  LogicalPartition lp0 = runtime->get_logical_partition(ctx, lr0, ip0);
  LogicalPartition lp1 = runtime->get_logical_partition(ctx, lr1, ip1);

  LogicalRegion lsr0 = runtime->get_logical_subregion_by_color(ctx, lp0, 0);
  LogicalRegion lsr1 = runtime->get_logical_subregion_by_color(ctx, lp1, 1);

  runtime->fill_field<int>(ctx, lr0, lr0, 0, 1);
  runtime->fill_field<int>(ctx, lsr0, lr0, 0, 2);

  runtime->fill_field<int>(ctx, lr1, lr1, 0, 1);
  runtime->fill_field<int>(ctx, lsr1, lr1, 0, 3);

  runtime->checkpoint(ctx, task);

  // Invalid, actually
  abort(Runtime::get_input_args());

  TaskLauncher sum_launcher(1, TaskArgument());
  sum_launcher.add_region_requirement(RegionRequirement(lr0, READ_ONLY, EXCLUSIVE, lr0));
  sum_launcher.add_field(0, 0);
  Future sum_future = runtime->execute_task(ctx, sum_launcher);
  int sum = sum_future.get_result<int>();
  assert(sum == 15);

  sum_launcher.region_requirements.clear();
  sum_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
  sum_launcher.add_field(0, 0);
  sum_future = runtime->execute_task(ctx, sum_launcher);
  sum = sum_future.get_result<int>();
  assert(sum == 20);
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
    TaskVariantRegistrar registrar(2, "write_region");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<write_region>(registrar, "write_region");
  }

  return Runtime::start(argc, argv);
}
