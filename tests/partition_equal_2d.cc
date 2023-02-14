#include <signal.h>

#include <iostream>

#include "legion.h"
#include "resilience.h"

using namespace Legion;

int sum(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
        Runtime *runtime) {
  PhysicalRegion pr = regions[0];
  const FieldAccessor<READ_ONLY, int, 2> acc(pr, 0);
  auto domain =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  int total = 0;
  for (PointInRectIterator<2> pir(domain); pir(); pir++) {
    total += acc[*pir];
  }
  return total;
}

void write_region(const Task *task, const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime) {
  PhysicalRegion pr = regions[0];
  const FieldAccessor<READ_WRITE, int, 2> acc(pr, 0);
  auto domain =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  for (PointInRectIterator<2> pir(domain); pir(); pir++) {
    acc[*pir] = pir->x + pir->y + 1;
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
  using ResilientLegion::FutureMap;
  using ResilientLegion::LogicalRegion;
  using ResilientLegion::Runtime;
  Runtime runtime__(runtime_);
  Runtime *runtime = &runtime__;
  runtime->make_checkpointable();

  int N = 10;
  const Rect<2> domain(Point<2>(0, 0), Point<2>(N - 1, N - 1));
  IndexSpace ispace = runtime->create_index_space(ctx, domain);
  FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator fal = runtime->create_field_allocator(ctx, fspace);
    fal.allocate_field(sizeof(int), 0);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, ispace, fspace);

  int n = 2;
  Rect<2> color_bounds(Point<2>(0, 0), Point<2>(n - 1, n - 1));
  IndexSpace cspace = runtime->create_index_space(ctx, color_bounds);

  IndexPartition ip = runtime->create_equal_partition(ctx, ispace, cspace);
  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);

  LogicalRegion lsr = runtime->get_logical_subregion_by_color(ctx, lp, Point<2>(0, 0));

  runtime->fill_field<int>(ctx, lr, lr, 0, 1);
  runtime->fill_field<int>(ctx, lsr, lr, 0, 2);

  runtime->checkpoint(ctx, task);

  // Invalid, actually
  abort(Legion::Runtime::get_input_args());

  TaskLauncher sum_launcher(1, TaskArgument());
  sum_launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
  sum_launcher.add_field(0, 0);
  Future sum_future = runtime->execute_task(ctx, sum_launcher);
  int sum = sum_future.get_result<int>();
  assert(sum == 120);
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
