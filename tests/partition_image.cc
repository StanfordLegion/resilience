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

void write(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
           Runtime *runtime) {
  PhysicalRegion pr = regions[0];
  const FieldAccessor<READ_WRITE, Point<1>, 1> acc(pr, 1);
  auto domain =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(domain); pir(); pir++) {
    acc[*pir] = Point<1>(*pir % 5);
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
  IndexSpaceT<1> index_space = runtime->create_index_space(ctx, domain);
  FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator fal = runtime->create_field_allocator(ctx, fspace);
    fal.allocate_field(sizeof(Point<1>), 0);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, index_space, fspace);

  TaskLauncher write_launcher(2, TaskArgument());
  write_launcher.add_region_requirement(
      RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
  write_launcher.add_field(0, 0);
  runtime->execute_task(ctx, write_launcher);

  int n = 1;
  IndexSpace color_space = runtime->create_index_space(ctx, Rect<1>(0, n));
  IndexPartition ip = runtime->create_equal_partition(ctx, index_space, color_space);
  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);

  IndexSpaceT<1> index_space_cpy = runtime->create_index_space(ctx, domain);

  IndexPartition ip_cpy =
      runtime_->create_partition_by_image(ctx, index_space_cpy, lp, lr, 0, color_space);

  // Elliott: TODO?
  // LogicalRegion
  // LogicalPartition lp_cpy = runtime->get_logical_partition(ctx,
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
