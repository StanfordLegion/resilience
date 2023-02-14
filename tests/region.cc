#include <signal.h>

#include <iostream>

#include "resilience.h"

using namespace ResilientLegion;

void read_region(const Task *task, const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime) {
  PhysicalRegion pr = regions[0];
  const FieldAccessor<READ_ONLY, int, 1> acc(pr, 0);
  const Rect<1> domain(0, 10);
  for (PointInRectIterator<1> pir(domain); pir(); pir++) {
    std::cout << "Data: " << acc[*pir] << std::endl;
  }
}

void write_region(const Task *task, const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime_) {
  PhysicalRegion pr = regions[0];
  const FieldAccessor<READ_WRITE, int, 1> acc(pr, 0);
  const Rect<1> domain(0, 10);
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
               Runtime *runtime) {
  runtime->make_checkpointable();

  int N = 10;
  const Rect<1> domain(0, N);
  IndexSpaceT<1> ispace = runtime->create_index_space(ctx, domain);
  FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator fal = runtime->create_field_allocator(ctx, fspace);
    fal.allocate_field(sizeof(int), 0);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, ispace, fspace);

  TaskLauncher write_launcher(2, TaskArgument());
  write_launcher.add_region_requirement(RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
  write_launcher.add_field(0, 0);
  runtime->execute_task(ctx, write_launcher);

  // Static method calls are invalid after starting the runtime...
  runtime->checkpoint(ctx, task);
  abort(Runtime::get_input_args());

  TaskLauncher read_launcher(1, TaskArgument());
  read_launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
  read_launcher.add_field(0, 0);
  runtime->execute_task(ctx, read_launcher);

  std::cout << "Done!" << std::endl;
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
