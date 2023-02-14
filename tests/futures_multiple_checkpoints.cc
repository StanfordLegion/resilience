#include <signal.h>

#include <iostream>

#include "legion.h"
#include "resilience.h"

using namespace Legion;

int foo(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
        Runtime *runtime) {
  return *(int *)task->args;
}

void abort(InputArgs args) {
  for (int i = 1; i < args.argc; i++) {
    if (strstr(args.argv[i], "-abort")) raise(SIGSEGV);
  }
}

void top_level(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Legion::Runtime *runtime_) {
  using namespace ResilientLegion;
  using ResilientLegion::Future;
  using ResilientLegion::Runtime;
  Runtime runtime__(runtime_);
  Runtime *runtime = &runtime__;
  runtime->make_checkpointable();

  int x = 2;
  int y = 3;
  int z = 4;

  int rx = -1;
  int ry = -1;
  int rz = -1;

  TaskLauncher fx_launcher(1, TaskArgument(&x, sizeof(int)));
  Future fx = runtime->execute_task(ctx, fx_launcher);
  rx = fx.get_result<int>();

  runtime->checkpoint(ctx, task);
  // Invalid, actually
  abort(Legion::Runtime::get_input_args());

  TaskLauncher fy_launcher(1, TaskArgument(&y, sizeof(int)));
  Future fy = runtime->execute_task(ctx, fy_launcher);

  ry = fy.get_result<int>();

  TaskLauncher fz_launcher(1, TaskArgument(&z, sizeof(int)));
  Future fz = runtime->execute_task(ctx, fz_launcher);

  rz = fz.get_result<int>();

  runtime->checkpoint(ctx, task);

  std::cout << "rx, ry, rz : " << rx << " " << ry << " " << rz << std::endl;
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(0);
  {
    TaskVariantRegistrar registrar(0, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(1, "foo");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<int, foo>(registrar, "foo");
  }
  return Runtime::start(argc, argv);
}
