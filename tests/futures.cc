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
               Runtime *runtime) {
  runtime->enable_checkpointing();

  int x = 2;
  int y = 3;

  TaskLauncher fx_launcher(1, TaskArgument(&x, sizeof(int)));
  Future fx = runtime->execute_task(ctx, fx_launcher);
  int rx = fx.get_result<int>();

  TaskLauncher fy_launcher(1, TaskArgument(&y, sizeof(int)));
  Future fy = runtime->execute_task(ctx, fy_launcher);

  runtime->checkpoint(ctx, task);
  // Invalid, actually
  abort(Runtime::get_input_args());

  int ry = fy.get_result<int>();

  std::cout << "rx, ry : " << rx << " " << ry << std::endl;
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
