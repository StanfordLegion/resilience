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

#include <signal.h>

#include <iostream>

#include "resilience.h"

using namespace ResilientLegion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  MAKE_BOOL_TASK_ID,
  MAKE_INT_TASK_ID,
};

bool make_bool(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Runtime *runtime) {
  bool arg = *(bool *)task->args;
  return arg;
}

int make_int(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
             Runtime *runtime) {
  int arg = *(int *)task->args;
  return arg;
}

void abort(InputArgs args) {
  for (int i = 1; i < args.argc; i++) {
    if (strstr(args.argv[i], "-abort")) raise(SIGSEGV);
  }
}

void top_level(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
               Runtime *runtime) {
  runtime->enable_checkpointing(ctx);

  for (int interval = 4; interval >= 1; --interval) {
    for (int i = 0; i < interval; ++i) {
      bool t = true;
      TaskLauncher b1_launcher(MAKE_BOOL_TASK_ID, TaskArgument(&t, sizeof(t)));
      Future fb1 = runtime->execute_task(ctx, b1_launcher);
      runtime->checkpoint(ctx);
      bool b1 = fb1.get_result<bool>();
      assert(b1);
    }

    for (int i = 0; i < interval; ++i) {
      bool f = false;
      TaskLauncher b2_launcher(MAKE_BOOL_TASK_ID, TaskArgument(&f, sizeof(f)));
      Future fb2 = runtime->execute_task(ctx, b2_launcher);
      runtime->checkpoint(ctx);
      bool b2 = fb2.get_result<bool>();
      assert(!b2);
    }

    for (int i = 0; i < interval; ++i) {
      int v = 123;
      TaskLauncher i1_launcher(MAKE_INT_TASK_ID, TaskArgument(&v, sizeof(v)));
      Future fi1 = runtime->execute_task(ctx, i1_launcher);
      runtime->checkpoint(ctx);
      int i1 = fi1.get_result<int>();
      assert(i1 == v);
    }

    for (int i = 0; i < interval; ++i) {
      int v = 456;
      TaskLauncher i2_launcher(MAKE_INT_TASK_ID, TaskArgument(&v, sizeof(v)));
      Future fi2 = runtime->execute_task(ctx, i2_launcher);
      runtime->checkpoint(ctx);
      int i2 = fi2.get_result<int>();
      assert(i2 == v);
    }
  }
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
    TaskVariantRegistrar registrar(MAKE_BOOL_TASK_ID, "make_bool");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<bool, make_bool>(registrar, "make_bool");
  }
  {
    TaskVariantRegistrar registrar(MAKE_INT_TASK_ID, "make_int");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int, make_int>(registrar, "make_int");
  }
  return Runtime::start(argc, argv);
}
