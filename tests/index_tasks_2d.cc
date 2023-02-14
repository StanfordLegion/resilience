/* Copyright 2022 Stanford University
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

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "legion_resilience.h"

using namespace ResilientLegion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INDEX_SPACE_TASK_ID,
};

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
  runtime->make_checkpointable();

  int num_points = 4;

  Rect<2> launch_bounds(Point<2>(0, 0), Point<2>(num_points - 1, num_points - 1));

  // Legion supports launching an array of tasks with a
  // single call.  We call these index tasks as we are launching
  // an array of tasks with one task for each point in the
  // array.  Index tasks are launched similar to single
  // tasks by using an index task launcher.  IndexLauncher
  // objects take the additional arguments of an ArgumentMap,
  // a TaskArgument which is a global argument that will
  // be passed to all tasks launched, and a domain describing
  // the points to be launched.
  IndexLauncher index_launcher(INDEX_SPACE_TASK_ID, launch_bounds, TaskArgument(NULL, 0),
                               ArgumentMap());
  // Index tasks are launched the same as single tasks, but
  // return a future map which will store a future for all
  // points in the index space task launch.  Application
  // tasks can either wait on the future map for all tasks
  // in the index space to finish, or it can pull out
  // individual futures for specific points on which to wait.
  FutureMap fm = runtime->execute_index_space(ctx, index_launcher);
  // Here we wait for all the futures to be ready
  fm.wait_all_results(runtime);
  // Now we can check that the future results that came back
  // from all the points in the index task are double
  // their input.

  runtime->checkpoint(ctx, task);

  long long total = 0;
  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < num_points; j++) {
      auto result = fm.get_result<long long>(Point<2>(i, j), runtime);
      total += result;
    }
  }
  assert(total == 48);
}

long long index_space_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime) {
  // The point for this task is available in the task
  // structure under the 'index_point' field.
  assert(task->index_point.get_dim() == 2);
  // Values passed through an argument map are available
  // through the local_args and local_arglen fields.
  long long x = task->index_point.point_data[0];
  long long y = task->index_point.point_data[1];
  return x + y;
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INDEX_SPACE_TASK_ID, "index_space_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<long long, index_space_task>(registrar,
                                                                   "index_space_task");
  }

  return Runtime::start(argc, argv);
}
