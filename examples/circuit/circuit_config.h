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


#ifndef __CIRCUIT_CONFIG_H__
#define __CIRCUIT_CONFIG_H__

#include "legion.h"
#include "resilience.h"

enum {
  TOP_LEVEL_TASK_ID,
  CALC_NEW_CURRENTS_TASK_ID,
  DISTRIBUTE_CHARGE_TASK_ID,
  UPDATE_VOLTAGES_TASK_ID,
  CHECK_FIELD_TASK_ID,
#ifndef SEQUENTIAL_LOAD_CIRCUIT
  INIT_NODES_TASK_ID,
  INIT_WIRES_TASK_ID,
  INIT_LOCATION_TASK_ID,
#endif
};

enum {
  REDUCE_ID = LEGION_REDOP_SUM_FLOAT32,
};

enum {
  COLOCATION_NEXT_TAG = 1,
  COLOCATION_PREV_TAG = 2,
};

void update_mappers(Legion::Machine machine, ResilientLegion::Runtime *rt,
                    const std::set<Legion::Processor> &local_procs);

#endif // __CIRCUIT_CONFIG_H__
