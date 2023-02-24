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

#ifndef RESILIENCE_LAUNCHER_H
#define RESILIENCE_LAUNCHER_H

#include "legion.h"
#include "resilience/future.h"
#include "resilience/types.h"

namespace ResilientLegion {

// Wrappers for Legion launchers

struct TaskLauncher {
public:
  inline TaskLauncher(void);
  inline TaskLauncher(TaskID tid, UntypedBuffer arg,
                      Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
                      MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
                      const char *provenance = "");
  inline IndexSpaceRequirement &add_index_requirement(const IndexSpaceRequirement &req);
  inline RegionRequirement &add_region_requirement(const RegionRequirement &req);
  inline void add_field(unsigned idx, FieldID fid, bool inst = true);
  inline void add_future(Future f);
  inline void add_grant(Grant g);
  inline void add_wait_barrier(PhaseBarrier bar);
  inline void add_arrival_barrier(PhaseBarrier bar);
  inline void add_wait_handshake(LegionHandshake handshake);
  inline void add_arrival_handshake(LegionHandshake handshake);
  inline void set_predicate_false_future(Future f);
  inline void set_predicate_false_result(UntypedBuffer arg);
  inline void set_independent_requirements(bool independent);

public:
  TaskID task_id;
  std::vector<IndexSpaceRequirement> index_requirements;
  std::vector<RegionRequirement> region_requirements;
  std::vector<Future> futures;
  std::vector<Grant> grants;
  std::vector<PhaseBarrier> wait_barriers;
  std::vector<PhaseBarrier> arrive_barriers;
  UntypedBuffer argument;
  Predicate predicate;
  MapperID map_id;
  MappingTagID tag;
  UntypedBuffer map_arg;
  DomainPoint point;
  IndexSpace sharding_space;
  Future predicate_false_future;
  UntypedBuffer predicate_false_result;
  std::string provenance;
  const std::vector<StaticDependence> *static_dependences;
  bool enable_inlining;
  bool local_function_task;
  bool independent_requirements;
  bool elide_future_return;
  bool silence_warnings;

private:
  // This is dangerous because Futures could escape through here
  inline operator Legion::TaskLauncher() const;
  friend class Runtime;
};

struct IndexTaskLauncher {
public:
  inline IndexTaskLauncher(void);
  inline IndexTaskLauncher(TaskID tid, Domain domain, UntypedBuffer global_arg,
                           ArgumentMap map, Predicate pred = Predicate::TRUE_PRED,
                           bool must = false, MapperID id = 0, MappingTagID tag = 0,
                           UntypedBuffer map_arg = UntypedBuffer(),
                           const char *provenance = "");
  inline IndexTaskLauncher(TaskID tid, IndexSpace launch_space, UntypedBuffer global_arg,
                           ArgumentMap map, Predicate pred = Predicate::TRUE_PRED,
                           bool must = false, MapperID id = 0, MappingTagID tag = 0,
                           UntypedBuffer map_arg = UntypedBuffer(),
                           const char *provenance = "");
  inline IndexSpaceRequirement &add_index_requirement(const IndexSpaceRequirement &req);
  inline RegionRequirement &add_region_requirement(const RegionRequirement &req);
  inline void add_field(unsigned idx, FieldID fid, bool inst = true);
  inline void add_future(Future f);
  inline void add_grant(Grant g);
  inline void add_wait_barrier(PhaseBarrier bar);
  inline void add_arrival_barrier(PhaseBarrier bar);
  inline void add_wait_handshake(LegionHandshake handshake);
  inline void add_arrival_handshake(LegionHandshake handshake);
  inline void set_predicate_false_future(Future f);
  inline void set_predicate_false_result(UntypedBuffer arg);
  inline void set_independent_requirements(bool independent);

public:
  TaskID task_id;
  Domain launch_domain;
  IndexSpace launch_space;
  IndexSpace sharding_space;
  std::vector<IndexSpaceRequirement> index_requirements;
  std::vector<RegionRequirement> region_requirements;
  std::vector<Future> futures;
  std::vector<ArgumentMap> point_futures;
  std::vector<Grant> grants;
  std::vector<PhaseBarrier> wait_barriers;
  std::vector<PhaseBarrier> arrive_barriers;
  UntypedBuffer global_arg;
  ArgumentMap argument_map;
  Predicate predicate;
  bool concurrent;
  bool must_parallelism;
  MapperID map_id;
  MappingTagID tag;
  UntypedBuffer map_arg;
  Future predicate_false_future;
  UntypedBuffer predicate_false_result;
  std::string provenance;
  const std::vector<StaticDependence> *static_dependences;
  bool enable_inlining;
  bool independent_requirements;
  bool elide_future_return;
  bool silence_warnings;

private:
  // This is dangerous because Futures could escape through here
  inline operator Legion::IndexTaskLauncher() const;
  friend class Runtime;
};

typedef IndexTaskLauncher IndexLauncher;

struct FillLauncher {
public:
  inline FillLauncher(void);
  inline FillLauncher(LogicalRegion handle, LogicalRegion parent, UntypedBuffer arg,
                      Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
                      MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
                      const char *provenance = "");
  inline FillLauncher(LogicalRegion handle, LogicalRegion parent, Future f,
                      Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
                      MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
                      const char *provenance = "");
  inline void set_argument(UntypedBuffer arg);
  inline void set_future(Future f);
  inline void add_field(FieldID fid);
  inline void add_grant(Grant g);
  inline void add_wait_barrier(PhaseBarrier bar);
  inline void add_arrival_barrier(PhaseBarrier bar);
  inline void add_wait_handshake(LegionHandshake handshake);
  inline void add_arrival_handshake(LegionHandshake handshake);

public:
  LogicalRegion handle;
  LogicalRegion parent;
  UntypedBuffer argument;
  Future future;
  Predicate predicate;
  std::set<FieldID> fields;
  std::vector<Grant> grants;
  std::vector<PhaseBarrier> wait_barriers;
  std::vector<PhaseBarrier> arrive_barriers;
  MapperID map_id;
  MappingTagID tag;
  UntypedBuffer map_arg;
  DomainPoint point;
  IndexSpace sharding_space;
  std::string provenance;
  const std::vector<StaticDependence> *static_dependences;
  bool silence_warnings;

private:
  // This is dangerous because Futures could escape through here
  inline operator Legion::FillLauncher() const;
  friend class Runtime;
};

struct IndexFillLauncher {
public:
  inline IndexFillLauncher(void);
  inline IndexFillLauncher(Domain domain, LogicalRegion handle, LogicalRegion parent,
                           UntypedBuffer arg, ProjectionID projection = 0,
                           Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
                           MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
                           const char *provenance = "");
  inline IndexFillLauncher(Domain domain, LogicalRegion handle, LogicalRegion parent,
                           Future f, ProjectionID projection = 0,
                           Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
                           MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
                           const char *provenance = "");
  inline IndexFillLauncher(IndexSpace space, LogicalRegion handle, LogicalRegion parent,
                           UntypedBuffer arg, ProjectionID projection = 0,
                           Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
                           MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
                           const char *provenance = "");
  inline IndexFillLauncher(IndexSpace space, LogicalRegion handle, LogicalRegion parent,
                           Future f, ProjectionID projection = 0,
                           Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
                           MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
                           const char *provenance = "");
  inline IndexFillLauncher(Domain domain, LogicalPartition handle, LogicalRegion parent,
                           UntypedBuffer arg, ProjectionID projection = 0,
                           Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
                           MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
                           const char *provenance = "");
  inline IndexFillLauncher(Domain domain, LogicalPartition handle, LogicalRegion parent,
                           Future f, ProjectionID projection = 0,
                           Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
                           MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
                           const char *provenance = "");
  inline IndexFillLauncher(IndexSpace space, LogicalPartition handle,
                           LogicalRegion parent, UntypedBuffer arg,
                           ProjectionID projection = 0,
                           Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
                           MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
                           const char *provenance = "");
  inline IndexFillLauncher(IndexSpace space, LogicalPartition handle,
                           LogicalRegion parent, Future f, ProjectionID projection = 0,
                           Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
                           MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
                           const char *provenance = "");
  inline void set_argument(UntypedBuffer arg);
  inline void set_future(Future f);
  inline void add_field(FieldID fid);
  inline void add_grant(Grant g);
  inline void add_wait_barrier(PhaseBarrier bar);
  inline void add_arrival_barrier(PhaseBarrier bar);
  inline void add_wait_handshake(LegionHandshake handshake);
  inline void add_arrival_handshake(LegionHandshake handshake);

public:
  Domain launch_domain;
  IndexSpace launch_space;
  IndexSpace sharding_space;
  LogicalRegion region;
  LogicalPartition partition;
  LogicalRegion parent;
  ProjectionID projection;
  UntypedBuffer argument;
  Future future;
  Predicate predicate;
  std::set<FieldID> fields;
  std::vector<Grant> grants;
  std::vector<PhaseBarrier> wait_barriers;
  std::vector<PhaseBarrier> arrive_barriers;
  MapperID map_id;
  MappingTagID tag;
  UntypedBuffer map_arg;
  std::string provenance;
  const std::vector<StaticDependence> *static_dependences;
  bool silence_warnings;

private:
  // This is dangerous because Futures could escape through here
  inline operator Legion::IndexFillLauncher() const;
  friend class Runtime;
};

struct TimingLauncher {
public:
  inline TimingLauncher(TimingMeasurement measurement);
  inline void add_precondition(const Future &f);

public:
  TimingMeasurement measurement;
  std::set<Future> preconditions;
  std::string provenance;

private:
  // This is dangerous because Futures could escape through here
  inline operator Legion::TimingLauncher() const;
  friend class Runtime;
};

struct TunableLauncher {
public:
  inline TunableLauncher(TunableID tid, MapperID mapper = 0, MappingTagID tag = 0,
                         size_t return_type_size = SIZE_MAX);

public:
  TunableID tunable;
  MapperID mapper;
  MappingTagID tag;
  UntypedBuffer arg;
  std::vector<Future> futures;
  size_t return_type_size;
  std::string provenance;

private:
  // This is dangerous because Futures could escape through here
  inline operator Legion::TunableLauncher() const;
  friend class Runtime;
};

}  // namespace ResilientLegion

#endif  // RESILIENCE_LAUNCHER_H
