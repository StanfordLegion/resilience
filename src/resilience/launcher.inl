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

namespace ResilientLegion {

TaskLauncher::TaskLauncher(void)
    : task_id(0),
      argument(UntypedBuffer()),
      predicate(Predicate::TRUE_PRED),
      map_id(0),
      tag(0),
      point(DomainPoint(0)),
      sharding_space(IndexSpace::NO_SPACE),
      static_dependences(NULL),
      enable_inlining(false),
      local_function_task(false),
      independent_requirements(false),
      elide_future_return(false),
      silence_warnings(false) {}

TaskLauncher::TaskLauncher(TaskID tid, UntypedBuffer arg, Predicate pred, MapperID mid,
                           MappingTagID t, UntypedBuffer marg, const char *prov)
    : task_id(tid),
      argument(arg),
      predicate(pred),
      map_id(mid),
      tag(t),
      map_arg(marg),
      point(DomainPoint(0)),
      sharding_space(IndexSpace::NO_SPACE),
      provenance(prov),
      static_dependences(NULL),
      enable_inlining(false),
      local_function_task(false),
      independent_requirements(false),
      elide_future_return(false),
      silence_warnings(false) {}

inline IndexSpaceRequirement &TaskLauncher::add_index_requirement(
    const IndexSpaceRequirement &req) {
  index_requirements.push_back(req);
  return index_requirements.back();
}

inline RegionRequirement &TaskLauncher::add_region_requirement(
    const RegionRequirement &req) {
  region_requirements.push_back(req);
  return region_requirements.back();
}

inline void TaskLauncher::add_field(unsigned idx, FieldID fid, bool inst) {
  region_requirements.at(idx).add_field(fid, inst);
}

inline void TaskLauncher::add_future(Future f) { futures.push_back(f); }

inline void TaskLauncher::add_grant(Grant g) { grants.push_back(g); }

inline void TaskLauncher::add_wait_barrier(PhaseBarrier bar) {
  assert(bar.exists());
  wait_barriers.push_back(bar);
}

inline void TaskLauncher::add_arrival_barrier(PhaseBarrier bar) {
  assert(bar.exists());
  arrive_barriers.push_back(bar);
}

inline void TaskLauncher::add_wait_handshake(LegionHandshake handshake) {
  wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
}

inline void TaskLauncher::add_arrival_handshake(LegionHandshake handshake) {
  arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
}

inline void TaskLauncher::set_predicate_false_future(Future f) {
  predicate_false_future = f;
}

inline void TaskLauncher::set_predicate_false_result(UntypedBuffer arg) {
  predicate_false_result = arg;
}

inline void TaskLauncher::set_independent_requirements(bool independent) {
  independent_requirements = independent;
}

inline TaskLauncher::operator Legion::TaskLauncher() const {
  Legion::TaskLauncher result;

  result.task_id = task_id;
  result.index_requirements = index_requirements;
  result.region_requirements = region_requirements;
  for (auto &future : futures) {
    result.futures.push_back(future);
  }
  result.grants = grants;
  result.wait_barriers = wait_barriers;
  result.arrive_barriers = arrive_barriers;
  result.argument = argument;
  result.predicate = predicate;
  result.map_id = map_id;
  result.tag = tag;
  result.map_arg = map_arg;
  result.point = point;
  result.sharding_space = sharding_space;
  result.predicate_false_future = predicate_false_future;
  result.predicate_false_result = predicate_false_result;
  result.provenance = provenance;
  result.static_dependences = static_dependences;
  result.enable_inlining = enable_inlining;
  result.local_function_task = local_function_task;
  result.independent_requirements = independent_requirements;
  result.elide_future_return = elide_future_return;
  result.silence_warnings = silence_warnings;

  return result;
}

inline IndexTaskLauncher::IndexTaskLauncher(void)
    : task_id(0),
      launch_domain(Domain::NO_DOMAIN),
      launch_space(IndexSpace::NO_SPACE),
      sharding_space(IndexSpace::NO_SPACE),
      global_arg(UntypedBuffer()),
      argument_map(ArgumentMap()),
      predicate(Predicate::TRUE_PRED),
      concurrent(false),
      must_parallelism(false),
      map_id(0),
      tag(0),
      static_dependences(NULL),
      enable_inlining(false),
      independent_requirements(false),
      elide_future_return(false),
      silence_warnings(false) {}

inline IndexTaskLauncher::IndexTaskLauncher(TaskID tid, Domain dom, UntypedBuffer global,
                                            ArgumentMap map, Predicate pred, bool must,
                                            MapperID mid, MappingTagID t,
                                            UntypedBuffer marg, const char *prov)
    : task_id(tid),
      launch_domain(dom),
      launch_space(IndexSpace::NO_SPACE),
      sharding_space(IndexSpace::NO_SPACE),
      global_arg(global),
      argument_map(map),
      predicate(pred),
      concurrent(false),
      must_parallelism(must),
      map_id(mid),
      tag(t),
      map_arg(marg),
      provenance(prov),
      static_dependences(NULL),
      enable_inlining(false),
      independent_requirements(false),
      elide_future_return(false),
      silence_warnings(false) {}

inline IndexTaskLauncher::IndexTaskLauncher(TaskID tid, IndexSpace space,
                                            UntypedBuffer global, ArgumentMap map,
                                            Predicate pred, bool must, MapperID mid,
                                            MappingTagID t, UntypedBuffer marg,
                                            const char *prov)
    : task_id(tid),
      launch_domain(Domain::NO_DOMAIN),
      launch_space(space),
      sharding_space(IndexSpace::NO_SPACE),
      global_arg(global),
      argument_map(map),
      predicate(pred),
      concurrent(false),
      must_parallelism(must),
      map_id(mid),
      tag(t),
      map_arg(marg),
      provenance(prov),
      static_dependences(NULL),
      enable_inlining(false),
      independent_requirements(false),
      elide_future_return(false),
      silence_warnings(false) {}

inline IndexSpaceRequirement &IndexTaskLauncher::add_index_requirement(
    const IndexSpaceRequirement &req) {
  index_requirements.push_back(req);
  return index_requirements.back();
}

inline RegionRequirement &IndexTaskLauncher::add_region_requirement(
    const RegionRequirement &req) {
  region_requirements.push_back(req);
  return region_requirements.back();
}

inline void IndexTaskLauncher::add_field(unsigned idx, FieldID fid, bool inst) {
  region_requirements.at(idx).add_field(fid, inst);
}

inline void IndexTaskLauncher::add_future(Future f) { futures.push_back(f); }

inline void IndexTaskLauncher::add_grant(Grant g) { grants.push_back(g); }

inline void IndexTaskLauncher::add_wait_barrier(PhaseBarrier bar) {
  assert(bar.exists());
  wait_barriers.push_back(bar);
}

inline void IndexTaskLauncher::add_arrival_barrier(PhaseBarrier bar) {
  assert(bar.exists());
  arrive_barriers.push_back(bar);
}

inline void IndexTaskLauncher::add_wait_handshake(LegionHandshake handshake) {
  wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
}

inline void IndexTaskLauncher::add_arrival_handshake(LegionHandshake handshake) {
  arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
}

inline void IndexTaskLauncher::set_predicate_false_future(Future f) {
  predicate_false_future = f;
}

inline void IndexTaskLauncher::set_predicate_false_result(UntypedBuffer arg) {
  predicate_false_result = arg;
}

inline void IndexTaskLauncher::set_independent_requirements(bool independent) {
  independent_requirements = independent;
}

inline IndexTaskLauncher::operator Legion::IndexTaskLauncher() const {
  Legion::IndexTaskLauncher result;

  result.task_id = task_id;
  result.launch_domain = launch_domain;
  result.launch_space = launch_space;
  result.sharding_space = sharding_space;
  result.index_requirements = index_requirements;
  result.region_requirements = region_requirements;
  for (auto &future : futures) {
    result.futures.push_back(future);
  }
  result.point_futures = point_futures;
  result.grants = grants;
  result.wait_barriers = wait_barriers;
  result.arrive_barriers = arrive_barriers;
  result.global_arg = global_arg;
  result.argument_map = argument_map;
  result.predicate = predicate;
  result.concurrent = concurrent;
  result.must_parallelism = must_parallelism;
  result.map_id = map_id;
  result.tag = tag;
  result.map_arg = map_arg;
  result.predicate_false_future = predicate_false_future;
  result.predicate_false_result = predicate_false_result;
  result.provenance = provenance;
  result.static_dependences = static_dependences;
  result.enable_inlining = enable_inlining;
  result.independent_requirements = independent_requirements;
  result.elide_future_return = elide_future_return;
  result.silence_warnings = silence_warnings;

  return result;
}

inline FillLauncher::FillLauncher(void)
    : handle(LogicalRegion::NO_REGION),
      parent(LogicalRegion::NO_REGION),
      map_id(0),
      tag(0),
      point(DomainPoint(0)),
      static_dependences(NULL),
      silence_warnings(false) {}

inline FillLauncher::FillLauncher(LogicalRegion h, LogicalRegion p, UntypedBuffer arg,
                                  Predicate pred, MapperID id, MappingTagID t,
                                  UntypedBuffer marg, const char *prov)
    : handle(h),
      parent(p),
      argument(arg),
      predicate(pred),
      map_id(id),
      tag(t),
      map_arg(marg),
      point(DomainPoint(0)),
      provenance(prov),
      static_dependences(NULL),
      silence_warnings(false) {}

inline FillLauncher::FillLauncher(LogicalRegion h, LogicalRegion p, Future f,
                                  Predicate pred, MapperID id, MappingTagID t,
                                  UntypedBuffer marg, const char *prov)
    : handle(h),
      parent(p),
      future(f),
      predicate(pred),
      map_id(id),
      tag(t),
      map_arg(marg),
      point(DomainPoint(0)),
      provenance(prov),
      static_dependences(NULL),
      silence_warnings(false) {}

inline void FillLauncher::set_argument(UntypedBuffer arg) { argument = arg; }

inline void FillLauncher::set_future(Future f) { future = f; }

inline void FillLauncher::add_field(FieldID fid) { fields.insert(fid); }

inline void FillLauncher::add_grant(Grant g) { grants.push_back(g); }

inline void FillLauncher::add_wait_barrier(PhaseBarrier pb) {
  assert(pb.exists());
  wait_barriers.push_back(pb);
}

inline void FillLauncher::add_arrival_barrier(PhaseBarrier pb) {
  assert(pb.exists());
  arrive_barriers.push_back(pb);
}

inline void FillLauncher::add_wait_handshake(LegionHandshake handshake) {
  wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
}

inline void FillLauncher::add_arrival_handshake(LegionHandshake handshake) {
  arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
}

inline FillLauncher::operator Legion::FillLauncher() const {
  Legion::FillLauncher result;

  result.handle = handle;
  result.parent = parent;
  result.argument = argument;
  result.future = future;
  result.predicate = predicate;
  result.fields = fields;
  result.grants = grants;
  result.wait_barriers = wait_barriers;
  result.arrive_barriers = arrive_barriers;
  result.map_id = map_id;
  result.tag = tag;
  result.map_arg = map_arg;
  result.point = point;
  result.sharding_space = sharding_space;
  result.provenance = provenance;
  result.static_dependences = static_dependences;
  result.silence_warnings = silence_warnings;

  return result;
}

inline IndexFillLauncher::IndexFillLauncher(void)
    : launch_domain(Domain::NO_DOMAIN),
      launch_space(IndexSpace::NO_SPACE),
      sharding_space(IndexSpace::NO_SPACE),
      region(LogicalRegion::NO_REGION),
      partition(LogicalPartition::NO_PART),
      projection(0),
      map_id(0),
      tag(0),
      static_dependences(NULL),
      silence_warnings(false) {}

inline IndexFillLauncher::IndexFillLauncher(Domain dom, LogicalRegion h, LogicalRegion p,
                                            UntypedBuffer arg, ProjectionID proj,
                                            Predicate pred, MapperID id, MappingTagID t,
                                            UntypedBuffer marg, const char *prov)
    : launch_domain(dom),
      launch_space(IndexSpace::NO_SPACE),
      partition(LogicalPartition::NO_PART),
      parent(p),
      projection(proj),
      argument(arg),
      predicate(pred),
      map_id(id),
      tag(t),
      map_arg(marg),
      provenance(prov),
      static_dependences(NULL),
      silence_warnings(false) {}

inline IndexFillLauncher::IndexFillLauncher(Domain dom, LogicalRegion h, LogicalRegion p,
                                            Future f, ProjectionID proj, Predicate pred,
                                            MapperID id, MappingTagID t,
                                            UntypedBuffer marg, const char *prov)
    : launch_domain(dom),
      launch_space(IndexSpace::NO_SPACE),
      sharding_space(IndexSpace::NO_SPACE),
      region(h),
      partition(LogicalPartition::NO_PART),
      parent(p),
      projection(proj),
      future(f),
      predicate(pred),
      map_id(id),
      tag(t),
      map_arg(marg),
      provenance(prov),
      static_dependences(NULL),
      silence_warnings(false) {}

inline IndexFillLauncher::IndexFillLauncher(IndexSpace space, LogicalRegion h,
                                            LogicalRegion p, UntypedBuffer arg,
                                            ProjectionID proj, Predicate pred,
                                            MapperID id, MappingTagID t,
                                            UntypedBuffer marg, const char *prov)
    : launch_domain(Domain::NO_DOMAIN),
      launch_space(space),
      sharding_space(IndexSpace::NO_SPACE),
      region(h),
      partition(LogicalPartition::NO_PART),
      parent(p),
      projection(proj),
      argument(arg),
      predicate(pred),
      map_id(id),
      tag(t),
      map_arg(marg),
      provenance(prov),
      static_dependences(NULL),
      silence_warnings(false) {}

inline IndexFillLauncher::IndexFillLauncher(IndexSpace space, LogicalRegion h,
                                            LogicalRegion p, Future f, ProjectionID proj,
                                            Predicate pred, MapperID id, MappingTagID t,
                                            UntypedBuffer marg, const char *prov)
    : launch_domain(Domain::NO_DOMAIN),
      launch_space(space),
      sharding_space(IndexSpace::NO_SPACE),
      region(h),
      partition(LogicalPartition::NO_PART),
      parent(p),
      projection(proj),
      future(f),
      predicate(pred),
      map_id(id),
      tag(t),
      map_arg(marg),
      provenance(prov),
      static_dependences(NULL),
      silence_warnings(false) {}

inline IndexFillLauncher::IndexFillLauncher(Domain dom, LogicalPartition h,
                                            LogicalRegion p, UntypedBuffer arg,
                                            ProjectionID proj, Predicate pred,
                                            MapperID id, MappingTagID t,
                                            UntypedBuffer marg, const char *prov)
    : launch_domain(dom),
      launch_space(IndexSpace::NO_SPACE),
      sharding_space(IndexSpace::NO_SPACE),
      region(LogicalRegion::NO_REGION),
      partition(h),
      parent(p),
      projection(proj),
      argument(arg),
      predicate(pred),
      map_id(id),
      tag(t),
      map_arg(marg),
      provenance(prov),
      static_dependences(NULL),
      silence_warnings(false) {}

inline IndexFillLauncher::IndexFillLauncher(Domain dom, LogicalPartition h,
                                            LogicalRegion p, Future f, ProjectionID proj,
                                            Predicate pred, MapperID id, MappingTagID t,
                                            UntypedBuffer marg, const char *prov)
    : launch_domain(dom),
      launch_space(IndexSpace::NO_SPACE),
      sharding_space(IndexSpace::NO_SPACE),
      region(LogicalRegion::NO_REGION),
      partition(h),
      parent(p),
      projection(proj),
      future(f),
      predicate(pred),
      map_id(id),
      tag(t),
      map_arg(marg),
      provenance(prov),
      static_dependences(NULL),
      silence_warnings(false) {}

inline IndexFillLauncher::IndexFillLauncher(IndexSpace space, LogicalPartition h,
                                            LogicalRegion p, UntypedBuffer arg,
                                            ProjectionID proj, Predicate pred,
                                            MapperID id, MappingTagID t,
                                            UntypedBuffer marg, const char *prov)
    : launch_domain(Domain::NO_DOMAIN),
      launch_space(space),
      sharding_space(IndexSpace::NO_SPACE),
      region(LogicalRegion::NO_REGION),
      partition(h),
      parent(p),
      projection(proj),
      argument(arg),
      predicate(pred),
      map_id(id),
      tag(t),
      map_arg(marg),
      provenance(prov),
      static_dependences(NULL),
      silence_warnings(false) {}

inline IndexFillLauncher::IndexFillLauncher(IndexSpace space, LogicalPartition h,
                                            LogicalRegion p, Future f, ProjectionID proj,
                                            Predicate pred, MapperID id, MappingTagID t,
                                            UntypedBuffer marg, const char *prov)
    : launch_domain(Domain::NO_DOMAIN),
      launch_space(space),
      sharding_space(IndexSpace::NO_SPACE),
      region(LogicalRegion::NO_REGION),
      partition(h),
      parent(p),
      projection(proj),
      future(f),
      predicate(pred),
      map_id(id),
      tag(t),
      map_arg(marg),
      provenance(prov),
      static_dependences(NULL),
      silence_warnings(false) {}

inline void IndexFillLauncher::set_argument(UntypedBuffer arg) { argument = arg; }

inline void IndexFillLauncher::set_future(Future f) { future = f; }

inline void IndexFillLauncher::add_field(FieldID fid) { fields.insert(fid); }

inline void IndexFillLauncher::add_grant(Grant g) { grants.push_back(g); }

inline void IndexFillLauncher::add_wait_barrier(PhaseBarrier pb) {
  assert(pb.exists());
  wait_barriers.push_back(pb);
}

inline void IndexFillLauncher::add_arrival_barrier(PhaseBarrier pb) {
  assert(pb.exists());
  arrive_barriers.push_back(pb);
}

inline void IndexFillLauncher::add_wait_handshake(LegionHandshake handshake) {
  wait_barriers.push_back(handshake.get_legion_wait_phase_barrier());
}

inline void IndexFillLauncher::add_arrival_handshake(LegionHandshake handshake) {
  arrive_barriers.push_back(handshake.get_legion_arrive_phase_barrier());
}

inline IndexFillLauncher::operator Legion::IndexFillLauncher() const {
  Legion::IndexFillLauncher result;

  result.launch_domain = launch_domain;
  result.launch_space = launch_space;
  result.sharding_space = sharding_space;
  result.region = region;
  result.partition = partition;
  result.parent = parent;
  result.projection = projection;
  result.argument = argument;
  result.future = future;
  result.predicate = predicate;
  result.fields = fields;
  result.grants = grants;
  result.wait_barriers = wait_barriers;
  result.arrive_barriers = arrive_barriers;
  result.map_id = map_id;
  result.tag = tag;
  result.map_arg = map_arg;
  result.provenance = provenance;
  result.static_dependences = static_dependences;
  result.silence_warnings = silence_warnings;

  return result;
}

inline TimingLauncher::TimingLauncher(TimingMeasurement m) : measurement(m) {}

inline void TimingLauncher::add_precondition(const Future &f) { preconditions.insert(f); }

inline TimingLauncher::operator Legion::TimingLauncher() const {
  Legion::TimingLauncher result(measurement);

  for (auto &future : preconditions) {
    result.preconditions.insert(future);
  }
  result.provenance = provenance;

  return result;
}

inline TunableLauncher::TunableLauncher(TunableID tid, MapperID m, MappingTagID t,
                                        size_t return_size)
    : tunable(tid), mapper(m), tag(t), return_type_size(return_size) {}

inline TunableLauncher::operator Legion::TunableLauncher() const {
  Legion::TunableLauncher result(tunable, mapper, tag, return_type_size);

  result.arg = arg;
  for (auto &future : futures) {
    result.futures.push_back(future);
  }
  result.provenance = provenance;

  return result;
}

}  // namespace ResilientLegion
