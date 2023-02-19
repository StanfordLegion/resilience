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

#include "resilience.h"

using namespace ResilientLegion;

static Logger log_resilience("resilience");

bool Runtime::initial_replay(false);
resilient_tag_t Runtime::initial_checkpoint_tag(0);
TaskID Runtime::write_checkpoint_task_id;

Runtime::Runtime(Legion::Runtime *lrt_)
    : lrt(lrt_),
      enabled(false),
      replay(false),
      api_tag(0),
      future_tag(0),
      future_map_tag(0),
      index_space_tag(0),
      region_tag(0),
      partition_tag(0),
      checkpoint_tag(0),
      max_api_tag(0),
      max_future_tag(0),
      max_future_map_tag(0),
      max_index_space_tag(0),
      max_region_tag(0),
      max_partition_tag(0),
      max_checkpoint_tag(0) {}

bool Runtime::skip_api_call() {
  bool skip = replay && api_tag < max_api_tag;
  api_tag++;
  return skip;
}

void Runtime::attach_name(FieldSpace handle, const char *name, bool is_mutable) {
  lrt->attach_name(handle, name, is_mutable);
}

void Runtime::attach_name(FieldSpace handle, FieldID fid, const char *name,
                          bool is_mutable) {
  lrt->attach_name(handle, fid, name, is_mutable);
}

void Runtime::attach_name(IndexSpace handle, const char *name, bool is_mutable) {
  lrt->attach_name(handle, name, is_mutable);
}

void Runtime::attach_name(LogicalRegion handle, const char *name, bool is_mutable) {
  lrt->attach_name(handle, name, is_mutable);
}

void Runtime::attach_name(IndexPartition handle, const char *name, bool is_mutable) {
  if (!enabled) {
    lrt->attach_name(handle, name, is_mutable);
    return;
  }

  if (skip_api_call()) return;
  lrt->attach_name(handle, name, is_mutable);
}

void Runtime::attach_name(LogicalPartition handle, const char *name, bool is_mutable) {
  lrt->attach_name(handle, name, is_mutable);
}

void Runtime::issue_execution_fence(Context ctx, const char *provenance) {
  if (skip_api_call()) return;
  lrt->issue_execution_fence(ctx, provenance);
}

void Runtime::issue_copy_operation(Context ctx, const CopyLauncher &launcher) {
  if (skip_api_call()) return;
  lrt->issue_copy_operation(ctx, launcher);
}

void Runtime::issue_copy_operation(Context ctx, const IndexCopyLauncher &launcher) {
  if (skip_api_call()) return;
  lrt->issue_copy_operation(ctx, launcher);
}

void Runtime::set_registration_callback(RegistrationCallbackFnptr callback) {
  Legion::Runtime::set_registration_callback(callback);
}

void callback_wrapper(const RegistrationCallbackArgs &args) {
  auto FUNC = *static_cast<void (**)(Machine, Runtime *, const std::set<Processor> &)>(
      args.buffer.get_ptr());
  Runtime new_runtime_(args.runtime);
  Runtime *new_runtime = &new_runtime_;
  FUNC(args.machine, new_runtime, args.local_procs);
}

void Runtime::add_registration_callback(void (*FUNC)(Machine, Runtime *,
                                                     const std::set<Processor> &),
                                        bool dedup, size_t dedup_tag) {
  auto fptr = &FUNC;
  UntypedBuffer buffer(fptr, sizeof(fptr));
  Legion::Runtime::add_registration_callback(callback_wrapper, buffer, dedup, dedup_tag);
}

const InputArgs &Runtime::get_input_args(void) {
  return Legion::Runtime::get_input_args();
}

void Runtime::set_top_level_task_id(TaskID top_id) {
  Legion::Runtime::set_top_level_task_id(top_id);
}

void Runtime::preregister_projection_functor(ProjectionID pid,
                                             ProjectionFunctor *functor) {
  Legion::Runtime::preregister_projection_functor(pid, functor);
}

void Runtime::preregister_sharding_functor(ShardingID sid, ShardingFunctor *functor) {
  Legion::Runtime::preregister_sharding_functor(sid, functor);
}

LayoutConstraintID Runtime::preregister_layout(const LayoutConstraintRegistrar &registrar,
                                               LayoutConstraintID layout_id) {
  return Legion::Runtime::preregister_layout(registrar, layout_id);
}

static void write_checkpoint(const Task *task, const std::vector<PhysicalRegion> &regions,
                             Context ctx, Legion::Runtime *runtime) {
  resilient_tag_t checkpoint_tag = task->futures[0].get_result<resilient_tag_t>();
  std::string serialized_data(
      static_cast<const char *>(task->futures[1].get_untyped_pointer()),
      task->futures[1].get_untyped_size());
  std::string file_name = "checkpoint." + std::to_string(checkpoint_tag);
  file_name += ".dat";
  log_resilience.info() << "write_checkpoint: File name is " << file_name;
  std::ofstream file(file_name, std::ios::binary);
  if (!file) {
    log_resilience.error() << "unable to open file '" << file_name
                           << "': " << strerror(errno);
    abort();
  }
  file << serialized_data;
  file.close();
  if (!file) {
    log_resilience.error() << "error in closing file '" << file_name
                           << "': " << strerror(errno);
    abort();
  }
}

int Runtime::start(int argc, char **argv, bool background, bool supply_default_mapper) {
#ifndef NDEBUG
  bool check = false;
#endif

  // FIXME: filter out these arguments so applications don't need to see them
  for (int i = 1; i < argc; i++) {
    if (strstr(argv[i], "-replay")) initial_replay = true;
    if (strstr(argv[i], "-cpt")) {
#ifndef NDEBUG
      check = true;
#endif
      initial_checkpoint_tag = atoi(argv[++i]);
    }
  }

  if (initial_replay) {
    assert(check);
  }

  {
    write_checkpoint_task_id = Legion::Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(write_checkpoint_task_id, "write_checkpoint");
    ProcessorConstraint pc;
    pc.add_kind(Processor::LOC_PROC);
    pc.add_kind(Processor::IO_PROC);
    registrar.add_constraint(pc);
    registrar.set_leaf();
    Legion::Runtime::preregister_task_variant<write_checkpoint>(registrar,
                                                                "write_checkpoint");
  }

  return Legion::Runtime::start(argc, argv, background, supply_default_mapper);
}

bool Runtime::is_partition_eligible(IndexPartition ip) {
  auto ip_state = ipartition_state.find(ip);
  if (ip_state != ipartition_state.end()) {
    return ip_state->second.eligible;
  }

  bool eligible =
      lrt->is_index_partition_disjoint(ip) && lrt->is_index_partition_complete(ip);
  if (eligible) {
    IndexSpace ispace = lrt->get_parent_index_space(ip);
    while (lrt->has_parent_index_partition(ispace)) {
      IndexPartition partition = lrt->get_parent_index_partition(ispace);
      ispace = lrt->get_parent_index_space(partition);
      // We do not require parents to be disjoint, because we can write overlapping data,
      // as long as we're sure it's complete.
      if (!lrt->is_index_partition_complete(partition)) {
        eligible = false;
        break;
      }
    }
  }
  ipartition_state[ip].eligible = eligible;
  return eligible;
}

void Runtime::track_region_state(const RegionRequirement &rr) {
  auto region_tag = region_tags.at(rr.parent);
  auto &lr_state = region_tree_state.at(region_tag);

  // If this access is on a disjoint and complete partition, track it; it's probably a
  // good partition to save.
  if (rr.handle_type == LEGION_PARTITION_PROJECTION &&
      !(rr.privilege == LEGION_NO_ACCESS || rr.privilege == LEGION_REDUCE)) {
    LogicalPartition lp = rr.partition;
    IndexPartition ip = lp.get_index_partition();
    if (is_partition_eligible(ip)) {
      LogicalRegion parent = lrt->get_parent_logical_region(lp);
      lr_state.recent_partitions[parent] = lp;
    }
  }
}

FutureMap Runtime::execute_index_space(Context ctx, const IndexTaskLauncher &launcher) {
  if (!enabled) {
    Legion::FutureMap lfm = lrt->execute_index_space(ctx, launcher);
    FutureMap rfm;
    if (launcher.launch_domain == Domain::NO_DOMAIN)
      return FutureMap(lrt->get_index_space_domain(launcher.launch_space), lfm);
    else
      return FutureMap(launcher.launch_domain, lfm);
  }

  if (replay && future_map_tag < max_future_map_tag) {
    log_resilience.info() << "execute_index_space: no-op for replay, tag "
                          << future_map_tag;
    return future_maps.at(future_map_tag++);
  }

  for (auto &rr : launcher.region_requirements) {
    track_region_state(rr);
  }

  Legion::FutureMap fm = lrt->execute_index_space(ctx, launcher);

  FutureMap rfm;
  if (launcher.launch_domain == Domain::NO_DOMAIN)
    rfm = FutureMap(lrt->get_index_space_domain(launcher.launch_space), fm);
  else
    rfm = FutureMap(launcher.launch_domain, fm);

  future_maps.push_back(rfm);
  future_map_tag++;
  return rfm;
}

Future Runtime::execute_index_space(Context ctx, const IndexTaskLauncher &launcher,
                                    ReductionOpID redop, bool deterministic) {
  if (!enabled) {
    return lrt->execute_index_space(ctx, launcher, redop, deterministic);
  }

  if (replay && future_tag < max_future_tag) {
    log_resilience.info() << "execute_index_space: no-op for replay, tag "
                          << future_map_tag;
    return futures.at(future_tag++);
  }

  for (auto &rr : launcher.region_requirements) {
    track_region_state(rr);
  }

  Future f = lrt->execute_index_space(ctx, launcher, redop, deterministic);
  futures.push_back(f);
  future_tag++;
  return f;
}

Future Runtime::execute_task(Context ctx, TaskLauncher launcher) {
  if (!enabled) {
    return lrt->execute_task(ctx, launcher);
  }

  if (replay && future_tag < max_future_tag) {
    log_resilience.info() << "execute_task: no-op for replay, tag " << future_tag;
    /* It is ok to return an empty ResilentFuture because get_result knows to
     * fetch the actual result from Runtime.futures by looking at the
     * tag. get_result should never be called on an empty Future.
     */
    return futures.at(future_tag++);
  }
  log_resilience.info() << "execute_task: launching task_id " << launcher.task_id;

  for (auto &rr : launcher.region_requirements) {
    track_region_state(rr);
  }

  Future ft = lrt->execute_task(ctx, launcher);
  futures.push_back(ft);
  future_tag++;
  return ft;
}

Domain Runtime::get_index_space_domain(Context ctx, IndexSpace handle) {
  return lrt->get_index_space_domain(ctx, handle);
}

Domain Runtime::get_index_space_domain(IndexSpace handle) {
  return lrt->get_index_space_domain(handle);
}

Domain Runtime::get_index_partition_color_space(Context ctx, IndexPartition p) {
  return lrt->get_index_partition_color_space(ctx, p);
}

Future Runtime::get_current_time(Context ctx, Future precondition) {
  if (!enabled) {
    return lrt->get_current_time(ctx, precondition.lft);
  }

  if (replay && future_tag < max_future_tag) {
    return futures.at(future_tag++);
  }

  Future ft = lrt->get_current_time(ctx, precondition);
  futures.push_back(ft);
  future_tag++;
  return ft;
}

Future Runtime::get_current_time_in_microseconds(Context ctx, Future precondition) {
  if (!enabled) {
    return lrt->get_current_time_in_microseconds(ctx, precondition.lft);
  }

  if (replay && future_tag < max_future_tag) {
    return futures.at(future_tag++);
  }

  Future ft = lrt->get_current_time_in_microseconds(ctx, precondition);
  futures.push_back(ft);
  future_tag++;
  return ft;
}

Predicate Runtime::create_predicate(Context ctx, const Future &f) {
  if (!enabled) {
    return lrt->create_predicate(ctx, f);
  }

  // FIXME (Elliott): Future value escapes
  return lrt->create_predicate(ctx, f.lft);
}

Predicate Runtime::create_predicate(Context ctx, const PredicateLauncher &launcher) {
  if (!enabled) {
    return lrt->create_predicate(ctx, launcher);
  }

  return lrt->create_predicate(ctx, launcher);
}

Predicate Runtime::predicate_not(Context ctx, const Predicate &p) {
  if (!enabled) {
    return lrt->predicate_not(ctx, p);
  }

  return lrt->predicate_not(ctx, p);
}

Future Runtime::get_predicate_future(Context ctx, const Predicate &p) {
  if (!enabled) {
    return lrt->get_predicate_future(ctx, p);
  }

  if (replay && future_tag < max_future_tag) {
    return futures.at(future_tag++);
  }

  Future rf = lrt->get_predicate_future(ctx, p);
  futures.push_back(rf);
  future_tag++;
  return rf;
}

FieldSpace Runtime::create_field_space(Context ctx) {
  return lrt->create_field_space(ctx);
}

FieldAllocator Runtime::create_field_allocator(Context ctx, FieldSpace handle) {
  return lrt->create_field_allocator(ctx, handle);
}

void Runtime::initialize_region(Context ctx, const LogicalRegion lr) {
  FieldSpace fspace = lr.get_field_space();
  std::vector<FieldID> fids;
  lrt->get_field_space_fields(fspace, fids);

  size_t max_bytes = 0;
  for (auto &fid : fids) {
    size_t bytes = lrt->get_field_size(fspace, fid);
    max_bytes = std::max(bytes, max_bytes);
  }
  std::vector<uint8_t> buffer(max_bytes, 0);
  for (auto &fid : fids) {
    size_t bytes = lrt->get_field_size(fspace, fid);
    lrt->fill_field(ctx, lr, lr, fid, buffer.data(), bytes);
  }
}

LogicalRegion Runtime::create_logical_region(Context ctx, IndexSpace index,
                                             FieldSpace fields, bool task_local,
                                             const char *provenance) {
  LogicalRegion lr =
      lrt->create_logical_region(ctx, index, fields, task_local, provenance);
  if (!enabled) {
    return lr;
  }

  // Region restored in replay:
  if (replay && region_tag < max_region_tag) {
    // Nothing to do here. No need to initialize, we'll no-op any operations that touch
    // this region before the checkpoint.

    // Note: we create this region even if it's already destroyed. While there are some
    // API calls we can no-op (like attach_name), there are others that make this more
    // tricky (like get_index_space_domain) and it's just easier to go through the full
    // object lifecycle.
  } else {
    // New region. Construct its state:
    state.region_state.emplace_back();

    // We initialize the data here to ensure we will never hit uninitialized data later.
    initialize_region(ctx, lr);
  }

  assert(regions.size() == region_tag);
  regions.push_back(lr);
  region_tree_state.emplace_back();
  assert(region_tree_state.size() == regions.size());
  assert(regions.size() <= state.region_state.size());
  region_tags[lr] = region_tag;
  region_tag++;
  return lr;
}

/* Inline mappings need to be disallowed */
PhysicalRegion Runtime::map_region(Context ctx, const InlineLauncher &launcher) {
  if (!enabled) {
    return lrt->map_region(ctx, launcher);
  }

  log_resilience.error() << "Inline mappings are not permitted in checkpointed tasks";
  abort();
}

void Runtime::unmap_region(Context ctx, PhysicalRegion region) {
  return lrt->unmap_region(ctx, region);
}

void Runtime::destroy_index_space(Context ctx, IndexSpace handle) {
  lrt->destroy_index_space(ctx, handle);
}

void Runtime::destroy_field_space(Context ctx, FieldSpace handle) {
  lrt->destroy_field_space(ctx, handle);
}

void Runtime::destroy_logical_region(Context ctx, LogicalRegion handle) {
  if (!enabled) {
    lrt->destroy_logical_region(ctx, handle);
    return;
  }

  auto region_tag = region_tags.at(handle);
  auto &lr_state = state.region_state.at(region_tag);
  lr_state.destroyed = true;
  lrt->destroy_logical_region(ctx, handle);
}

void Runtime::destroy_index_partition(Context ctx, IndexPartition handle) {
  if (!enabled) {
    lrt->destroy_index_partition(ctx, handle);
    return;
  }

  if (skip_api_call()) return;

  resilient_tag_t ip_tag = ipartition_tags.at(handle);
  IndexPartitionState &ip_state = state.ipartition_state.at(ip_tag);
  ip_state.destroyed = true;
  lrt->destroy_index_partition(ctx, handle);
}

IndexSpace Runtime::restore_index_space(Context ctx) {
  IndexSpaceSerializer ris = state.ispaces.at(index_space_tag);
  IndexSpace is = ris.inflate(this, ctx);
  assert(ispaces.size() == index_space_tag);
  ispaces.push_back(is);
  index_space_tag++;
  return is;
}

IndexSpace Runtime::create_index_space(Context ctx, const Domain &bounds) {
  if (!enabled) {
    return lrt->create_index_space(ctx, bounds);
  }

  if (replay && index_space_tag < max_index_space_tag) {
    return restore_index_space(ctx);
  }

  IndexSpace is = lrt->create_index_space(ctx, bounds);
  ispaces.push_back(is);
  index_space_tag++;
  return is;
}

IndexSpace Runtime::create_index_space_union(Context ctx, IndexPartition parent,
                                             const DomainPoint &color,
                                             const std::vector<IndexSpace> &handles) {
  if (!enabled) {
    return lrt->create_index_space_union(ctx, parent, color, handles);
  }

  if (replay && index_space_tag < max_index_space_tag) {
    index_space_tag++;
    return lrt->get_index_subspace(ctx, parent, color);
  }

  IndexSpace is = lrt->create_index_space_union(ctx, parent, color, handles);
  ispaces.push_back(is);
  index_space_tag++;
  return is;
}

IndexSpace Runtime::create_index_space_union(Context ctx, IndexPartition parent,
                                             const DomainPoint &color,
                                             IndexPartition handle) {
  if (!enabled) {
    return lrt->create_index_space_union(ctx, parent, color, handle);
  }

  if (replay && index_space_tag < max_index_space_tag) {
    index_space_tag++;
    return lrt->get_index_subspace(ctx, parent, color);
  }

  IndexSpace is = lrt->create_index_space_union(ctx, parent, color, handle);
  ispaces.push_back(is);
  index_space_tag++;
  return is;
}

IndexSpace Runtime::create_index_space_difference(
    Context ctx, IndexPartition parent, const DomainPoint &color, IndexSpace initial,
    const std::vector<IndexSpace> &handles) {
  if (!enabled) {
    return lrt->create_index_space_difference(ctx, parent, color, initial, handles);
  }

  if (replay && index_space_tag < max_index_space_tag) {
    index_space_tag++;
    return lrt->get_index_subspace(ctx, parent, color);
  }

  IndexSpace is =
      lrt->create_index_space_difference(ctx, parent, color, initial, handles);
  ispaces.push_back(is);
  index_space_tag++;
  return is;
}

IndexPartition Runtime::restore_index_partition(Context ctx, IndexSpace index_space,
                                                IndexSpace color_space) {
  if (state.ipartition_state.at(partition_tag).destroyed) {
    IndexPartition ip = IndexPartition::NO_PART;
    ipartitions.push_back(ip);
    partition_tag++;
    return IndexPartition::NO_PART;
  }

  IndexPartitionSerializer rip = state.ipartitions.at(partition_tag);
  IndexPartition ip = rip.inflate(this, ctx, index_space, color_space);
  ipartitions.push_back(ip);
  ipartition_tags[ip] = partition_tag;
  partition_tag++;
  return ip;
}

IndexPartition Runtime::create_equal_partition(Context ctx, IndexSpace parent,
                                               IndexSpace color_space) {
  if (!enabled) {
    return lrt->create_equal_partition(ctx, parent, color_space);
  }

  if (replay && partition_tag < max_partition_tag) {
    return restore_index_partition(ctx, parent, color_space);
  }

  IndexPartition ip = lrt->create_equal_partition(ctx, parent, color_space);
  ipartitions.push_back(ip);
  ipartition_tags[ip] = partition_tag;
  state.ipartition_state.emplace_back();
  partition_tag++;
  return ip;
}

IndexPartition Runtime::create_pending_partition(Context ctx, IndexSpace parent,
                                                 IndexSpace color_space) {
  if (!enabled) {
    return lrt->create_pending_partition(ctx, parent, color_space);
  }

  if (replay && partition_tag < max_partition_tag) {
    return restore_index_partition(ctx, parent, color_space);
  }

  IndexPartition ip = lrt->create_pending_partition(ctx, parent, color_space);
  ipartitions.push_back(ip);
  ipartition_tags[ip] = partition_tag;
  state.ipartition_state.emplace_back();
  partition_tag++;
  return ip;
}

IndexPartition Runtime::create_partition_by_field(Context ctx, LogicalRegion handle,
                                                  LogicalRegion parent, FieldID fid,
                                                  IndexSpace color_space) {
  if (!enabled) {
    return lrt->create_partition_by_field(ctx, handle, parent, fid, color_space);
  }

  if (replay && partition_tag < max_partition_tag) {
    return restore_index_partition(ctx, handle.get_index_space(), color_space);
  }

  IndexPartition ip =
      lrt->create_partition_by_field(ctx, handle, parent, fid, color_space);
  ipartitions.push_back(ip);
  ipartition_tags[ip] = partition_tag;
  state.ipartition_state.emplace_back();
  partition_tag++;
  return ip;
}

IndexPartition Runtime::create_partition_by_image(Context ctx, IndexSpace handle,
                                                  LogicalPartition projection,
                                                  LogicalRegion parent, FieldID fid,
                                                  IndexSpace color_space) {
  if (!enabled) {
    return lrt->create_partition_by_image(ctx, handle, projection, parent, fid,
                                          color_space);
  }

  if (replay && partition_tag < max_partition_tag) {
    return restore_index_partition(ctx, handle, color_space);
  }

  IndexPartition ip =
      lrt->create_partition_by_image(ctx, handle, projection, parent, fid, color_space);
  ipartitions.push_back(ip);
  ipartition_tags[ip] = partition_tag;
  state.ipartition_state.emplace_back();
  partition_tag++;
  return ip;
}

IndexPartition Runtime::create_partition_by_preimage(Context ctx,
                                                     IndexPartition projection,
                                                     LogicalRegion handle,
                                                     LogicalRegion parent, FieldID fid,
                                                     IndexSpace color_space) {
  if (!enabled) {
    return lrt->create_partition_by_preimage(ctx, projection, handle, parent, fid,
                                             color_space);
  }

  if (replay && partition_tag < max_partition_tag) {
    return restore_index_partition(ctx, handle.get_index_space(), color_space);
  }

  IndexPartition ip = lrt->create_partition_by_preimage(ctx, projection, handle, parent,
                                                        fid, color_space);
  ipartitions.push_back(ip);
  ipartition_tags[ip] = partition_tag;
  state.ipartition_state.emplace_back();
  partition_tag++;
  return ip;
}

IndexPartition Runtime::create_partition_by_difference(Context ctx, IndexSpace parent,
                                                       IndexPartition handle1,
                                                       IndexPartition handle2,
                                                       IndexSpace color_space) {
  if (!enabled) {
    return lrt->create_partition_by_difference(ctx, parent, handle1, handle2,
                                               color_space);
  }

  if (replay && partition_tag < max_partition_tag) {
    return restore_index_partition(ctx, parent, color_space);
  }

  IndexPartition ip =
      lrt->create_partition_by_difference(ctx, parent, handle1, handle2, color_space);
  ipartitions.push_back(ip);
  ipartition_tags[ip] = partition_tag;
  state.ipartition_state.emplace_back();
  partition_tag++;
  return ip;
}

Color Runtime::create_cross_product_partitions(
    Context ctx, IndexPartition handle1, IndexPartition handle2,
    std::map<IndexSpace, IndexPartition> &handles) {
  return lrt->create_cross_product_partitions(ctx, handle1, handle2, handles);
}

LogicalPartition Runtime::get_logical_partition(Context ctx, LogicalRegion parent,
                                                IndexPartition handle) {
  return lrt->get_logical_partition(ctx, parent, handle);
}

LogicalPartition Runtime::get_logical_partition(LogicalRegion parent,
                                                IndexPartition handle) {
  return lrt->get_logical_partition(parent, handle);
}

LogicalPartition Runtime::get_logical_partition_by_tree(IndexPartition handle,
                                                        FieldSpace fspace,
                                                        RegionTreeID tid) {
  return lrt->get_logical_partition_by_tree(handle, fspace, tid);
}

LogicalRegion Runtime::get_logical_subregion_by_color(Context ctx,
                                                      LogicalPartition parent, Color c) {
  return lrt->get_logical_subregion_by_color(ctx, parent, c);
}

LogicalRegion Runtime::get_logical_subregion_by_color(Context ctx,
                                                      LogicalPartition parent,
                                                      DomainPoint c) {
  return lrt->get_logical_subregion_by_color(ctx, parent, c);
}

LogicalRegion Runtime::get_logical_subregion_by_color(LogicalPartition parent,
                                                      const DomainPoint &c) {
  return lrt->get_logical_subregion_by_color(parent, c);
}

Legion::Mapping::MapperRuntime *Runtime::get_mapper_runtime(void) {
  return lrt->get_mapper_runtime();
}

void Runtime::replace_default_mapper(Legion::Mapping::Mapper *mapper, Processor proc) {
  lrt->replace_default_mapper(mapper, proc);
}

ptr_t Runtime::safe_cast(Context ctx, ptr_t pointer, LogicalRegion region) {
  return lrt->safe_cast(ctx, pointer, region);
}

DomainPoint Runtime::safe_cast(Context ctx, DomainPoint point, LogicalRegion region) {
  return lrt->safe_cast(ctx, point, region);
}

void Runtime::fill_field(Context ctx, LogicalRegion handle, LogicalRegion parent,
                         FieldID fid, const void *value, size_t value_size,
                         Predicate pred) {
  if (skip_api_call()) return;
  lrt->fill_field(ctx, handle, parent, fid, value, value_size, pred);
}

Future Runtime::select_tunable_value(Context ctx, const TunableLauncher &launcher) {
  if (!enabled) {
    return lrt->select_tunable_value(ctx, launcher);
  }

  if (replay && future_tag < max_future_tag) {
    return futures.at(future_tag++);
  }

  Future rf = lrt->select_tunable_value(ctx, launcher);
  futures.push_back(rf);
  future_tag++;
  return rf;
}

void Runtime::fill_fields(Context ctx, const FillLauncher &launcher) {
  if (!enabled) {
    lrt->fill_fields(ctx, launcher);
    return;
  }

  if (skip_api_call()) return;
  lrt->fill_fields(ctx, launcher);
}

void Runtime::get_field_space_fields(FieldSpace handle, std::set<FieldID> &fields) {
  lrt->get_field_space_fields(handle, fields);
}

size_t Runtime::get_field_size(FieldSpace handle, FieldID fid) {
  return lrt->get_field_size(handle, fid);
}

Processor Runtime::get_executing_processor(Context ctx) {
  return lrt->get_executing_processor(ctx);
}

void Runtime::print_once(Context ctx, FILE *f, const char *message) {
  lrt->print_once(ctx, f, message);
}

static void generate_disk_file(const char *file_name) {
  std::ofstream file(file_name, std::ios::binary);
  if (!file) {
    log_resilience.error() << "unable to open file '" << file_name
                           << "': " << strerror(errno);
    abort();
  }
}

static void covering_set_partition(
    Legion::Runtime *lrt, LogicalPartition partition, unsigned depth,
    const std::map<LogicalRegion, std::set<LogicalPartition>> &region_tree,
    CoveringSet &result);

static bool covering_set_region(
    Legion::Runtime *lrt, LogicalRegion region, unsigned depth,
    const std::map<LogicalRegion, std::set<LogicalPartition>> &region_tree,
    CoveringSet &result) {
  auto partitions = region_tree.find(region);
  if (partitions != region_tree.end()) {
    // If this region is partitioned, find the deepest covering set.
    CoveringSet best;
    for (auto partition : partitions->second) {
      CoveringSet attempt;
      covering_set_partition(lrt, partition, depth + 1, region_tree, attempt);
      if (attempt.depth > best.depth) {
        best = std::move(attempt);
      }
    }
    result.partitions.insert(best.partitions.begin(), best.partitions.end());
    result.regions.insert(best.regions.begin(), best.regions.end());
    result.depth = std::max(result.depth, best.depth);
    return true;
  } else {
    return false;
  }
}

static void covering_set_partition(
    Legion::Runtime *lrt, LogicalPartition partition, unsigned depth,
    const std::map<LogicalRegion, std::set<LogicalPartition>> &region_tree,
    CoveringSet &result) {
  // For each region, find the best way to cover it.
  IndexPartition ip = partition.get_index_partition();
  Domain domain = lrt->get_index_partition_color_space(ip);
  bool recurses = false;
  std::vector<LogicalRegion> uncovered;
  for (Domain::DomainPointIterator i(domain); i; ++i) {
    LogicalRegion subregion = lrt->get_logical_subregion_by_color(partition, *i);
    bool recurse = covering_set_region(lrt, subregion, depth + 1, region_tree, result);
    recurses = recurses || recurse;
    if (!recurse) {
      uncovered.push_back(subregion);
    }
  }
  if (!recurses) {
    // If nothing recursed, use this partition.
    result.partitions.insert(partition);
    result.depth = std::max(result.depth, depth);
  } else {
    // Otherwise make sure we don't lose any uncovered regions.
    result.regions.insert(uncovered.begin(), uncovered.end());
  }
}

void Runtime::compute_covering_set(LogicalRegion r, CoveringSet &covering_set) {
  auto region_tag = region_tags.at(r);
  auto &lr_state = region_tree_state.at(region_tag);

  // Hack: reverse-engineer the region tree since the runtime provides no downward queries
  // on regions.
  std::map<LogicalRegion, std::set<LogicalPartition>> region_tree;
  for (auto &recent : lr_state.recent_partitions) {
    LogicalRegion region = recent.first;
    LogicalPartition partition = recent.second;

    region_tree[region].insert(partition);
    while (lrt->has_parent_logical_partition(region)) {
      partition = lrt->get_parent_logical_partition(region);
      region = lrt->get_parent_logical_region(partition);
      region_tree[region].insert(partition);
    }
  }

  if (!covering_set_region(lrt, r, 0, region_tree, covering_set)) {
    // If nothing else works, just choose the region itself.
    covering_set.regions.insert(r);
  }

  log_resilience.info() << "Computed covering set:";
  for (auto &region : covering_set.regions) {
    log_resilience.info() << "  Region: " << region;
  }
  for (auto &partition : covering_set.partitions) {
    log_resilience.info() << "  Partition: " << partition;
  }
}

void Runtime::restore_region_content(Context ctx, LogicalRegion lr) {
  resilient_tag_t tag = region_tags.at(lr);

  log_resilience.info() << "Reconstructing logical region from checkpoint, tag " << tag;
  LogicalRegion cpy =
      lrt->create_logical_region(ctx, lr.get_index_space(), lr.get_field_space());

  std::vector<FieldID> fids;
  lrt->get_field_space_fields(lr.get_field_space(), fids);
  AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, cpy, false, false);

  char file_name[4096];
  snprintf(file_name, sizeof(file_name), "checkpoint.%ld.lr.%lu.dat", checkpoint_tag,
           tag);
  log_resilience.info() << "Reading from file " << file_name;
  al.attach_file(file_name, fids, LEGION_FILE_READ_ONLY);

  PhysicalRegion pr = lrt->attach_external_resource(ctx, al);

  CopyLauncher cl;
  cl.add_copy_requirements(RegionRequirement(cpy, READ_ONLY, EXCLUSIVE, cpy),
                           RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));

  for (auto &fid : fids) {
    cl.add_src_field(0, fid);
    cl.add_dst_field(0, fid);
  }

  // FIXME: Convert to index launch
  lrt->issue_copy_operation(ctx, cl);
  lrt->detach_external_resource(ctx, pr);
  lrt->destroy_logical_region(ctx, cpy);
}

void Runtime::save_region_content(Context ctx, LogicalRegion lr) {
  CoveringSet covering_set;
  compute_covering_set(lr, covering_set);

  resilient_tag_t tag = region_tags.at(lr);
  char file_name[4096];
  snprintf(file_name, sizeof(file_name), "checkpoint.%ld.lr.%lu.dat", checkpoint_tag,
           tag);

  log_resilience.info() << "save_logical_region: lr " << lr << " file_name " << file_name;
  generate_disk_file(file_name);

  LogicalRegion cpy =
      lrt->create_logical_region(ctx, lr.get_index_space(), lr.get_field_space());

  std::vector<FieldID> fids;
  lrt->get_field_space_fields(lr.get_field_space(), fids);

  AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, cpy, false, false);
  al.attach_file(file_name, fids, LEGION_FILE_READ_WRITE);

  PhysicalRegion pr = lrt->attach_external_resource(ctx, al);

  CopyLauncher cl;
  cl.add_copy_requirements(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr),
                           RegionRequirement(cpy, WRITE_DISCARD, EXCLUSIVE, cpy));

  for (auto &fid : fids) {
    cl.add_src_field(0, fid);
    cl.add_dst_field(0, fid);
  }

  // FIXME (Elliott): Index launch this?
  lrt->issue_copy_operation(ctx, cl);
  lrt->detach_external_resource(ctx, pr);
  lrt->destroy_logical_region(ctx, cpy);
}

void Runtime::checkpoint(Context ctx) {
  if (!enabled) {
    log_resilience.error()
        << "Must enable checkpointing with runtime->enable_checkpointing()";
    abort();
  }

  if (replay && checkpoint_tag == max_checkpoint_tag - 1) {
    // This is the checkpoint we originally saved. Restore all region data at this point.
    log_resilience.info() << "In checkpoint: restoring regions from tag "
                          << checkpoint_tag;

    for (resilient_tag_t i = 0; i < regions.size(); ++i) {
      auto &lr = regions.at(i);
      auto &lr_state = state.region_state.at(i);
      if (!lr_state.destroyed) {
        restore_region_content(ctx, lr);
      }
    }
  }

  if (replay && checkpoint_tag < max_checkpoint_tag) {
    log_resilience.info() << "In checkpoint: skipping tag " << checkpoint_tag << " max "
                          << max_checkpoint_tag;
    checkpoint_tag++;
    return;
  }

  log_resilience.info() << "In checkpoint: tag " << checkpoint_tag;
  log_resilience.info() << "Number of logical regions " << regions.size();

  for (size_t i = 0; i < regions.size(); ++i) {
    auto &lr = regions.at(i);
    auto &lr_state = state.region_state.at(i);

    if (lr_state.destroyed) {
      continue;
    }
    save_region_content(ctx, lr);
  }

  log_resilience.info() << "Saved all logical regions!";

  // Synchornize checkpoint state
  // Note: this is incremental!
  state.max_api_tag = api_tag;
  state.max_future_tag = future_tag;
  state.max_future_map_tag = future_map_tag;
  state.max_index_space_tag = index_space_tag;
  state.max_region_tag = region_tag;
  state.max_partition_tag = partition_tag;
  state.max_checkpoint_tag = checkpoint_tag + 1;

  for (resilient_tag_t i = state.futures.size(); i < futures.size(); ++i) {
    auto &ft = futures.at(i);
    state.futures.emplace_back(ft);
  }

  for (resilient_tag_t i = state.future_maps.size(); i < future_maps.size(); ++i) {
    auto &ft = future_maps.at(i);
    state.future_maps.emplace_back(ft);
  }

  for (resilient_tag_t i = state.ispaces.size(); i < ispaces.size(); ++i) {
    auto &is = ispaces.at(i);
    state.ispaces.emplace_back(lrt->get_index_space_domain(ctx, is));
  }

  // Partition table does not include deleted entries, but do the best we can to avoid
  // useless work.
  resilient_tag_t ip_start = 0;
  {
    auto ip_rbegin = state.ipartitions.rbegin();
    auto ip_rend = state.ipartitions.rend();
    if (ip_rbegin != ip_rend) {
      ip_start = ip_rbegin->first;
    }
  }
  for (resilient_tag_t i = ip_start; i < ipartitions.size(); ++i) {
    auto &ip_state = state.ipartition_state.at(i);
    if (ip_state.destroyed) continue;

    auto &ip = ipartitions.at(i);
    Domain color_space = lrt->get_index_partition_color_space(ip);
    state.ipartitions[i] = IndexPartitionSerializer(this, ip, color_space);
  }
  for (auto it = state.ipartitions.begin(); it != state.ipartitions.end();) {
    auto &ip_state = state.ipartition_state.at(it->first);
    if (ip_state.destroyed) {
      state.ipartitions.erase(it++);
    } else {
      ++it;
    }
  }

  // Sanity checks
  assert(state.max_future_tag == state.futures.size());
  assert(state.max_future_map_tag == state.future_maps.size());
  assert(state.max_region_tag == state.region_state.size());
  assert(state.max_index_space_tag == state.ispaces.size());
  assert(state.max_partition_tag == state.ipartition_state.size());
  assert(state.max_checkpoint_tag == checkpoint_tag + 1);

  std::stringstream serialized;
  {
#ifdef DEBUG_LEGION
    cereal::XMLOutputArchive oarchive(serialized);
#else
    cereal::BinaryOutputArchive oarchive(serialized);
#endif
    oarchive(state);
  }
  std::string serialized_data = serialized.str();
  Future checkpoint_tag_f =
      Legion::Future::from_value<resilient_tag_t>(lrt, checkpoint_tag);
  Future serialized_data_f = Legion::Future::from_untyped_pointer(
      lrt, serialized_data.data(), serialized_data.size());

  {
    TaskLauncher launcher(write_checkpoint_task_id, TaskArgument());
    launcher.add_future(checkpoint_tag_f);
    launcher.add_future(serialized_data_f);
    lrt->execute_task(ctx, launcher);
  }

  checkpoint_tag++;
}

void Runtime::enable_checkpointing(Context ctx) {
  bool first_time = !enabled;
  enabled = true;
  if (!first_time) return;

  // These values get parsed in Runtime::start
  replay = initial_replay;
  resilient_tag_t load_checkpoint_tag = initial_checkpoint_tag;

  log_resilience.info() << "In enable_checkpointing: replay " << replay
                        << " load_checkpoint_tag " << load_checkpoint_tag;

  if (replay) {
    char file_name[4096];
    snprintf(file_name, sizeof(file_name), "checkpoint.%ld.dat", load_checkpoint_tag);
    {
      std::ifstream file(file_name, std::ios::binary);
      // This is a hack, but apparently C++ iostream exception messages are useless, so
      // this is what we've got. See: https://codereview.stackexchange.com/a/58130
      if (!file) {
        log_resilience.error() << "unable to open file '" << file_name
                               << "': " << strerror(errno);
        abort();
      }
#ifdef DEBUG_LEGION
      cereal::XMLInputArchive iarchive(file);
#else
      cereal::BinaryInputArchive iarchive(file);
#endif
      iarchive(state);
      file.close();
      if (!file) {
        log_resilience.error() << "error in closing file '" << file_name
                               << "': " << strerror(errno);
        abort();
      }
    }

    log_resilience.info() << "After loading checkpoint, max: api " << state.max_api_tag
                          << " future " << state.max_future_tag << " future_map "
                          << state.max_future_map_tag << " index_space "
                          << state.max_index_space_tag << " region_tag "
                          << state.max_region_tag << " partition "
                          << state.max_partition_tag << " checkpoint "
                          << state.max_checkpoint_tag;

    // Sanity checks
    assert(state.max_future_tag == state.futures.size());
    assert(state.max_future_map_tag == state.future_maps.size());
    assert(state.max_region_tag == state.region_state.size());
    assert(state.max_index_space_tag == state.ispaces.size());
    assert(state.max_partition_tag == state.ipartition_state.size());
    assert(state.max_checkpoint_tag == load_checkpoint_tag + 1);

    // Restore state
    for (auto &ft : state.futures) futures.emplace_back(ft);
    for (auto &fm : state.future_maps) {
      FutureMap fm_ = fm.inflate(this, ctx);
      future_maps.push_back(fm_);
    }

    max_api_tag = state.max_api_tag;
    max_future_tag = state.max_future_tag;
    max_future_map_tag = state.max_future_map_tag;
    max_index_space_tag = state.max_index_space_tag;
    max_region_tag = state.max_region_tag;
    max_partition_tag = state.max_partition_tag;
    max_checkpoint_tag = state.max_checkpoint_tag;
  }
}
