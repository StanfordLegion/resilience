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

Logger log_resilience("resilience");

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

  if (replay && api_tag < max_api_tag) {
    api_tag++;
    return;
  }
  api_tag++;
  lrt->attach_name(handle, name, is_mutable);
}

void Runtime::attach_name(LogicalPartition handle, const char *name, bool is_mutable) {
  lrt->attach_name(handle, name, is_mutable);
}

void Runtime::issue_execution_fence(Context ctx, const char *provenance) {
  lrt->issue_execution_fence(ctx, provenance);
}

void Runtime::issue_copy_operation(Context ctx, const CopyLauncher &launcher) {
  if (replay && api_tag < max_api_tag) {
    api_tag++;
    return;
  }
  api_tag++;
  lrt->issue_copy_operation(ctx, launcher);
}

void Runtime::issue_copy_operation(Context ctx, const IndexCopyLauncher &launcher) {
  if (replay && api_tag < max_api_tag) {
    api_tag++;
    return;
  }
  api_tag++;
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

int Runtime::start(int argc, char **argv, bool background, bool supply_default_mapper) {
  return Legion::Runtime::start(argc, argv, background, supply_default_mapper);
}

void FutureMap::wait_all_results(Runtime *runtime) {
  /* What if this FutureMap occured after the checkpoint?! */
  if (runtime->enabled && runtime->replay) return;
  fm.wait_all_results();
}

void Runtime::track_region_state(const RegionRequirement &rr) {
  auto region_tag = region_tags.at(rr.parent);
  auto &state = region_state[region_tag];

  // If this access is on a disjoint and complete partition, track it; it's probably a
  // good partition to save.
  if (rr.handle_type == LEGION_PARTITION_PROJECTION &&
      !(rr.privilege == LEGION_NO_ACCESS || rr.privilege == LEGION_REDUCE)) {
    LogicalPartition lp = rr.partition;
    IndexPartition ip = lp.get_index_partition();
    if (lrt->is_index_partition_disjoint(ip) && lrt->is_index_partition_complete(ip)) {
      LogicalRegion parent = lrt->get_parent_logical_region(lp);
      state.recent_partitions[parent] = lp.get_index_partition();
    }
  }
}

FutureMap Runtime::execute_index_space(Context ctx, const IndexTaskLauncher &launcher) {
  if (!enabled) {
    return lrt->execute_index_space(ctx, launcher);
  }

  if (replay && future_map_tag < max_future_map_tag) {
    log_resilience.info() << "execute_index_space: no-op for replay, tag "
                          << future_map_tag;
    return future_maps[future_map_tag++];
  }

  for (auto &rr : launcher.region_requirements) {
    track_region_state(rr);
  }

  Legion::FutureMap fm = lrt->execute_index_space(ctx, launcher);

  FutureMap rfm;
  if (launcher.launch_domain == Domain::NO_DOMAIN)
    rfm = FutureMap(fm, lrt->get_index_space_domain(launcher.launch_space));
  else
    rfm = FutureMap(fm, launcher.launch_domain);

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
    return futures[future_tag++];
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
    return futures[future_tag++];
  }
  log_resilience.info() << "execute_task: launching task_id " << launcher.task_id;

  for (auto &rr : launcher.region_requirements) {
    track_region_state(rr);
  }

  Future ft = lrt->execute_task(ctx, launcher);
  future_tag++;
  futures.push_back(ft);
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
    /* Unlike an arbitrary task, get_current_time and friends are guaranteed to
     * return a non-void Future.
     */
    assert(!futures[future_tag].empty);
    return futures[future_tag++];
  }
  Future ft = lrt->get_current_time(ctx, precondition.lft);
  future_tag++;
  futures.push_back(ft);
  return ft;
}

Future Runtime::get_current_time_in_microseconds(Context ctx, Future precondition) {
  if (!enabled) {
    return lrt->get_current_time_in_microseconds(ctx, precondition.lft);
  }

  if (replay && future_tag < max_future_tag) {
    assert(!futures[future_tag].empty);
    return futures[future_tag++];
  }
  Future ft = lrt->get_current_time_in_microseconds(ctx, precondition.lft);
  future_tag++;
  futures.push_back(ft);
  return ft;
}

Predicate Runtime::create_predicate(Context ctx, const Future &f) {
  if (!enabled) {
    return lrt->create_predicate(ctx, f.lft);
  }

  // Assuming no predicate crosses the checkpoint boundary
  if (replay && api_tag < max_api_tag) {
    api_tag++;
    return Predicate(false);
  }
  api_tag++;
  return lrt->create_predicate(ctx, f.lft);
}

Predicate Runtime::create_predicate(Context ctx, const PredicateLauncher &launcher) {
  if (!enabled) {
    return lrt->create_predicate(ctx, launcher);
  }

  if (replay && api_tag < max_api_tag) {
    api_tag++;
    return Predicate(false);
  }
  api_tag++;
  return lrt->create_predicate(ctx, launcher);
}

Predicate Runtime::predicate_not(Context ctx, const Predicate &p) {
  if (!enabled) {
    return lrt->predicate_not(ctx, p);
  }

  if (replay && api_tag < max_api_tag) {
    api_tag++;
    return Predicate(false);
  }
  api_tag++;
  return lrt->predicate_not(ctx, p);
}

Future Runtime::get_predicate_future(Context ctx, const Predicate &p) {
  if (!enabled) {
    return lrt->get_predicate_future(ctx, p);
  }

  if (replay && future_tag < max_future_tag) {
    return futures[future_tag++];
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
  if (!enabled) {
    return lrt->create_logical_region(ctx, index, fields, task_local, provenance);
  }

  // Region restored in replay:
  LogicalRegion lr;
  if (replay && region_tag < max_region_tag) {
    if (region_state.at(region_tag).destroyed) {
      // If the region is already destroyed, still create it. (We still go through the
      // full object lifecycle.) But don't bother populating it; we don't need the
      // contents.
      lr = lrt->create_logical_region(ctx, index, fields);
    } else {
      // Create the region. We use a second (identical) copy for use with attach. We MUST
      // copy because detaching invalidates that data.

      log_resilience.info() << "Reconstructing logical region from checkpoint";
      lr = lrt->create_logical_region(ctx, index, fields);
      LogicalRegion cpy = lrt->create_logical_region(ctx, index, fields);

      std::vector<FieldID> fids;
      lrt->get_field_space_fields(fields, fids);
      AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, cpy);

      char file_name[4096];
      snprintf(file_name, sizeof(file_name), "checkpoint.%ld.lr.%ld.dat", checkpoint_tag,
               region_tag);
      al.attach_file(file_name, fids, LEGION_FILE_READ_ONLY);

      PhysicalRegion pr = lrt->attach_external_resource(ctx, al);

      CopyLauncher cl;
      cl.add_copy_requirements(RegionRequirement(cpy, READ_ONLY, EXCLUSIVE, cpy),
                               RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));

      for (auto &fid : fids) {
        cl.add_src_field(0, fid);
        cl.add_dst_field(0, fid);
      }

      // FIXME: Convert to index launch
      lrt->issue_copy_operation(ctx, cl);
      lrt->detach_external_resource(ctx, pr);
    }
  } else {
    // New region (or not replay):
    lr = lrt->create_logical_region(ctx, index, fields);
    region_state.push_back(LogicalRegionState());

    // We initialize the data here to ensure we will never hit uninitialized data later.
    initialize_region(ctx, lr);
  }

  assert(regions.size() == region_tag);
  regions.push_back(lr);
  assert(regions.size() <= region_state.size());
  region_tags[lr] = region_tag;
  region_tag++;
  return lr;
}

/* Inline mappings need to be disallowed */
PhysicalRegion Runtime::map_region(Context ctx, const InlineLauncher &launcher) {
  return lrt->map_region(ctx, launcher);
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

  for (size_t i = 0; i < regions.size(); ++i) {
    auto &lr = regions[i];
    auto &state = region_state[i];

    if (lr == handle) {
      state.destroyed = true;
      break;
    }
  }
  lrt->destroy_logical_region(ctx, handle);
}

void Runtime::destroy_index_partition(Context ctx, IndexPartition handle) {
  if (!enabled) {
    lrt->destroy_index_partition(ctx, handle);
    return;
  }

  if (replay && api_tag < max_api_tag) {
    api_tag++;
    return;
  }

  for (auto &rip : partitions) {
    if (rip.ip == handle) {
      // Should not delete an already deleted partition
      assert(rip.is_valid);
      rip.is_valid = false;
      break;
    }
  }
  api_tag++;
  lrt->destroy_index_partition(ctx, handle);
}

IndexSpace Runtime::restore_index_space(Context ctx) {
  ResilientIndexSpace ris = index_spaces[index_space_tag++];
  std::vector<Domain> rects;

  for (auto &rect : ris.domain.rects) {
    rects.push_back(Domain(rect));
  }

  IndexSpace is = lrt->create_index_space(ctx, rects);
  return is;
}

IndexSpace Runtime::create_index_space(Context ctx, const Domain &bounds) {
  if (!enabled) {
    return lrt->create_index_space(ctx, bounds);
  }

  if (replay && index_space_tag < max_index_space_tag) restore_index_space(ctx);

  IndexSpace is = lrt->create_index_space(ctx, bounds);
  ResilientIndexSpace ris(lrt->get_index_space_domain(ctx, is));
  index_spaces.push_back(ris);
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
  ResilientIndexSpace ris(lrt->get_index_space_domain(ctx, is));
  index_spaces.push_back(ris);
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
  ResilientIndexSpace ris(lrt->get_index_space_domain(ctx, is));
  index_spaces.push_back(ris);
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
  ResilientIndexSpace ris(lrt->get_index_space_domain(ctx, is));
  index_spaces.push_back(ris);
  index_space_tag++;
  return is;
}

void ResilientIndexPartition::save(Context ctx, Legion::Runtime *lrt, DomainPoint dp) {
  IndexSpace sub_is = lrt->get_index_subspace(ctx, ip, dp);
  if (sub_is == IndexSpace::NO_SPACE) return;
  ResilientIndexSpace sub_ris(lrt->get_index_space_domain(ctx, sub_is));
  map[dp] = sub_ris;
}

void ResilientIndexPartition::setup_for_checkpoint(Context ctx, Legion::Runtime *lrt) {
  if (!is_valid) return;

  Domain color_domain = lrt->get_index_partition_color_space(ctx, ip);

  color_space = color_domain; /* Implicit conversion */

  for (Domain::DomainPointIterator i(color_domain); i; ++i) {
    save(ctx, lrt, *i);
  }
}

IndexPartition Runtime::restore_index_partition(Context ctx, IndexSpace index_space,
                                                IndexSpace color_space) {
  ResilientIndexPartition rip = partitions[partition_tag++];
  MultiDomainPointColoring *mdpc = new MultiDomainPointColoring();

  /* For rect in color space
   *   For point in rect
   *     Get the index space under this point
   *     For rect in index space
   *       Insert into mdpc at point
   */
  for (auto &rect : rip.color_space.domain.rects) {
    for (Domain::DomainPointIterator i((Domain)rect); i; ++i) {
      ResilientIndexSpace ris = rip.map[*i];
      for (auto &rect_ris : ris.domain.rects) (*mdpc)[*i].insert(rect_ris);
    }
  }

  /* Assuming the domain cannot change */
  Domain color_domain = lrt->get_index_space_domain(ctx, color_space);
  IndexPartition ip = lrt->create_index_partition(ctx, index_space, color_domain, *mdpc);
  return ip;
}

IndexPartition Runtime::create_equal_partition(Context ctx, IndexSpace parent,
                                               IndexSpace color_space) {
  if (!enabled) {
    return lrt->create_equal_partition(ctx, parent, color_space);
  }

  if (replay && !partitions[partition_tag].is_valid) {
    partition_tag++;
    return IndexPartition::NO_PART;
  }

  if (replay && partition_tag < max_partition_tag)
    return restore_index_partition(ctx, parent, color_space);

  ResilientIndexPartition rip = lrt->create_equal_partition(ctx, parent, color_space);
  partitions.push_back(rip);
  partition_tag++;
  return rip.ip;
}

IndexPartition Runtime::create_pending_partition(Context ctx, IndexSpace parent,
                                                 IndexSpace color_space) {
  if (!enabled) {
    return lrt->create_pending_partition(ctx, parent, color_space);
  }

  if (replay && !partitions[partition_tag].is_valid) {
    partition_tag++;
    return IndexPartition::NO_PART;
  }

  if (replay && partition_tag < max_partition_tag)
    return restore_index_partition(ctx, parent, color_space);

  ResilientIndexPartition rip = lrt->create_pending_partition(ctx, parent, color_space);
  partitions.push_back(rip);
  partition_tag++;
  return rip.ip;
}

IndexPartition Runtime::create_partition_by_field(Context ctx, LogicalRegion handle,
                                                  LogicalRegion parent, FieldID fid,
                                                  IndexSpace color_space) {
  if (!enabled) {
    return lrt->create_partition_by_field(ctx, handle, parent, fid, color_space);
  }

  if (replay && !partitions[partition_tag].is_valid) {
    partition_tag++;
    return IndexPartition::NO_PART;
  }

  if (replay && partition_tag < max_partition_tag)
    return restore_index_partition(ctx, handle.get_index_space(), color_space);

  ResilientIndexPartition rip =
      lrt->create_partition_by_field(ctx, handle, parent, fid, color_space);
  partitions.push_back(rip);
  partition_tag++;
  return rip.ip;
}

IndexPartition Runtime::create_partition_by_image(Context ctx, IndexSpace handle,
                                                  LogicalPartition projection,
                                                  LogicalRegion parent, FieldID fid,
                                                  IndexSpace color_space) {
  if (!enabled) {
    return lrt->create_partition_by_image(ctx, handle, projection, parent, fid,
                                          color_space);
  }

  if (replay && !partitions[partition_tag].is_valid) {
    partition_tag++;
    return IndexPartition::NO_PART;
  }

  if (replay && partition_tag < max_partition_tag)
    return restore_index_partition(ctx, handle, color_space);

  ResilientIndexPartition rip =
      lrt->create_partition_by_image(ctx, handle, projection, parent, fid, color_space);
  partitions.push_back(rip);
  partition_tag++;
  return rip.ip;
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

  if (replay && !partitions[partition_tag].is_valid) {
    partition_tag++;
    return IndexPartition::NO_PART;
  }

  if (replay && partition_tag < max_partition_tag)
    return restore_index_partition(ctx, handle.get_index_space(), color_space);

  ResilientIndexPartition rip = lrt->create_partition_by_preimage(
      ctx, projection, handle, parent, fid, color_space);
  partitions.push_back(rip);
  partition_tag++;
  return rip.ip;
}

IndexPartition Runtime::create_partition_by_difference(Context ctx, IndexSpace parent,
                                                       IndexPartition handle1,
                                                       IndexPartition handle2,
                                                       IndexSpace color_space) {
  if (!enabled) {
    return lrt->create_partition_by_difference(ctx, parent, handle1, handle2,
                                               color_space);
  }

  if (replay && !partitions[partition_tag].is_valid) {
    partition_tag++;
    return IndexPartition::NO_PART;
  }

  if (replay && partition_tag < max_partition_tag)
    return restore_index_partition(ctx, parent, color_space);

  ResilientIndexPartition rip =
      lrt->create_partition_by_difference(ctx, parent, handle1, handle2, color_space);
  partitions.push_back(rip);
  partition_tag++;
  return rip.ip;
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

bool generate_disk_file(const char *file_name) {
  int fd = open(file_name, O_CREAT | O_WRONLY, 0666);
  if (fd < 0) {
    perror("open");
    return false;
  }
  close(fd);
  return true;
}

ptr_t Runtime::safe_cast(Context ctx, ptr_t pointer, LogicalRegion region) {
  return lrt->safe_cast(ctx, pointer, region);
}

DomainPoint Runtime::safe_cast(Context ctx, DomainPoint point, LogicalRegion region) {
  return lrt->safe_cast(ctx, point, region);
}

// Is this ok since the mapper fills in this future?
Future Runtime::select_tunable_value(Context ctx, const TunableLauncher &launcher) {
  if (!enabled) {
    return lrt->select_tunable_value(ctx, launcher);
  }

  if (replay && future_tag < max_future_tag) {
    return futures[future_tag++];
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

  if (replay && api_tag < max_api_tag) {
    api_tag++;
    return;
  }
  api_tag++;
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

void Runtime::save_logical_region(Context ctx, const Task *task,
                                  Legion::LogicalRegion &lr, const char *file_name) {
  log_resilience.info() << "save_logical_region: lr " << lr << " file_name " << file_name;
  bool ok = generate_disk_file(file_name);
  assert(ok);

  LogicalRegion cpy =
      lrt->create_logical_region(ctx, lr.get_index_space(), lr.get_field_space());

  std::vector<FieldID> fids;
  lrt->get_field_space_fields(lr.get_field_space(), fids);

  AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, cpy, false);
  al.attach_file(file_name, fids, LEGION_FILE_READ_WRITE);

  PhysicalRegion pr = lrt->attach_external_resource(ctx, al);

  CopyLauncher cl;
  cl.add_copy_requirements(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr),
                           RegionRequirement(cpy, READ_WRITE, EXCLUSIVE, cpy));

  for (long unsigned i = 0; i < fids.size(); i++) {
    if (i % task->get_total_shards() == task->get_shard_id()) {
      cl.add_src_field(0, fids[i]);
      cl.add_dst_field(0, fids[i]);
    }
  }

  // Index launch this?
  lrt->issue_copy_operation(ctx, cl);

  {
    Legion::Future f = lrt->detach_external_resource(ctx, pr);
    f.get_void_result(true);
  }
}

void resilient_write(const Task *task, const std::vector<PhysicalRegion> &regions,
                     Context ctx, Legion::Runtime *runtime) {
  const char *cstr = static_cast<char *>(task->args);
  std::string str(cstr, task->arglen);
  std::string tag = str.substr(0, str.find(","));
  std::string file_name = "checkpoint." + tag;
  file_name += ".dat";
  log_resilience.info() << "File name is " << file_name;
  std::ofstream file(file_name);
  file << str.substr(str.find(",") + 1, str.size());
  file.close();
}

void Runtime::checkpoint(Context ctx, const Task *task) {
  if (!enabled) {
    log_resilience.error()
        << "Must enable checkpointing with runtime->enable_checkpointing()";
    assert(false);
  }
  // FIXME (Elliott): we disable ALL checkpointing on replay??
  // (Should this be checkpoint_tag < max_checkpoint_tag?)
  // (No, that doesn't work. Must be some state that doesn't get re-initialized on replay)
  if (replay) return;

  log_resilience.info() << "In checkpoint: tag " << checkpoint_tag;
  log_resilience.info() << "Number of logical regions " << regions.size();

  char file_name[4096];
  int counter = 0;
  for (size_t i = 0; i < regions.size(); ++i) {
    auto &lr = regions[i];
    auto &state = region_state[i];

    if (state.destroyed) {
      log_resilience.info() << "Skipping region " << counter << " destroyed "
                            << state.destroyed;
      counter++;
      continue;
    }
    snprintf(file_name, sizeof(file_name), "checkpoint.%ld.lr.%d.dat", checkpoint_tag,
             counter);
    log_resilience.info() << "Saving region " << counter << " to file " << file_name;
    save_logical_region(ctx, task, lr, file_name);
    counter++;
  }

  log_resilience.info() << "Saved all logical regions!";

  max_api_tag = api_tag;
  max_future_tag = future_tag;
  max_future_map_tag = future_map_tag;
  max_index_space_tag = index_space_tag;
  max_region_tag = region_tag;
  max_partition_tag = partition_tag;
  max_checkpoint_tag = checkpoint_tag;

  for (auto &ft : futures) ft.setup_for_checkpoint();

  for (auto &fm : future_maps) fm.setup_for_checkpoint();

  // Do not need to setup index spaces

  for (auto &ip : partitions) ip.setup_for_checkpoint(ctx, lrt);

  std::stringstream serialized;
  {
    cereal::XMLOutputArchive oarchive(serialized);
    oarchive(*this);
  }
  std::string tmp = std::to_string(checkpoint_tag) + ",";
  tmp += serialized.str();
  const char *cstr = tmp.c_str();

  // FIXME (Elliott): static (library) task registration
  TaskID tid = lrt->generate_dynamic_task_id();
  {
    TaskVariantRegistrar registrar(tid, "resilient_write");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    lrt->register_task_variant<resilient_write>(registrar);
  }
  TaskLauncher resilient_write_launcher(tid, TaskArgument(cstr, strlen(cstr)));
  lrt->execute_task(ctx, resilient_write_launcher);

  checkpoint_tag++;
}

void Runtime::enable_checkpointing() {
  bool first_time = !enabled;
  enabled = true;
  if (!first_time) return;

  InputArgs args = Legion::Runtime::get_input_args();

  bool check = false;

  for (int i = 1; i < args.argc; i++) {
    if (strstr(args.argv[i], "-replay")) replay = true;
    if (strstr(args.argv[i], "-cpt")) {
      /* Ideally we'd go through a preset directory to find the latest
       * checkpoint. For now we require the user to tell us which checkpoint file
       * they want to use.
       */
      check = true;
      max_checkpoint_tag = atoi(args.argv[++i]);
    }
  }

  log_resilience.info() << "In enable_checkpointing: replay " << replay << " check "
                        << check << " max_checkpoint_tag " << max_checkpoint_tag;

  if (replay) {
    assert(check);
    char file_name[4096];
    snprintf(file_name, sizeof(file_name), "checkpoint.%ld.dat", max_checkpoint_tag);
    std::ifstream file(file_name);
    cereal::XMLInputArchive iarchive(file);
    iarchive(*this);
    file.close();
  }
}

Future Future::from_untyped_pointer(Runtime *runtime, const void *buffer, size_t bytes) {
  if (runtime->replay && runtime->future_tag < runtime->max_future_tag) {
    return runtime->futures[runtime->future_tag++];
  }
  Future f = Legion::Future::from_untyped_pointer(runtime->lrt, buffer, bytes);
  runtime->futures.push_back(f);
  runtime->future_tag++;
  return f;
}
