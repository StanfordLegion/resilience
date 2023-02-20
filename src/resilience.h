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

#ifndef RESILIENCE_H
#define RESILIENCE_H

#include <fcntl.h>
#include <unistd.h>

#include <cassert>

#include "legion.h"
#ifdef DEBUG_LEGION
#include <cereal/archives/xml.hpp>
#else
#include <cereal/archives/binary.hpp>
#endif
#include <cereal/types/array.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

#include "resilience/future.h"
#include "resilience/serializer.h"
#include "resilience/types.h"

namespace ResilientLegion {

// A covering set is a collection of partitions and regions that are pairwise disjoint,
// and the union of which covers the entire root of the region tree.
class CoveringSet {
public:
  std::set<LogicalPartition> partitions;
  std::set<LogicalRegion> regions;
  unsigned depth;

  CoveringSet() : depth(0) {}
};

class RegionTreeState {
public:
  // Recently-used, disjoint and complete partitions are good candidates to save.
  std::map<LogicalRegion, LogicalPartition> recent_partitions;

  RegionTreeState() = default;
};

class IndexPartitionTreeState {
public:
  // Is the partition: (a) disjoint, (b) complete, (c) all parents complete?
  // We do not track parent disjointness because we can save overlapping subsets, as long
  // as they are complete in aggregate.
  bool eligible;

  IndexPartitionTreeState() : eligible(false) {}
};

class Runtime {
public:
  // Constructors
  Runtime(Legion::Runtime *);

public:
  // Wrapper methods
  void attach_name(FieldSpace handle, const char *name, bool is_mutable = false);
  void attach_name(FieldSpace handle, FieldID fid, const char *name,
                   bool is_mutable = false);
  void attach_name(IndexSpace handle, const char *name, bool is_mutable = false);
  void attach_name(LogicalRegion handle, const char *name, bool is_mutable = false);
  void attach_name(IndexPartition handle, const char *name, bool is_mutable = false);
  void attach_name(LogicalPartition handle, const char *name, bool is_mutable = false);

  void issue_execution_fence(Context ctx, const char *provenance = NULL);

  void issue_copy_operation(Context ctx, const CopyLauncher &launcher);

  void issue_copy_operation(Context ctx, const IndexCopyLauncher &launcher);

  template <typename REDOP>
  static void register_reduction_op(ReductionOpID redop_id,
                                    bool permit_duplicates = false) {
    Legion::Runtime::register_reduction_op<REDOP>(redop_id, permit_duplicates);
  }

  static void add_registration_callback(RegistrationCallbackFnptr callback,
                                        bool dedup = true, size_t dedup_tag = 0);

  static void set_registration_callback(RegistrationCallbackFnptr callback);

  static const InputArgs &get_input_args(void);

  static void set_top_level_task_id(TaskID top_id);

  static void preregister_projection_functor(ProjectionID pid,
                                             ProjectionFunctor *functor);

  static void preregister_sharding_functor(ShardingID sid, ShardingFunctor *functor);

  template <void (*TASK_PTR)(const Task *task, const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)>
  static void task_wrapper_void(const Task *task_,
                                const std::vector<PhysicalRegion> &regions_, Context ctx_,
                                Legion::Runtime *runtime_) {
    Runtime new_runtime_(runtime_);
    Runtime *new_runtime = &new_runtime_;
    TASK_PTR(task_, regions_, ctx_, new_runtime);
  }

  template <void (*TASK_PTR)(const Task *, const std::vector<PhysicalRegion> &, Context,
                             Runtime *)>
  static VariantID preregister_task_variant(const TaskVariantRegistrar &registrar,
                                            const char *task_name = NULL,
                                            VariantID vid = LEGION_AUTO_GENERATE_ID) {
    return Legion::Runtime::preregister_task_variant<task_wrapper_void<TASK_PTR>>(
        registrar, task_name, vid);
  }

  template <typename T,
            T (*TASK_PTR)(const Task *task, const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)>
  static T task_wrapper(const Task *task_, const std::vector<PhysicalRegion> &regions_,
                        Context ctx_, Legion::Runtime *runtime_) {
    Runtime new_runtime_(runtime_);
    Runtime *new_runtime = &new_runtime_;
    return TASK_PTR(task_, regions_, ctx_, new_runtime);
  }

  template <typename T, T (*TASK_PTR)(const Task *, const std::vector<PhysicalRegion> &,
                                      Context, Runtime *)>
  static VariantID preregister_task_variant(const TaskVariantRegistrar &registrar,
                                            const char *task_name = NULL,
                                            VariantID vid = LEGION_AUTO_GENERATE_ID) {
    return Legion::Runtime::preregister_task_variant<T, task_wrapper<T, TASK_PTR>>(
        registrar, task_name, vid);
  }

  static LayoutConstraintID preregister_layout(
      const LayoutConstraintRegistrar &registrar,
      LayoutConstraintID layout_id = LEGION_AUTO_GENERATE_ID);

  static int start(int argc, char **argv, bool background = false,
                   bool supply_default_mapper = true);

  Future execute_task(Context, TaskLauncher,
                      std::vector<OutputRequirement> *outputs = NULL);

  FutureMap execute_index_space(Context, const IndexTaskLauncher &launcher,
                                std::vector<OutputRequirement> *outputs = NULL);
  Future execute_index_space(Context, const IndexTaskLauncher &launcher,
                             ReductionOpID redop, bool deterministic = false,
                             std::vector<OutputRequirement> *outputs = NULL);

  Domain get_index_space_domain(Context, IndexSpace);
  Domain get_index_space_domain(IndexSpace);

  Domain get_index_partition_color_space(Context ctx, IndexPartition p);

  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  DomainT<COLOR_DIM, COLOR_COORD_T> get_index_partition_color_space(
      IndexPartitionT<DIM, COORD_T> p) {
    return lrt->get_index_partition_color_space<DIM, COORD_T, COLOR_DIM, COLOR_COORD_T>(
        p);
  }

  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  IndexSpaceT<COLOR_DIM, COLOR_COORD_T> get_index_partition_color_space_name(
      IndexPartitionT<DIM, COORD_T> p) {
    return lrt
        ->get_index_partition_color_space_name<DIM, COORD_T, COLOR_DIM, COLOR_COORD_T>(p);
  }

  Future get_current_time(Context ctx, Future precondition = Legion::Future());
  Future get_current_time_in_microseconds(Context ctx,
                                          Future precondition = Legion::Future());
  Future get_current_time_in_nanoseconds(Context ctx,
                                         Future precondition = Legion::Future());

  Predicate create_predicate(Context ctx, const Future &f, const char *provenance = NULL);
  Predicate create_predicate(Context ctx, const PredicateLauncher &launcher);

  Predicate predicate_not(Context ctx, const Predicate &p, const char *provenance = NULL);

  Future get_predicate_future(Context ctx, const Predicate &p,
                              const char *provenance = NULL);

  template <int DIM, typename COORD_T>
  IndexSpaceT<DIM, COORD_T> create_index_space(Context ctx,
                                               const Rect<DIM, COORD_T> &bounds,
                                               const char *provenance = NULL) {
    if (!enabled) {
      return lrt->create_index_space(ctx, bounds, provenance);
    }

    if (replay && index_space_tag < state.max_index_space_tag) {
      IndexSpace is = restore_index_space(ctx, provenance);
      return static_cast<IndexSpaceT<DIM, COORD_T>>(is);
    }

    IndexSpace is = lrt->create_index_space(ctx, bounds, provenance);
    ispaces.push_back(is);
    index_space_tag++;
    return static_cast<IndexSpaceT<DIM, COORD_T>>(is);
  }

  IndexSpace create_index_space(Context ctx, const Domain &bounds, TypeTag type_tag = 0,
                                const char *provenance = NULL);

  IndexSpace create_index_space_union(Context ctx, IndexPartition parent,
                                      const DomainPoint &color,
                                      const std::vector<IndexSpace> &handles,
                                      const char *provenance = NULL);

  IndexSpace create_index_space_union(Context ctx, IndexPartition parent,
                                      const DomainPoint &color, IndexPartition handle,
                                      const char *provenance = NULL);

  IndexSpace create_index_space_difference(Context ctx, IndexPartition parent,
                                           const DomainPoint &color, IndexSpace initial,
                                           const std::vector<IndexSpace> &handles,
                                           const char *provenance = NULL);

  FieldSpace create_field_space(Context ctx, const char *provenance = NULL);

  FieldAllocator create_field_allocator(Context ctx, FieldSpace handle);

  LogicalRegion create_logical_region(Context ctx, IndexSpace index, FieldSpace fields,
                                      bool task_local = false,
                                      const char *provenance = NULL);

  template <int DIM, typename COORD_T>
  LogicalRegion create_logical_region(Context ctx, IndexSpaceT<DIM, COORD_T> index,
                                      FieldSpace fields, bool task_local = false,
                                      const char *provenance = NULL) {
    return create_logical_region(ctx, static_cast<IndexSpace>(index), fields, task_local,
                                 provenance);
  }

  PhysicalRegion map_region(Context ctx, const InlineLauncher &launcher);

  void unmap_region(Context ctx, PhysicalRegion region);

  void destroy_index_space(Context ctx, IndexSpace handle, const bool unordered = false,
                           const bool recurse = true, const char *provenance = NULL);

  void destroy_field_space(Context ctx, FieldSpace handle, const bool unordered = false,
                           const char *provenance = NULL);

  void destroy_logical_region(Context ctx, LogicalRegion handle,
                              const bool unordered = false,
                              const char *provenance = NULL);

  void destroy_index_partition(Context ctx, IndexPartition handle,
                               const bool unordered = false, const bool recurse = true,
                               const char *provenance = NULL);

  IndexPartition create_equal_partition(Context ctx, IndexSpace parent,
                                        IndexSpace color_space, size_t granularity = 1,
                                        Color color = LEGION_AUTO_GENERATE_ID,
                                        const char *provenance = NULL);

  IndexPartition create_pending_partition(Context ctx, IndexSpace parent,
                                          IndexSpace color_space,
                                          PartitionKind part_kind = LEGION_COMPUTE_KIND,
                                          Color color = LEGION_AUTO_GENERATE_ID,
                                          const char *provenance = NULL);

  Color create_cross_product_partitions(Context ctx, IndexPartition handle1,
                                        IndexPartition handle2,
                                        std::map<IndexSpace, IndexPartition> &handles,
                                        PartitionKind part_kind = LEGION_COMPUTE_KIND,
                                        Color color = LEGION_AUTO_GENERATE_ID,
                                        const char *provenance = NULL);

  IndexPartition create_partition_by_field(
      Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
      IndexSpace color_space, Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0,
      MappingTagID tag = 0, PartitionKind part_kind = LEGION_DISJOINT_KIND,
      UntypedBuffer map_arg = UntypedBuffer(), const char *provenance = NULL);

  IndexPartition create_partition_by_image(
      Context ctx, IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
      FieldID fid, IndexSpace color_space, PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0, MappingTagID tag = 0,
      UntypedBuffer map_arg = UntypedBuffer(), const char *provenance = NULL);

  IndexPartition create_partition_by_preimage(
      Context ctx, IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
      FieldID fid, IndexSpace color_space, PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0, MappingTagID tag = 0,
      UntypedBuffer map_arg = UntypedBuffer(), const char *provenance = NULL);

  IndexPartition create_partition_by_difference(
      Context ctx, IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
      IndexSpace color_space, PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);

  template <int DIM, int COLOR_DIM, typename COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_restriction(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent,
      IndexSpaceT<COLOR_DIM, COORD_T> color_space,
      Transform<DIM, COLOR_DIM, COORD_T> transform, Rect<DIM, COORD_T> extent,
      PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL) {
    if (!enabled) {
      return lrt->create_partition_by_restriction(ctx, parent, color_space, transform,
                                                  extent, part_kind, color, provenance);
    }

    if (replay && partition_tag < state.max_partition_tag) {
      return static_cast<IndexPartitionT<DIM, COORD_T>>(
          restore_index_partition(ctx, static_cast<IndexSpace>(parent),
                                  static_cast<IndexSpace>(color_space), provenance));
    }

    IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_restriction(
        ctx, parent, color_space, transform, extent, part_kind, color, provenance);
    register_index_partition(ip);
    return ip;
  }

  template <int DIM, typename COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_blockify(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent, Point<DIM, COORD_T> blocking_factor,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL) {
    if (!enabled) {
      return lrt->create_partition_by_blockify(ctx, parent, blocking_factor, color,
                                               provenance);
    }

    if (replay && partition_tag < state.max_partition_tag) {
      return static_cast<IndexPartitionT<DIM, COORD_T>>(
          restore_index_partition(ctx, static_cast<IndexSpace>(parent),
                                  static_cast<IndexSpace>(parent), provenance));
    }

    IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_blockify(
        ctx, parent, blocking_factor, color, provenance);
    register_index_partition(ip);
    return ip;
  }

  LogicalPartition get_logical_partition(Context ctx, LogicalRegion parent,
                                         IndexPartition handle);

  LogicalPartition get_logical_partition(LogicalRegion parent, IndexPartition handle);

  LogicalPartition get_logical_partition_by_tree(IndexPartition handle, FieldSpace fspace,
                                                 RegionTreeID tid);

  LogicalRegion get_logical_subregion_by_color(Context ctx, LogicalPartition parent,
                                               Color c);

  LogicalRegion get_logical_subregion_by_color(Context ctx, LogicalPartition parent,
                                               const DomainPoint &c);

  LogicalRegion get_logical_subregion_by_color(LogicalPartition parent,
                                               const DomainPoint &c);

  Legion::Mapping::MapperRuntime *get_mapper_runtime(void);

  void replace_default_mapper(Legion::Mapping::Mapper *mapper,
                              Processor proc = Processor::NO_PROC);

  ptr_t safe_cast(Context ctx, ptr_t pointer, LogicalRegion region);

  DomainPoint safe_cast(Context ctx, DomainPoint point, LogicalRegion region);

  template <int DIM, typename COORD_T>
  bool safe_cast(Context ctx, Point<DIM, COORD_T> point,
                 LogicalRegionT<DIM, COORD_T> region) {
    return lrt->safe_cast(ctx, point, region);
  }

  template <typename T>
  void fill_field(Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
                  const T &value, Predicate pred = Predicate::TRUE_PRED) {
    fill_field(ctx, handle, parent, fid, &value, sizeof(T), pred);
  }

  void fill_field(Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
                  const void *value, size_t value_size,
                  Predicate pred = Predicate::TRUE_PRED);

  Future select_tunable_value(Context ctx, const TunableLauncher &launcher);

  void fill_fields(Context ctx, const FillLauncher &launcher);

  void get_field_space_fields(FieldSpace handle, std::set<FieldID> &fields);

  size_t get_field_size(FieldSpace handle, FieldID fid);

  Processor get_executing_processor(Context ctx);

  void print_once(Context ctx, FILE *f, const char *message);

public:
  // Checkpointing methods
  void enable_checkpointing(Context ctx);

  void checkpoint(Context ctx);

private:
  // Internal methods
  IndexSpace restore_index_space(Context ctx, const char *provenance);
  IndexPartition restore_index_partition(Context ctx, IndexSpace index_space,
                                         IndexSpace color_space, const char *provenance);
  void register_index_partition(IndexPartition ip);
  bool is_partition_eligible(IndexPartition ip);
  void track_region_state(const RegionRequirement &rr);
  void initialize_region(Context ctx, LogicalRegion r);
  void compute_covering_set(LogicalRegion r, CoveringSet &covering_set);
  Path compute_region_path(LogicalRegion lr, LogicalRegion parent);
  Path compute_partition_path(LogicalPartition lp);
  LogicalRegion lookup_region_path(LogicalRegion root, const Path &path);
  LogicalPartition lookup_partition_path(LogicalRegion root, const Path &path);
  void restore_region_content(Context ctx, LogicalRegion r);
  void restore_region(Context ctx, LogicalRegion lr, LogicalRegion parent,
                      LogicalRegion cpy, const std::vector<FieldID> &fids,
                      resilient_tag_t tag, const PathSerializer &path);
  void restore_partition(Context ctx, LogicalPartition lp, LogicalRegion parent,
                         LogicalRegion cpy, const std::vector<FieldID> &fids,
                         resilient_tag_t tag, const PathSerializer &path);
  void save_region(Context ctx, LogicalRegion lr, LogicalRegion parent, LogicalRegion cpy,
                   const std::vector<FieldID> &fids, resilient_tag_t tag,
                   const PathSerializer &path);
  void save_partition(Context ctx, LogicalPartition lp, LogicalRegion parent,
                      LogicalRegion cpy, const std::vector<FieldID> &fids,
                      resilient_tag_t tag, const PathSerializer &path);
  void save_region_content(Context ctx, LogicalRegion r);
  bool skip_api_call();

private:
  Legion::Runtime *lrt;

  bool enabled, replay;
  // FIXME (Elliott): make these all maps
  std::vector<Future> futures;
  std::vector<FutureMap> future_maps;
  std::vector<IndexSpace> ispaces;
  std::map<LogicalRegion, resilient_tag_t> region_tags;
  std::vector<LogicalRegion> regions;
  std::vector<RegionTreeState> region_tree_state;
  std::vector<IndexPartition> ipartitions;
  std::map<IndexPartition, resilient_tag_t> ipartition_tags;
  std::map<IndexPartition, IndexPartitionTreeState> ipartition_state;
  resilient_tag_t api_tag, future_tag, future_map_tag, index_space_tag, region_tag,
      partition_tag, checkpoint_tag;
  resilient_tag_t max_api_tag, max_future_tag, max_future_map_tag, max_index_space_tag,
      max_region_tag, max_partition_tag, max_checkpoint_tag;

  CheckpointState state;

private:
  static bool initial_replay;
  static resilient_tag_t initial_checkpoint_tag;
  static TaskID write_checkpoint_task_id;

  friend class Future;
  friend class FutureMap;
  friend class FutureMapSerializer;
  friend class IndexSpaceSerializer;
  friend class IndexPartitionSerializer;
  friend class Path;
};

}  // namespace ResilientLegion

#include "resilience/future.inl"
#include "resilience/serializer.inl"

#endif  // RESILIENCE_H
