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
#include <cereal/archives/xml.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

#include "legion.h"

#define FRIEND_ALL_LEGION_RESILIENCE_CLASSES \
  friend class ResilientLegion::Future;      \
  friend class ResilientLegion::FutureMap

namespace ResilientLegion {

using Legion::Acquire;
using Legion::AcquireLauncher;
using Legion::AffineTransform;
using Legion::AlignmentConstraint;
using Legion::ArgumentMap;
using Legion::AttachLauncher;
using Legion::Close;
using Legion::CObjectWrapper;
using Legion::ColocationConstraint;
using Legion::Color;
using Legion::ColoredPoints;
using Legion::ColoringSerializer;
using Legion::Context;
using Legion::coord_t;
using Legion::Copy;
using Legion::CopyLauncher;
using Legion::DeferredBuffer;
using Legion::DeferredReduction;
using Legion::DeferredValue;
using Legion::DimensionKind;
using Legion::Domain;
using Legion::DomainAffineTransform;
using Legion::DomainColoringSerializer;
using Legion::DomainPoint;
using Legion::DomainScaleTransform;
using Legion::DomainT;
using Legion::DomainTransform;
using Legion::DynamicCollective;
using Legion::ExternalResources;
using Legion::FieldAccessor;
using Legion::FieldAllocator;
using Legion::FieldConstraint;
using Legion::FieldID;
using Legion::FieldSpace;
using Legion::FieldSpaceRequirement;
using Legion::Fill;
using Legion::FillLauncher;
using Legion::FutureFunctor;
using Legion::Grant;
using Legion::IndexAllocator;
using Legion::IndexAttachLauncher;
using Legion::IndexCopyLauncher;
using Legion::IndexFillLauncher;
using Legion::IndexIterator;
using Legion::IndexLauncher;
using Legion::IndexPartition;
using Legion::IndexPartitionT;
using Legion::IndexSpace;
using Legion::IndexSpaceRequirement;
using Legion::IndexSpaceT;
using Legion::IndexTaskLauncher;
using Legion::InlineLauncher;
using Legion::InlineMapping;
using Legion::InputArgs;
using Legion::LayoutConstraintID;
using Legion::LayoutConstraintRegistrar;
using Legion::LayoutConstraintSet;
using Legion::LegionHandshake;
using Legion::Lock;
using Legion::LockRequest;
using Legion::Logger;
using Legion::LogicalPartition;
using Legion::LogicalPartitionT;
using Legion::LogicalRegion;
using Legion::LogicalRegionT;
using Legion::Machine;
using Legion::Mappable;
using Legion::Memory;
using Legion::MemoryConstraint;
using Legion::MPILegionHandshake;
using Legion::MultiDomainPointColoring;
using Legion::MustEpoch;
using Legion::MustEpochLauncher;
using Legion::OrderingConstraint;
using Legion::Partition;
using Legion::PhaseBarrier;
using Legion::PhysicalRegion;
using Legion::PieceIterator;
using Legion::PieceIteratorT;
using Legion::Point;
using Legion::PointInDomainIterator;
using Legion::PointInRectIterator;
using Legion::Predicate;
using Legion::PredicateLauncher;
using Legion::Processor;
using Legion::ProcessorConstraint;
using Legion::ProjectionFunctor;
using Legion::ProjectionID;
using Legion::Rect;
using Legion::RectInDomainIterator;
using Legion::ReductionAccessor;
using Legion::ReductionOpID;
using Legion::RegionRequirement;
using Legion::RegionTreeID;
using Legion::RegistrationCallbackArgs;
using Legion::RegistrationCallbackFnptr;
using Legion::RegistrationWithArgsCallbackFnptr;
using Legion::Release;
using Legion::ReleaseLauncher;
using Legion::ScaleTransform;
using Legion::ShardingFunctor;
using Legion::ShardingID;
using Legion::Span;
using Legion::SpanIterator;
using Legion::SpecializedConstraint;
using Legion::StaticDependence;
using Legion::SumReduction;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskConfigOptions;
using Legion::TaskID;
using Legion::TaskLauncher;
using Legion::TaskVariantRegistrar;
using Legion::TimingLauncher;
using Legion::Transform;
using Legion::TunableLauncher;
using Legion::TypeTag;
using Legion::UnsafeFieldAccessor;
using Legion::Unserializable;
using Legion::UntypedBuffer;
using Legion::UntypedDeferredBuffer;
using Legion::UntypedDeferredValue;
using Legion::VariantID;

// Forward declaration
class Runtime;

class Future {
public:
  Legion::Future lft;
  std::vector<char> result;
  bool empty; /* Problematic with predicates? */
  bool is_fill;

  Future(Legion::Future lft_) : lft(lft_), empty(false), is_fill(false) {}
  Future() : lft(Legion::Future()), empty(true), is_fill(false) {}

  operator Legion::Future() const {
    // This is an invalid pointer during replay, but it should never actually
    // be used in a replay execution. So effectively this is only to satisfy
    // the type checker.
    return lft;
  }

  void setup_for_checkpoint() {
    if (is_fill) return;

    const void *ptr = lft.get_untyped_pointer();
    size_t size = lft.get_untyped_size();
    char *buf = (char *)ptr;
    std::vector<char> tmp(buf, buf + size);
    result = tmp;
  }

  /* Did this have to be declared const? */
  template <class T>
  inline T get_result(bool silence_warnings = false) {
    assert(!is_fill);
    if (!result.empty()) {
      return *reinterpret_cast<T *>(&result[0]);
    }
    const void *ptr = lft.get_untyped_pointer(silence_warnings);
    char *buf = (char *)ptr;
    std::vector<char> tmp(buf, buf + sizeof(T));
    result = tmp;
    return *static_cast<const T *>(ptr);
  }

  void get_void_result(bool silence_warnings = false, const char *warning_string = NULL) {
    assert(!is_fill);
    if (!result.empty()) return;
    lft.get_void_result(silence_warnings, warning_string);
  }

  static Future from_untyped_pointer(Runtime *runtime, const void *buffer, size_t bytes);

  template <typename T>
  static Future from_value(Runtime *runtime, const T &value);

  template <class Archive>
  void serialize(Archive &ar) {
    ar(empty, is_fill, result);
  }
};

class DomainPointSerializer {
public:
  DomainPoint p;

  DomainPointSerializer() = default;
  DomainPointSerializer(DomainPoint p_) : p(p_) {}

  operator DomainPoint() const { return p; }

  bool operator<(const DomainPointSerializer &o) const { return p < o.p; }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(p.dim, p.point_data);
  }
};

class DomainRectSerializer {
public:
  DomainPointSerializer lo, hi;

  DomainRectSerializer() = default;
  DomainRectSerializer(DomainPoint lo_, DomainPoint hi_) : lo(lo_), hi(hi_) {}

  operator Domain() const { return Domain(lo, hi); }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(lo, hi);
  }
};

class DomainSerializer {
public:
  int dim;
  std::vector<DomainRectSerializer> rects;

  DomainSerializer() = default;

  DomainSerializer(Domain domain) {
    dim = domain.get_dim();

    switch (dim) {
#define DIMFUNC(DIM)                                                            \
  case DIM: {                                                                   \
    for (RectInDomainIterator<DIM> i(domain); i(); i++) add_rect(i->lo, i->hi); \
    break;                                                                      \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        assert(false);
    }
  }

private:
  void add_rect(DomainPoint lo_, DomainPoint hi_) {
    DomainRectSerializer r(lo_, hi_);
    rects.push_back(r);
  }

public:
  template <class Archive>
  void serialize(Archive &ar) {
    ar(dim, rects);
  }
};

class ResilientIndexSpace {
public:
  DomainSerializer domain;

  ResilientIndexSpace() = default;
  ResilientIndexSpace(Domain d) : domain(d) {}

  template <class Archive>
  void serialize(Archive &ar) {
    ar(domain);
  }
};

class ResilientIndexPartition {
public:
  IndexPartition ip;
  ResilientIndexSpace color_space;
  std::map<DomainPointSerializer, ResilientIndexSpace> map;
  bool is_valid;

  ResilientIndexPartition() = default;
  ResilientIndexPartition(IndexPartition ip_) : ip(ip_), is_valid(true) {}

  void setup_for_checkpoint(Context ctx, Legion::Runtime *lrt);

  void save(Context ctx, Legion::Runtime *lrt, DomainPoint d);

  template <class Archive>
  void serialize(Archive &ar) {
    ar(color_space, map, is_valid);
  }
};

class FutureMap {
public:
  Legion::FutureMap fm;
  Domain d;
  std::map<DomainPointSerializer, std::vector<char>> map;

  FutureMap() = default;

  FutureMap(Legion::FutureMap fm_) : fm(fm_) {}

  FutureMap(Legion::FutureMap fm_, Domain d_) : fm(fm_), d(d_) {}

private:
  void get_and_save_result(DomainPoint dp) {
    Legion::Future ft = fm.get_future(dp);
    const void *ptr = ft.get_untyped_pointer();
    size_t size = ft.get_untyped_size();
    char *buf = (char *)ptr;
    std::vector<char> result(buf, buf + size);
    map[dp] = result;
  }

public:
  void setup_for_checkpoint() {
    for (Domain::DomainPointIterator i(d); i; ++i) {
      get_and_save_result(*i);
    }
  }

  template <typename T>
  T get_result(const DomainPoint &point, Runtime *runtime);

  void wait_all_results(Runtime *runtime);

  template <class Archive>
  void serialize(Archive &ar) {
    ar(map);
  }
};

class LogicalRegionState {
public:
  bool dirty, valid;  // FIXME (Elliott): valid == !destroyed

  LogicalRegionState() : dirty(false), valid(true) {}

  template <class Archive>
  void serialize(Archive &ar) {
    ar(dirty, valid);
  }
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

  static void add_registration_callback(
      void (*FUNC)(Machine machine, Runtime *runtime,
                   const std::set<Processor> &local_procs),
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

  Future execute_task(Context, TaskLauncher);

  FutureMap execute_index_space(Context, const IndexTaskLauncher &launcher);
  Future execute_index_space(Context, const IndexTaskLauncher &launcher,
                             ReductionOpID redop, bool deterministic = false);

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

  Future get_current_time(Context, Future = Legion::Future());
  Future get_current_time_in_microseconds(Context, Future = Legion::Future());

  Predicate create_predicate(Context ctx, const Future &f);
  Predicate create_predicate(Context ctx, const PredicateLauncher &launcher);

  Predicate predicate_not(Context ctx, const Predicate &p);

  Future get_predicate_future(Context ctx, const Predicate &p);

  template <int DIM, typename COORD_T>
  IndexSpaceT<DIM, COORD_T> create_index_space(Context ctx,
                                               const Rect<DIM, COORD_T> &bounds) {
    if (replay && index_space_tag < max_index_space_tag) {
      IndexSpace is = restore_index_space(ctx);
      return static_cast<IndexSpaceT<DIM, COORD_T>>(is);
    }
    IndexSpace is = lrt->create_index_space(ctx, bounds);
    ResilientIndexSpace ris(lrt->get_index_space_domain(ctx, is));
    index_spaces.push_back(ris);
    index_space_tag++;
    return static_cast<IndexSpaceT<DIM, COORD_T>>(is);
  }

  IndexSpace create_index_space(Context, const Domain &);

  IndexSpace create_index_space_union(Context ctx, IndexPartition parent,
                                      const DomainPoint &color,
                                      const std::vector<IndexSpace> &handles);

  IndexSpace create_index_space_union(Context ctx, IndexPartition parent,
                                      const DomainPoint &color, IndexPartition handle);

  IndexSpace create_index_space_difference(Context ctx, IndexPartition parent,
                                           const DomainPoint &color, IndexSpace initial,
                                           const std::vector<IndexSpace> &handles);

  FieldSpace create_field_space(Context ctx);

  FieldAllocator create_field_allocator(Context ctx, FieldSpace handle);

  LogicalRegion create_logical_region(Context ctx, IndexSpace index, FieldSpace fields,
                                      bool task_local = false,
                                      const char *provenance = NULL);

  template <int DIM, typename COORD_T>
  LogicalRegion create_logical_region(Context ctx, IndexSpaceT<DIM, COORD_T> index,
                                      FieldSpace fields) {
    return create_logical_region(
        ctx, static_cast<IndexSpace>(index),
        static_cast<FieldSpace>(fields));  // Isn't this cast redundant?
  }

  PhysicalRegion map_region(Context ctx, const InlineLauncher &launcher);

  void unmap_region(Context ctx, PhysicalRegion region);

  void destroy_index_space(Context ctx, IndexSpace handle);

  void destroy_field_space(Context ctx, FieldSpace handle);

  void destroy_logical_region(Context ctx, LogicalRegion handle);

  void destroy_index_partition(Context ctx, IndexPartition handle);

  IndexPartition create_equal_partition(Context ctx, IndexSpace parent,
                                        IndexSpace color_space);

  IndexPartition create_pending_partition(Context ctx, IndexSpace parent,
                                          IndexSpace color_space);

  Color create_cross_product_partitions(Context ctx, IndexPartition handle1,
                                        IndexPartition handle2,
                                        std::map<IndexSpace, IndexPartition> &handles);

  IndexPartition create_partition_by_field(Context ctx, LogicalRegion handle,
                                           LogicalRegion parent, FieldID fid,
                                           IndexSpace color_space);

  IndexPartition create_partition_by_image(Context ctx, IndexSpace handle,
                                           LogicalPartition projection,
                                           LogicalRegion parent, FieldID fid,
                                           IndexSpace color_space);

  IndexPartition create_partition_by_preimage(Context ctx, IndexPartition projection,
                                              LogicalRegion handle, LogicalRegion parent,
                                              FieldID fid, IndexSpace color_space);

  IndexPartition create_partition_by_difference(Context ctx, IndexSpace parent,
                                                IndexPartition handle1,
                                                IndexPartition handle2,
                                                IndexSpace color_space);

  template <int DIM, int COLOR_DIM, typename COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_restriction(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent,
      IndexSpaceT<COLOR_DIM, COORD_T> color_space,
      Transform<DIM, COLOR_DIM, COORD_T> transform, Rect<DIM, COORD_T> extent) {
    if (replay && !partitions[partition_tag].is_valid) {
      partition_tag++;
      return static_cast<IndexPartitionT<DIM, COORD_T>>(IndexPartition::NO_PART);
    }

    if (replay && partition_tag < max_partition_tag) {
      return static_cast<IndexPartitionT<DIM, COORD_T>>(restore_index_partition(
          ctx, static_cast<IndexSpace>(parent), static_cast<IndexSpace>(color_space)));
    }

    IndexPartitionT<DIM, COORD_T> ip =
        lrt->create_partition_by_restriction(ctx, parent, color_space, transform, extent);
    partition_tag++;
    partitions.push_back(static_cast<ResilientIndexPartition>(ip));
    return ip;
  }

  template <int DIM, typename COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_blockify(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent, Point<DIM, COORD_T> blocking_factor,
      Color color = LEGION_AUTO_GENERATE_ID) {
    if (replay && !partitions[partition_tag].is_valid) {
      partition_tag++;
      return static_cast<IndexPartitionT<DIM, COORD_T>>(IndexPartition::NO_PART);
      // return IndexPartition::NO_PART;
    }

    // FIXME
    if (replay && partition_tag < max_partition_tag) {
      return static_cast<IndexPartitionT<DIM, COORD_T>>(restore_index_partition(
          ctx, static_cast<IndexSpace>(parent), static_cast<IndexSpace>(parent)));
    }

    IndexPartitionT<DIM, COORD_T> ip =
        lrt->create_partition_by_blockify(ctx, parent, blocking_factor, color);
    partition_tag++;
    partitions.push_back(static_cast<ResilientIndexPartition>(ip));
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
                                               DomainPoint c);

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

  // FIXME: Use api_tag instead
  template <typename T>
  void fill_field(Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
                  const T &value, Predicate pred = Predicate::TRUE_PRED) {
    if (replay && future_tag < max_future_tag) {
      std::cout << "No-oping this fill\n";
      future_tag++;
      return;
    }
    lrt->fill_field<T>(ctx, handle, parent, fid, value);
    future_tag++;
    /* We have to push something into the vector here because future_tag gets
     * out of sync with the vector otherwise. And the user never sees this
     * ResilientFuture so we're fine. */
    Future ft;
    ft.is_fill = true;
    futures.push_back(ft);
  }

  Future select_tunable_value(Context ctx, const TunableLauncher &launcher);

  void fill_fields(Context ctx, const FillLauncher &launcher);

  void get_field_space_fields(FieldSpace handle, std::set<FieldID> &fields);

  size_t get_field_size(FieldSpace handle, FieldID fid);

  Processor get_executing_processor(Context ctx);

  void print_once(Context ctx, FILE *f, const char *message);

  void save_logical_region(Context ctx, const Task *task, Legion::LogicalRegion &lr,
                           const char *file_name);

  void save_index_partition(Context ctx, IndexSpace color_space, IndexPartition ip);

  IndexSpace restore_index_space(Context ctx);

  IndexPartition restore_index_partition(Context ctx, IndexSpace index_space,
                                         IndexSpace color_space);

public:
  // Checkpointing methods
  void enable_checkpointing();

  void checkpoint(Context ctx, const Task *task);

public:
  // Serialization methods
  template <class Archive>
  void serialize(Archive &ar) {
    ar(max_api_tag, max_future_tag, max_future_map_tag, max_region_tag,
       max_index_space_tag, max_partition_tag, futures, future_maps, region_state,
       index_spaces, partitions);
  }

private:
  // Internal methods
  bool resolve_predicate(Context ctx, const Predicate &p);
  void track_region_state(const RegionRequirement &rr);

private:
  Legion::Runtime *lrt;

  bool enabled, replay;
  std::vector<Future> futures;
  std::vector<ResilientIndexSpace> index_spaces;
  std::vector<LogicalRegion> regions;  // Not persisted
  std::vector<LogicalRegionState> region_state;
  std::vector<ResilientIndexPartition> partitions;
  std::vector<FutureMap> future_maps;
  long unsigned api_tag, future_tag, future_map_tag, index_space_tag, region_tag,
      partition_tag, checkpoint_tag;
  long unsigned max_api_tag, max_future_tag, max_future_map_tag, max_index_space_tag,
      max_region_tag, max_partition_tag, max_checkpoint_tag;

  FRIEND_ALL_LEGION_RESILIENCE_CLASSES;
};

}  // namespace ResilientLegion

#include "legion_resilience.inl"

#endif  // RESILIENCE_H
