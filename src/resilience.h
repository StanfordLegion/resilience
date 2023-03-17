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
#include "resilience/launcher.h"
#include "resilience/mapper.h"
#include "resilience/projection.h"
#include "resilience/serializer.h"
#include "resilience/types.h"

namespace ResilientLegion {

class FutureState {
public:
  size_t ref_count;
  bool escaped;
  resilient_tag_t moved_to;

  FutureState() : ref_count(0), escaped(false), moved_to(SIZE_MAX) {}
};

class FutureMapState {
public:
  size_t ref_count;
  bool escaped;

  FutureMapState() : ref_count(0), escaped(false) {}
};

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
  // Constructors, destructors
  Runtime(Legion::Runtime *);
  ~Runtime();

public:
  // Wrapper methods
  IndexSpace create_index_space(Context ctx, const Domain &bounds, TypeTag type_tag = 0,
                                const char *provenance = NULL);
  template <int DIM, typename COORD_T>
  IndexSpaceT<DIM, COORD_T> create_index_space(Context ctx,
                                               const Rect<DIM, COORD_T> &bounds,
                                               const char *provenance = NULL);
  IndexSpace create_index_space(Context ctx, const std::vector<DomainPoint> &points,
                                const char *provenance = NULL);
  IndexSpace create_index_space(Context ctx, size_t dimensions, const Future &f,
                                TypeTag type_tag = 0, const char *provenance = NULL);
  template <int DIM, typename COORD_T>
  IndexSpaceT<DIM, COORD_T> create_index_space(Context ctx, const Future &f,
                                               const char *provenance = NULL);
  template <int DIM, typename COORD_T>
  IndexSpaceT<DIM, COORD_T> create_index_space(
      Context ctx, const std::vector<Point<DIM, COORD_T>> &points,
      const char *provenance = NULL);

  IndexSpace union_index_spaces(Context ctx, const std::vector<IndexSpace> &spaces,
                                const char *provenance = NULL);
  template <int DIM, typename COORD_T>
  IndexSpaceT<DIM, COORD_T> union_index_spaces(
      Context ctx, const std::vector<IndexSpaceT<DIM, COORD_T>> &spaces,
      const char *provenance = NULL);

  IndexSpace intersect_index_spaces(Context ctx, const std::vector<IndexSpace> &spaces,
                                    const char *provenance = NULL);
  template <int DIM, typename COORD_T>
  IndexSpaceT<DIM, COORD_T> intersect_index_spaces(
      Context ctx, const std::vector<IndexSpaceT<DIM, COORD_T>> &spaces,
      const char *provenance = NULL);

  IndexSpace subtract_index_spaces(Context ctx, IndexSpace left, IndexSpace right,
                                   const char *provenance = NULL);
  template <int DIM, typename COORD_T>
  IndexSpaceT<DIM, COORD_T> subtract_index_spaces(Context ctx,
                                                  IndexSpaceT<DIM, COORD_T> left,
                                                  IndexSpaceT<DIM, COORD_T> right,
                                                  const char *provenance = NULL);

  IndexSpace create_index_space(Context ctx, size_t max_num_elmts);

  void create_shared_ownership(Context ctx, IndexSpace handle);

  void destroy_index_space(Context ctx, IndexSpace handle, const bool unordered = false,
                           const bool recurse = true, const char *provenance = NULL);

  IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                        const Domain &color_space,
                                        const PointColoring &coloring,
                                        PartitionKind part_kind = LEGION_COMPUTE_KIND,
                                        Color color = LEGION_AUTO_GENERATE_ID,
                                        bool allocable = false);
  IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                        const Coloring &coloring, bool disjoint,
                                        Color color = LEGION_AUTO_GENERATE_ID);
  IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                        const Domain &color_space,
                                        const DomainPointColoring &coloring,
                                        PartitionKind part_kind = LEGION_COMPUTE_KIND,
                                        Color color = LEGION_AUTO_GENERATE_ID);
  IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                        Domain color_space,
                                        const DomainColoring &coloring, bool disjoint,
                                        Color color = LEGION_AUTO_GENERATE_ID);
  IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                        const Domain &color_space,
                                        const MultiDomainPointColoring &coloring,
                                        PartitionKind part_kind = LEGION_COMPUTE_KIND,
                                        Color color = LEGION_AUTO_GENERATE_ID);
  IndexPartition create_index_partition(Context ctx, IndexSpace parent,
                                        Domain color_space,
                                        const MultiDomainColoring &coloring,
                                        bool disjoint,
                                        Color color = LEGION_AUTO_GENERATE_ID);

  void create_shared_ownership(Context ctx, IndexPartition handle);

  void destroy_index_partition(Context ctx, IndexPartition handle,
                               const bool unordered = false, const bool recurse = true,
                               const char *provenance = NULL);

  IndexPartition create_equal_partition(Context ctx, IndexSpace parent,
                                        IndexSpace color_space, size_t granularity = 1,
                                        Color color = LEGION_AUTO_GENERATE_ID,
                                        const char *provenance = NULL);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  IndexPartitionT<DIM, COORD_T> create_equal_partition(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent,
      IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, size_t granularity = 1,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);

  IndexPartition create_partition_by_weights(Context ctx, IndexSpace parent,
                                             const std::map<DomainPoint, int> &weights,
                                             IndexSpace color_space,
                                             size_t granularity = 1,
                                             Color color = LEGION_AUTO_GENERATE_ID,
                                             const char *provenance = NULL);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_weights(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent,
      const std::map<Point<COLOR_DIM, COLOR_COORD_T>, int> &weights,
      IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, size_t granularity = 1,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);
  IndexPartition create_partition_by_weights(Context ctx, IndexSpace parent,
                                             const std::map<DomainPoint, size_t> &weights,
                                             IndexSpace color_space,
                                             size_t granularity = 1,
                                             Color color = LEGION_AUTO_GENERATE_ID,
                                             const char *provenance = NULL);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_weights(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent,
      const std::map<Point<COLOR_DIM, COLOR_COORD_T>, size_t> &weights,
      IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, size_t granularity = 1,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);
  IndexPartition create_partition_by_weights(Context ctx, IndexSpace parent,
                                             const FutureMap &weights,
                                             IndexSpace color_space,
                                             size_t granularity = 1,
                                             Color color = LEGION_AUTO_GENERATE_ID,
                                             const char *provenance = NULL);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_weights(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent, const FutureMap &weights,
      IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, size_t granularity = 1,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);

  IndexPartition create_partition_by_union(Context ctx, IndexSpace parent,
                                           IndexPartition handle1, IndexPartition handle2,
                                           IndexSpace color_space,
                                           PartitionKind part_kind = LEGION_COMPUTE_KIND,
                                           Color color = LEGION_AUTO_GENERATE_ID,
                                           const char *provenance = NULL);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_union(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent,
      IndexPartitionT<DIM, COORD_T> handle1, IndexPartitionT<DIM, COORD_T> handle2,
      IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
      PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);

  IndexPartition create_partition_by_intersection(
      Context ctx, IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
      IndexSpace color_space, PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_intersection(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent,
      IndexPartitionT<DIM, COORD_T> handle1, IndexPartitionT<DIM, COORD_T> handle2,
      IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
      PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);
  IndexPartition create_partition_by_intersection(
      Context ctx, IndexSpace parent, IndexPartition partition,
      PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, bool dominates = false,
      const char *provenance = NULL);
  template <int DIM, typename COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_intersection(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent,
      IndexPartitionT<DIM, COORD_T> partition,
      PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, bool dominates = false,
      const char *provenance = NULL);

  IndexPartition create_partition_by_difference(
      Context ctx, IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
      IndexSpace color_space, PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_difference(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent,
      IndexPartitionT<DIM, COORD_T> handle1, IndexPartitionT<DIM, COORD_T> handle2,
      IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
      PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);

  Color create_cross_product_partitions(Context ctx, IndexPartition handle1,
                                        IndexPartition handle2,
                                        std::map<IndexSpace, IndexPartition> &handles,
                                        PartitionKind part_kind = LEGION_COMPUTE_KIND,
                                        Color color = LEGION_AUTO_GENERATE_ID,
                                        const char *provenance = NULL);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  Color create_cross_product_partitions(
      Context ctx, IndexPartitionT<DIM, COORD_T> handle1,
      IndexPartitionT<DIM, COORD_T> handle2,
      typename std::map<IndexSpaceT<DIM, COORD_T>, IndexPartitionT<DIM, COORD_T>>
          &handles,
      PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);

  IndexPartition create_partition_by_restriction(
      Context ctx, IndexSpace parent, IndexSpace color_space, DomainTransform transform,
      Domain extent, PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);
  template <int DIM, int COLOR_DIM, typename COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_restriction(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent,
      IndexSpaceT<COLOR_DIM, COORD_T> color_space,
      Transform<DIM, COLOR_DIM, COORD_T> transform, Rect<DIM, COORD_T> extent,
      PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);

  IndexPartition create_partition_by_blockify(Context ctx, IndexSpace parent,
                                              DomainPoint blocking_factor,
                                              Color color = LEGION_AUTO_GENERATE_ID,
                                              const char *provenance = NULL);
  template <int DIM, typename COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_blockify(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent, Point<DIM, COORD_T> blocking_factor,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);
  IndexPartition create_partition_by_blockify(Context ctx, IndexSpace parent,
                                              DomainPoint blocking_factor,
                                              DomainPoint origin,
                                              Color color = LEGION_AUTO_GENERATE_ID,
                                              const char *provenance = NULL);
  template <int DIM, typename COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_blockify(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent, Point<DIM, COORD_T> blocking_factor,
      Point<DIM, COORD_T> origin, Color color = LEGION_AUTO_GENERATE_ID,
      const char *provenance = NULL);

  IndexPartition create_partition_by_domain(Context ctx, IndexSpace parent,
                                            const std::map<DomainPoint, Domain> &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections = true,
                                            PartitionKind part_kind = LEGION_COMPUTE_KIND,
                                            Color color = LEGION_AUTO_GENERATE_ID,
                                            const char *provenance = NULL);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_domain(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent,
      const std::map<Point<COLOR_DIM, COLOR_COORD_T>, DomainT<DIM, COORD_T>> &domains,
      IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
      bool perform_intersections = true, PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);
  IndexPartition create_partition_by_domain(Context ctx, IndexSpace parent,
                                            const FutureMap &domain_future_map,
                                            IndexSpace color_space,
                                            bool perform_intersections = true,
                                            PartitionKind part_kind = LEGION_COMPUTE_KIND,
                                            Color color = LEGION_AUTO_GENERATE_ID,
                                            const char *provenance = NULL);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_domain(
      Context ctx, IndexSpaceT<DIM, COORD_T> parent, const FutureMap &domain_future_map,
      IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space,
      bool perform_intersections = true, PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, const char *provenance = NULL);

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
  IndexPartition create_partition_by_image_range(
      Context ctx, IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
      FieldID fid, IndexSpace color_space, PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0, MappingTagID tag = 0,
      UntypedBuffer map_arg = UntypedBuffer(), const char *provenance = NULL);

  IndexPartition create_partition_by_preimage(
      Context ctx, IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
      FieldID fid, IndexSpace color_space, PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0, MappingTagID tag = 0,
      UntypedBuffer map_arg = UntypedBuffer(), const char *provenance = NULL);
  IndexPartition create_partition_by_preimage_range(
      Context ctx, IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
      FieldID fid, IndexSpace color_space, PartitionKind part_kind = LEGION_COMPUTE_KIND,
      Color color = LEGION_AUTO_GENERATE_ID, MapperID id = 0, MappingTagID tag = 0,
      UntypedBuffer map_arg = UntypedBuffer(), const char *provenance = NULL);

  IndexPartition create_pending_partition(Context ctx, IndexSpace parent,
                                          IndexSpace color_space,
                                          PartitionKind part_kind = LEGION_COMPUTE_KIND,
                                          Color color = LEGION_AUTO_GENERATE_ID,
                                          const char *provenance = NULL);

  IndexSpace create_index_space_union(Context ctx, IndexPartition parent,
                                      const DomainPoint &color,
                                      const std::vector<IndexSpace> &handles,
                                      const char *provenance = NULL);
  IndexSpace create_index_space_union(Context ctx, IndexPartition parent,
                                      const DomainPoint &color, IndexPartition handle,
                                      const char *provenance = NULL);

  IndexSpace create_index_space_intersection(Context ctx, IndexPartition parent,
                                             const DomainPoint &color,
                                             const std::vector<IndexSpace> &handles,
                                             const char *provenance = NULL);
  IndexSpace create_index_space_intersection(Context ctx, IndexPartition parent,
                                             const DomainPoint &color,
                                             IndexPartition handle,
                                             const char *provenance = NULL);

  IndexSpace create_index_space_difference(Context ctx, IndexPartition parent,
                                           const DomainPoint &color, IndexSpace initial,
                                           const std::vector<IndexSpace> &handles,
                                           const char *provenance = NULL);

  IndexPartition get_index_partition(Context ctx, IndexSpace parent, Color color);
  IndexPartition get_index_partition(Context ctx, IndexSpace parent,
                                     const DomainPoint &color);
  IndexPartition get_index_partition(IndexSpace parent, Color color);
  IndexPartition get_index_partition(IndexSpace parent, const DomainPoint &color);
  template <int DIM, typename COORD_T>
  IndexPartitionT<DIM, COORD_T> get_index_partition(IndexSpaceT<DIM, COORD_T> parent,
                                                    Color color);

  IndexSpace get_index_subspace(Context ctx, IndexPartition p, Color color);
  IndexSpace get_index_subspace(Context ctx, IndexPartition p, const DomainPoint &color);
  IndexSpace get_index_subspace(IndexPartition p, Color color);
  IndexSpace get_index_subspace(IndexPartition p, const DomainPoint &color);

  bool has_index_subspace(Context ctx, IndexPartition p, const DomainPoint &color);
  bool has_index_subspace(IndexPartition p, const DomainPoint &color);

  bool has_multiple_domains(Context ctx, IndexSpace handle);
  bool has_multiple_domains(IndexSpace handle);

  Domain get_index_space_domain(Context ctx, IndexSpace handle);
  Domain get_index_space_domain(IndexSpace handle);

  Domain get_index_partition_color_space(Context ctx, IndexPartition p);
  Domain get_index_partition_color_space(IndexPartition p);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  DomainT<COLOR_DIM, COLOR_COORD_T> get_index_partition_color_space(
      IndexPartitionT<DIM, COORD_T> p);

  IndexSpace get_index_partition_color_space_name(Context ctx, IndexPartition p);
  IndexSpace get_index_partition_color_space_name(IndexPartition p);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  IndexSpaceT<COLOR_DIM, COLOR_COORD_T> get_index_partition_color_space_name(
      IndexPartitionT<DIM, COORD_T> p);

  bool is_index_partition_disjoint(Context ctx, IndexPartition p);
  bool is_index_partition_disjoint(IndexPartition p);
  bool is_index_partition_complete(Context ctx, IndexPartition p);
  bool is_index_partition_complete(IndexPartition p);

  Color get_index_space_color(Context ctx, IndexSpace handle);
  DomainPoint get_index_space_color_point(Context ctx, IndexSpace handle);
  Color get_index_space_color(IndexSpace handle);
  DomainPoint get_index_space_color_point(IndexSpace handle);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  Point<COLOR_DIM, COLOR_COORD_T> get_index_space_color(IndexSpaceT<DIM, COORD_T> handle);

  Color get_index_partition_color(Context ctx, IndexPartition handle);
  DomainPoint get_index_partition_color_point(Context ctx, IndexPartition handle);
  Color get_index_partition_color(IndexPartition handle);
  DomainPoint get_index_partition_color_point(IndexPartition handle);

  IndexSpace get_parent_index_space(Context ctx, IndexPartition handle);
  IndexSpace get_parent_index_space(IndexPartition handle);
  template <int DIM, typename COORD_T>
  IndexSpaceT<DIM, COORD_T> get_parent_index_space(IndexPartitionT<DIM, COORD_T> handle);

  bool has_parent_index_partition(Context ctx, IndexSpace handle);
  bool has_parent_index_partition(IndexSpace handle);

  IndexPartition get_parent_index_partition(Context ctx, IndexSpace handle);
  IndexPartition get_parent_index_partition(IndexSpace handle);
  template <int DIM, typename COORD_T>
  IndexPartitionT<DIM, COORD_T> get_parent_index_partition(
      IndexSpaceT<DIM, COORD_T> handle);

  ptr_t safe_cast(Context ctx, ptr_t pointer, LogicalRegion region);
  DomainPoint safe_cast(Context ctx, DomainPoint point, LogicalRegion region);
  template <int DIM, typename COORD_T>
  bool safe_cast(Context ctx, Point<DIM, COORD_T> point,
                 LogicalRegionT<DIM, COORD_T> region);

  FieldSpace create_field_space(Context ctx, const char *provenance = NULL);
  FieldSpace create_field_space(Context ctx, const std::vector<size_t> &field_sizes,
                                std::vector<FieldID> &resulting_fields,
                                CustomSerdezID serdez_id = 0,
                                const char *provenance = NULL);
  FieldSpace create_field_space(Context ctx, const std::vector<Future> &field_sizes,
                                std::vector<FieldID> &resulting_fields,
                                CustomSerdezID serdez_id = 0,
                                const char *provenance = NULL);

  void create_shared_ownership(Context ctx, FieldSpace handle);

  void destroy_field_space(Context ctx, FieldSpace handle, const bool unordered = false,
                           const char *provenance = NULL);

  size_t get_field_size(Context ctx, FieldSpace handle, FieldID fid);
  size_t get_field_size(FieldSpace handle, FieldID fid);

  void get_field_space_fields(Context ctx, FieldSpace handle,
                              std::vector<FieldID> &fields);
  void get_field_space_fields(FieldSpace handle, std::vector<FieldID> &fields);
  void get_field_space_fields(Context ctx, FieldSpace handle, std::set<FieldID> &fields);
  void get_field_space_fields(FieldSpace handle, std::set<FieldID> &fields);

  LogicalRegion create_logical_region(Context ctx, IndexSpace index, FieldSpace fields,
                                      bool task_local = false,
                                      const char *provenance = NULL);
  template <int DIM, typename COORD_T>
  LogicalRegion create_logical_region(Context ctx, IndexSpaceT<DIM, COORD_T> index,
                                      FieldSpace fields, bool task_local = false,
                                      const char *provenance = NULL);

  void create_shared_ownership(Context ctx, LogicalRegion handle);

  void destroy_logical_region(Context ctx, LogicalRegion handle,
                              const bool unordered = false,
                              const char *provenance = NULL);

  void destroy_logical_partition(Context ctx, LogicalPartition handle,
                                 const bool unordered = false);

  LogicalPartition get_logical_partition(Context ctx, LogicalRegion parent,
                                         IndexPartition handle);
  LogicalPartition get_logical_partition(LogicalRegion parent, IndexPartition handle);

  LogicalPartition get_logical_partition_by_color(Context ctx, LogicalRegion parent,
                                                  Color c);
  LogicalPartition get_logical_partition_by_color(Context ctx, LogicalRegion parent,
                                                  const DomainPoint &c);
  LogicalPartition get_logical_partition_by_color(LogicalRegion parent, Color c);
  LogicalPartition get_logical_partition_by_color(LogicalRegion parent,
                                                  const DomainPoint &c);
  template <int DIM, typename COORD_T>
  LogicalPartitionT<DIM, COORD_T> get_logical_partition_by_color(
      LogicalRegionT<DIM, COORD_T> parent, Color c);

  LogicalPartition get_logical_partition_by_tree(Context ctx, IndexPartition handle,
                                                 FieldSpace fspace, RegionTreeID tid);
  LogicalPartition get_logical_partition_by_tree(IndexPartition handle, FieldSpace fspace,
                                                 RegionTreeID tid);

  LogicalRegion get_logical_subregion(Context ctx, LogicalPartition parent,
                                      IndexSpace handle);
  LogicalRegion get_logical_subregion(LogicalPartition parent, IndexSpace handle);

  LogicalRegion get_logical_subregion_by_color(Context ctx, LogicalPartition parent,
                                               Color c);
  LogicalRegion get_logical_subregion_by_color(Context ctx, LogicalPartition parent,
                                               const DomainPoint &c);
  LogicalRegion get_logical_subregion_by_color(LogicalPartition parent, Color c);
  LogicalRegion get_logical_subregion_by_color(LogicalPartition parent,
                                               const DomainPoint &c);
  template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
  LogicalRegionT<DIM, COORD_T> get_logical_subregion_by_color(
      LogicalPartitionT<DIM, COORD_T> parent, Point<COLOR_DIM, COLOR_COORD_T> color);

  bool has_logical_subregion_by_color(Context ctx, LogicalPartition parent,
                                      const DomainPoint &c);
  bool has_logical_subregion_by_color(LogicalPartition parent, const DomainPoint &c);

  LogicalRegion get_logical_subregion_by_tree(Context ctx, IndexSpace handle,
                                              FieldSpace fspace, RegionTreeID tid);
  LogicalRegion get_logical_subregion_by_tree(IndexSpace handle, FieldSpace fspace,
                                              RegionTreeID tid);

  Color get_logical_region_color(Context ctx, LogicalRegion handle);
  DomainPoint get_logical_region_color_point(Context ctx, LogicalRegion handle);
  Color get_logical_region_color(LogicalRegion handle);
  DomainPoint get_logical_region_color_point(LogicalRegion handle);

  Color get_logical_partition_color(Context ctx, LogicalPartition handle);
  DomainPoint get_logical_partition_color_point(Context ctx, LogicalPartition handle);
  Color get_logical_partition_color(LogicalPartition handle);
  DomainPoint get_logical_partition_color_point(LogicalPartition handle);

  LogicalRegion get_parent_logical_region(Context ctx, LogicalPartition handle);
  LogicalRegion get_parent_logical_region(LogicalPartition handle);

  bool has_parent_logical_partition(Context ctx, LogicalRegion handle);
  bool has_parent_logical_partition(LogicalRegion handle);

  LogicalPartition get_parent_logical_partition(Context ctx, LogicalRegion handle);
  LogicalPartition get_parent_logical_partition(LogicalRegion handle);

  FieldAllocator create_field_allocator(Context ctx, FieldSpace handle);

  Future execute_task(Context ctx, const TaskLauncher &launcher,
                      std::vector<OutputRequirement> *outputs = NULL);

  FutureMap execute_index_space(Context, const IndexTaskLauncher &launcher,
                                std::vector<OutputRequirement> *outputs = NULL);
  Future execute_index_space(Context, const IndexTaskLauncher &launcher,
                             ReductionOpID redop, bool deterministic = false,
                             std::vector<OutputRequirement> *outputs = NULL);

  PhysicalRegion map_region(Context ctx, const InlineLauncher &launcher);

  void unmap_region(Context ctx, PhysicalRegion region);
  void unmap_all_regions(Context ctx);

  template <typename T>
  void fill_field(Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
                  const T &value, Predicate pred = Predicate::TRUE_PRED);
  void fill_field(Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
                  const void *value, size_t value_size,
                  Predicate pred = Predicate::TRUE_PRED);
  void fill_fields(Context ctx, const FillLauncher &launcher);
  void fill_fields(Context ctx, const IndexFillLauncher &launcher);

  PhysicalRegion attach_external_resource(Context ctx, const AttachLauncher &launcher);
  Future detach_external_resource(Context ctx, PhysicalRegion region,
                                  const bool flush = true, const bool unordered = false,
                                  const char *provenance = NULL);

  void issue_copy_operation(Context ctx, const CopyLauncher &launcher);
  void issue_copy_operation(Context ctx, const IndexCopyLauncher &launcher);

  Predicate create_predicate(Context ctx, const Future &f, const char *provenance = NULL);
  Predicate create_predicate(Context ctx, const PredicateLauncher &launcher);
  Predicate predicate_not(Context ctx, const Predicate &p, const char *provenance = NULL);
  Future get_predicate_future(Context ctx, const Predicate &p,
                              const char *provenance = NULL);

  Future issue_mapping_fence(Context ctx, const char *provenance = NULL);
  Future issue_execution_fence(Context ctx, const char *provenance = NULL);

  void begin_trace(Context ctx, TraceID tid, bool logical_only = false,
                   bool static_trace = false,
                   const std::set<RegionTreeID> *managed = NULL,
                   const char *provenance = NULL);
  void end_trace(Context ctx, TraceID tid, const char *provenance = NULL);
  TraceID generate_dynamic_trace_id(void);
  TraceID generate_library_trace_ids(const char *name, size_t count);
  static TraceID generate_static_trace_id(void);

  Future select_tunable_value(Context ctx, TunableID tid, MapperID mapper = 0,
                              MappingTagID tag = 0, const void *args = NULL,
                              size_t argsize = 0);
  Future select_tunable_value(Context ctx, const TunableLauncher &launcher);

  Future get_current_time(Context ctx, Future precondition = Future());
  Future get_current_time_in_microseconds(Context ctx, Future precondition = Future());
  Future get_current_time_in_nanoseconds(Context ctx, Future precondition = Future());
  Future issue_timing_measurement(Context ctx, const TimingLauncher &launcher);

  Processor get_executing_processor(Context ctx);

  void attach_semantic_information(TaskID task_id, SemanticTag tag, const void *buffer,
                                   size_t size, bool is_mutable = false,
                                   bool local_only = false);
  void attach_semantic_information(IndexSpace handle, SemanticTag tag, const void *buffer,
                                   size_t size, bool is_mutable = false);
  void attach_semantic_information(IndexPartition handle, SemanticTag tag,
                                   const void *buffer, size_t size,
                                   bool is_mutable = false);
  void attach_semantic_information(FieldSpace handle, SemanticTag tag, const void *buffer,
                                   size_t size, bool is_mutable = false);
  void attach_semantic_information(FieldSpace handle, FieldID fid, SemanticTag tag,
                                   const void *buffer, size_t size,
                                   bool is_mutable = false);
  void attach_semantic_information(LogicalRegion handle, SemanticTag tag,
                                   const void *buffer, size_t size,
                                   bool is_mutable = false);
  void attach_semantic_information(LogicalPartition handle, SemanticTag tag,
                                   const void *buffer, size_t size,
                                   bool is_mutable = false);

  void attach_name(TaskID task_id, const char *name, bool is_mutable = false,
                   bool local_only = false);
  void attach_name(IndexSpace handle, const char *name, bool is_mutable = false);
  void attach_name(IndexPartition handle, const char *name, bool is_mutable = false);
  void attach_name(FieldSpace handle, const char *name, bool is_mutable = false);
  void attach_name(FieldSpace handle, FieldID fid, const char *name,
                   bool is_mutable = false);
  void attach_name(LogicalRegion handle, const char *name, bool is_mutable = false);
  void attach_name(LogicalPartition handle, const char *name, bool is_mutable = false);

  bool retrieve_semantic_information(TaskID task_id, SemanticTag tag, const void *&result,
                                     size_t &size, bool can_fail = false,
                                     bool wait_until_ready = false);
  bool retrieve_semantic_information(IndexSpace handle, SemanticTag tag,
                                     const void *&result, size_t &size,
                                     bool can_fail = false,
                                     bool wait_until_ready = false);
  bool retrieve_semantic_information(IndexPartition handle, SemanticTag tag,
                                     const void *&result, size_t &size,
                                     bool can_fail = false,
                                     bool wait_until_ready = false);
  bool retrieve_semantic_information(FieldSpace handle, SemanticTag tag,
                                     const void *&result, size_t &size,
                                     bool can_fail = false,
                                     bool wait_until_ready = false);
  bool retrieve_semantic_information(FieldSpace handle, FieldID fid, SemanticTag tag,
                                     const void *&result, size_t &size,
                                     bool can_fail = false,
                                     bool wait_until_ready = false);
  bool retrieve_semantic_information(LogicalRegion handle, SemanticTag tag,
                                     const void *&result, size_t &size,
                                     bool can_fail = false,
                                     bool wait_until_ready = false);
  bool retrieve_semantic_information(LogicalPartition handle, SemanticTag tag,
                                     const void *&result, size_t &size,
                                     bool can_fail = false,
                                     bool wait_until_ready = false);

  void retrieve_name(TaskID task_id, const char *&result);
  void retrieve_name(IndexSpace handle, const char *&result);
  void retrieve_name(IndexPartition handle, const char *&result);
  void retrieve_name(FieldSpace handle, const char *&result);
  void retrieve_name(FieldSpace handle, FieldID fid, const char *&result);
  void retrieve_name(LogicalRegion handle, const char *&result);
  void retrieve_name(LogicalPartition handle, const char *&result);

  void print_once(Context ctx, FILE *f, const char *message);

  Mapping::MapperRuntime *get_mapper_runtime(void);

  MapperID generate_dynamic_mapper_id(void);
  MapperID generate_library_mapper_ids(const char *name, size_t count);
  static MapperID generate_static_mapper_id(void);
  void add_mapper(MapperID map_id, Mapping::Mapper *mapper,
                  Processor proc = Processor::NO_PROC);
  void replace_default_mapper(Mapping::Mapper *mapper,
                              Processor proc = Processor::NO_PROC);

  ProjectionID generate_dynamic_projection_id(void);
  ProjectionID generate_library_projection_ids(const char *name, size_t count);
  static ProjectionID generate_static_projection_id(void);
  void register_projection_functor(ProjectionID pid, ProjectionFunctor *functor,
                                   bool silence_warnings = false,
                                   const char *warning_string = NULL);
  static void preregister_projection_functor(ProjectionID pid,
                                             ProjectionFunctor *functor);

  ShardingID generate_dynamic_sharding_id(void);
  ShardingID generate_library_sharding_ids(const char *name, size_t count);
  static ShardingID generate_static_sharding_id(void);
  void register_sharding_functor(ShardingID sid, ShardingFunctor *functor,
                                 bool silence_warnings = false,
                                 const char *warning_string = NULL);
  static void preregister_sharding_functor(ShardingID sid, ShardingFunctor *functor);
  static ShardingFunctor *get_sharding_functor(ShardingID sid);

  ReductionOpID generate_dynamic_reduction_id(void);
  ReductionOpID generate_library_reduction_ids(const char *name, size_t count);
  static ReductionOpID generate_static_reduction_id(void);
  template <typename REDOP>
  static void register_reduction_op(ReductionOpID redop_id,
                                    bool permit_duplicates = false);
  static void register_reduction_op(ReductionOpID redop_id, ReductionOp *op,
                                    SerdezInitFnptr init_fnptr = NULL,
                                    SerdezFoldFnptr fold_fnptr = NULL,
                                    bool permit_duplicates = false);
  static const ReductionOp *get_reduction_op(ReductionOpID redop_id);

  static int start(int argc, char **argv, bool background = false,
                   bool supply_default_mapper = true);
  static void initialize(int *argc, char ***argv, bool filter = false);
  static int wait_for_shutdown(void);
  static void set_return_code(int return_code);
  static void set_top_level_task_id(TaskID top_id);
  static size_t get_maximum_dimension(void);

  static void add_registration_callback(RegistrationCallbackFnptr callback,
                                        bool dedup = true, size_t dedup_tag = 0);
  static void set_registration_callback(RegistrationCallbackFnptr callback);

  static const InputArgs &get_input_args(void);

  LayoutConstraintID register_layout(const LayoutConstraintRegistrar &registrar);
  void release_layout(LayoutConstraintID layout_id);
  static LayoutConstraintID preregister_layout(
      const LayoutConstraintRegistrar &registrar,
      LayoutConstraintID layout_id = LEGION_AUTO_GENERATE_ID);

  TaskID generate_dynamic_task_id(void);
  TaskID generate_library_task_ids(const char *name, size_t count);
  static TaskID generate_static_task_id(void);
  template <typename T, T (*TASK_PTR)(const Task *, const std::vector<PhysicalRegion> &,
                                      Context, Runtime *)>
  VariantID register_task_variant(const TaskVariantRegistrar &registrar,
                                  VariantID vid = LEGION_AUTO_GENERATE_ID);
  template <typename T, typename UDT,
            T (*TASK_PTR)(const Task *, const std::vector<PhysicalRegion> &, Context,
                          Runtime *, const UDT &)>
  VariantID register_task_variant(const TaskVariantRegistrar &registrar,
                                  const UDT &user_data,
                                  VariantID vid = LEGION_AUTO_GENERATE_ID);
  template <void (*TASK_PTR)(const Task *, const std::vector<PhysicalRegion> &, Context,
                             Runtime *)>
  VariantID register_task_variant(const TaskVariantRegistrar &registrar,
                                  VariantID vid = LEGION_AUTO_GENERATE_ID);
  template <typename UDT,
            void (*TASK_PTR)(const Task *, const std::vector<PhysicalRegion> &, Context,
                             Runtime *, const UDT &)>
  VariantID register_task_variant(const TaskVariantRegistrar &registrar,
                                  const UDT &user_data,
                                  VariantID vid = LEGION_AUTO_GENERATE_ID);
  VariantID register_task_variant(const TaskVariantRegistrar &registrar,
                                  const CodeDescriptor &codedesc,
                                  const void *user_data = NULL, size_t user_len = 0,
                                  size_t return_type_size = LEGION_MAX_RETURN_SIZE,
                                  VariantID vid = LEGION_AUTO_GENERATE_ID,
                                  bool has_return_type_size = true);
  template <typename T, T (*TASK_PTR)(const Task *, const std::vector<PhysicalRegion> &,
                                      Context, Runtime *)>
  static VariantID preregister_task_variant(const TaskVariantRegistrar &registrar,
                                            const char *task_name = NULL,
                                            VariantID vid = LEGION_AUTO_GENERATE_ID);
  template <typename T, typename UDT,
            T (*TASK_PTR)(const Task *, const std::vector<PhysicalRegion> &, Context,
                          Runtime *, const UDT &)>
  static VariantID preregister_task_variant(const TaskVariantRegistrar &registrar,
                                            const UDT &user_data,
                                            const char *task_name = NULL,
                                            VariantID vid = LEGION_AUTO_GENERATE_ID);
  template <void (*TASK_PTR)(const Task *, const std::vector<PhysicalRegion> &, Context,
                             Runtime *)>
  static VariantID preregister_task_variant(const TaskVariantRegistrar &registrar,
                                            const char *task_name = NULL,
                                            VariantID vid = LEGION_AUTO_GENERATE_ID);
  template <typename UDT,
            void (*TASK_PTR)(const Task *, const std::vector<PhysicalRegion> &, Context,
                             Runtime *, const UDT &)>
  static VariantID preregister_task_variant(const TaskVariantRegistrar &registrar,
                                            const UDT &user_data,
                                            const char *task_name = NULL,
                                            VariantID vid = LEGION_AUTO_GENERATE_ID);
  static VariantID preregister_task_variant(
      const TaskVariantRegistrar &registrar, const CodeDescriptor &codedesc,
      const void *user_data = NULL, size_t user_len = 0, const char *task_name = NULL,
      VariantID vid = LEGION_AUTO_GENERATE_ID,
      size_t return_type_size = LEGION_MAX_RETURN_SIZE, bool has_return_type_size = true,
      bool check_task_id = true);

  static void legion_task_preamble(const void *data, size_t datalen, Processor p,
                                   const Task *&task,
                                   const std::vector<PhysicalRegion> *&reg, Context &ctx,
                                   Runtime *&runtime);
  static void legion_task_postamble(
      Runtime *runtime, Context ctx, const void *retvalptr = NULL, size_t retvalsize = 0,
      bool owned = false, Realm::RegionInstance inst = Realm::RegionInstance::NO_INST,
      const void *metadataptr = NULL, size_t metadatasize = 0);

  ShardID get_shard_id(Context ctx, bool I_know_what_I_am_doing = false);
  size_t get_num_shards(Context ctx, bool I_know_what_I_am_doing = false);

public:
  // Checkpointing methods
  void enable_checkpointing(Context ctx);

  void checkpoint(Context ctx, Predicate pred = Predicate::TRUE_PRED);
  void auto_checkpoint(Context ctx, Predicate pred = Predicate::TRUE_PRED);

public:
  // Unsafe methods, please avoid unless you know what you're doing
  void allow_unsafe_inline_mapping(bool I_know_what_I_am_doing = false);
  bool is_replaying_checkpoint(bool I_know_what_I_am_doing = false);

private:
  // Internal methods
  bool skip_api_call();

  bool replay_index_space() const;
  IndexSpace restore_index_space(Context ctx, const char *provenance);
  void restore_index_space_recomputed(IndexSpace is);
  void register_index_space(IndexSpace is);

  bool replay_index_partition() const;
  IndexPartition restore_index_partition(Context ctx, IndexSpace index_space,
                                         IndexSpace color_space, Color color,
                                         const char *provenance);
#ifdef RESILIENCE_CROSS_PRODUCT_BYPASS
  void restore_index_partition_bypass(Context ctx, IndexPartition ip);
#endif
  void register_index_partition(IndexPartition ip);

  bool replay_future() const;
  Future restore_future();
  void register_future(const Future &f);

  bool replay_future_map() const;
  FutureMap restore_future_map(Context ctx);
  void register_future_map(const FutureMap &f);

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
                   const PathSerializer &path, Predicate pred);
  void save_partition(Context ctx, LogicalPartition lp, LogicalRegion parent,
                      LogicalRegion cpy, const std::vector<FieldID> &fids,
                      resilient_tag_t tag, const PathSerializer &path, Predicate pred);
  void save_region_content(Context ctx, LogicalRegion r, Predicate pred);

  static void register_mapper(Machine machine, Legion::Runtime *rt,
                              const std::set<Processor> &local_procs);
  static void fix_projection_functors(Machine machine, Legion::Runtime *rt,
                                      const std::set<Processor> &local_procs);

  template <typename T,
            T (*TASK_PTR)(const Task *task, const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)>
  static T task_wrapper(const Task *task, const std::vector<PhysicalRegion> &regions,
                        Context ctx, Legion::Runtime *runtime);
  template <typename T, typename UDT,
            T (*TASK_PTR)(const Task *task, const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime, const UDT &user_data)>
  static T task_wrapper_user_data(const Task *task,
                                  const std::vector<PhysicalRegion> &regions, Context ctx,
                                  Legion::Runtime *runtime, const UDT &user_data);
  template <void (*TASK_PTR)(const Task *task, const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)>
  static void task_wrapper_void(const Task *task,
                                const std::vector<PhysicalRegion> &regions, Context ctx,
                                Legion::Runtime *runtime);
  template <typename UDT,
            void (*TASK_PTR)(const Task *task, const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime, const UDT &user_data)>
  static void task_wrapper_void_user_data(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context ctx, Legion::Runtime *runtime,
                                          const UDT &user_data);

private:
  Legion::Runtime *lrt;

  bool enabled, replay;
  // FIXME (Elliott): make these all maps

  // Note: we only track the Future once to avoid inflating reference counts
  std::map<resilient_tag_t, Future> futures;
  std::map<Legion::Future, resilient_tag_t> future_tags;
  std::map<Legion::Future, FutureState> future_state;

  // Note: we only track the FutureMap once to avoid inflating reference counts
  std::map<resilient_tag_t, FutureMap> future_maps;
  std::map<Legion::FutureMap, resilient_tag_t> future_map_tags;
  std::map<Legion::FutureMap, FutureMapState> future_map_state;

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

  resilient_tag_t auto_step, auto_checkpoint_step;

  bool allow_inline_mapping;  // unsafe!!!

  CheckpointState state;
  ShardedCheckpointState sharded_state;
  Legion::IndexSpace shard_space;

  // For internal measurements
  Legion::Future replay_start;

private:
  static bool config_disable;
  static std::string config_prefix;
  static bool config_replay;
  static resilient_tag_t config_checkpoint_tag;
  static size_t config_max_instances;
  static long config_auto_steps;
  static bool config_measure_replay_time_and_exit;
  static bool config_skip_leak_check;

  static TaskID write_checkpoint_task_id;
  static TaskID read_checkpoint_task_id;
  static MapperID resilient_mapper_id;

  static std::vector<ProjectionFunctor *> preregistered_projection_functors;

  friend class Future;
  friend class FutureMap;
  friend class FutureMapSerializer;
  friend class IndexSpaceSerializer;
  friend class IndexPartitionSerializer;
  friend class Path;
  friend class ProjectionFunctor;
  friend class RegionTreeStateSerializer;
  friend class Mapping::ResilientMapper;
};

}  // namespace ResilientLegion

#include "resilience/future.inl"
#include "resilience/launcher.inl"
#include "resilience/resilience.inl"
#include "resilience/serializer.inl"

#endif  // RESILIENCE_H
