// -*- mode: c++ -*-
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

namespace ResilientLegion {

template <int DIM, typename COORD_T>
IndexSpaceT<DIM, COORD_T> Runtime::create_index_space(Context ctx,
                                                      const Rect<DIM, COORD_T> &bounds,
                                                      const char *provenance) {
  if (!enabled) {
    return lrt->create_index_space(ctx, bounds, provenance);
  }

  if (replay_index_space()) {
    return static_cast<IndexSpaceT<DIM, COORD_T>>(restore_index_space(ctx, provenance));
  }

  IndexSpace is = lrt->create_index_space(ctx, bounds, provenance);
  register_index_space(is);
  return static_cast<IndexSpaceT<DIM, COORD_T>>(is);
}

template <int DIM, typename COORD_T>
IndexSpaceT<DIM, COORD_T> Runtime::create_index_space(Context ctx, const Future &f,
                                                      const char *provenance) {
  if (!enabled) {
    return lrt->create_index_space<DIM, COORD_T>(ctx, f, provenance);
  }

  if (replay_index_space()) {
    return static_cast<IndexSpaceT<DIM, COORD_T>>(restore_index_space(ctx, provenance));
  }

  IndexSpace is = lrt->create_index_space<DIM, COORD_T>(ctx, f, provenance);
  register_index_space(is);
  return static_cast<IndexSpaceT<DIM, COORD_T>>(is);
}

template <int DIM, typename COORD_T>
IndexSpaceT<DIM, COORD_T> Runtime::create_index_space(
    Context ctx, const std::vector<Point<DIM, COORD_T>> &points, const char *provenance) {
  if (!enabled) {
    return lrt->create_index_space(ctx, points, provenance);
  }

  if (replay_index_space()) {
    return static_cast<IndexSpaceT<DIM, COORD_T>>(restore_index_space(ctx, provenance));
  }

  IndexSpace is = lrt->create_index_space(ctx, points, provenance);
  register_index_space(is);
  return static_cast<IndexSpaceT<DIM, COORD_T>>(is);
}

template <int DIM, typename COORD_T>
IndexSpaceT<DIM, COORD_T> Runtime::union_index_spaces(
    Context ctx, const std::vector<IndexSpaceT<DIM, COORD_T>> &spaces,
    const char *provenance) {
  if (!enabled) {
    return lrt->union_index_spaces(ctx, spaces, provenance);
  }

  if (replay_index_space()) {
    return static_cast<IndexSpaceT<DIM, COORD_T>>(restore_index_space(ctx, provenance));
  }

  IndexSpace is = lrt->union_index_spaces(ctx, spaces, provenance);
  register_index_space(is);
  return is;
}

template <int DIM, typename COORD_T>
IndexSpaceT<DIM, COORD_T> Runtime::intersect_index_spaces(
    Context ctx, const std::vector<IndexSpaceT<DIM, COORD_T>> &spaces,
    const char *provenance) {
  if (!enabled) {
    return lrt->intersect_index_spaces(ctx, spaces, provenance);
  }

  if (replay_index_space()) {
    return static_cast<IndexSpaceT<DIM, COORD_T>>(restore_index_space(ctx, provenance));
  }

  IndexSpace is = lrt->intersect_index_spaces(ctx, spaces, provenance);
  register_index_space(is);
  return is;
}

template <int DIM, typename COORD_T>
IndexSpaceT<DIM, COORD_T> Runtime::subtract_index_spaces(Context ctx,
                                                         IndexSpaceT<DIM, COORD_T> left,
                                                         IndexSpaceT<DIM, COORD_T> right,
                                                         const char *provenance) {
  if (!enabled) {
    return lrt->subtract_index_spaces(ctx, left, right, provenance);
  }

  if (replay_index_space()) {
    return static_cast<IndexSpaceT<DIM, COORD_T>>(restore_index_space(ctx, provenance));
  }

  IndexSpace is = lrt->subtract_index_spaces(ctx, left, right, provenance);
  register_index_space(is);
  return is;
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_equal_partition(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent,
    IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, size_t granularity, Color color,
    const char *provenance) {
  if (!enabled) {
    return lrt->create_equal_partition(ctx, parent, color_space, granularity, color,
                                       provenance);
  }

  if (replay_index_partition()) {
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, color_space, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_equal_partition(
      ctx, parent, color_space, granularity, color, provenance);
  register_index_partition(ip);
  return ip;
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_partition_by_weights(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent,
    const std::map<Point<COLOR_DIM, COLOR_COORD_T>, int> &weights,
    IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, size_t granularity, Color color,
    const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_weights(ctx, parent, weights, color_space,
                                            granularity, color, provenance);
  }

  if (replay_index_partition()) {
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, color_space, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_weights(
      ctx, parent, weights, color_space, granularity, color, provenance);
  register_index_partition(ip);
  return ip;
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_partition_by_weights(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent,
    const std::map<Point<COLOR_DIM, COLOR_COORD_T>, size_t> &weights,
    IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, size_t granularity, Color color,
    const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_weights(ctx, parent, weights, color_space,
                                            granularity, color, provenance);
  }

  if (replay_index_partition()) {
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, color_space, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_weights(
      ctx, parent, weights, color_space, granularity, color, provenance);
  register_index_partition(ip);
  return ip;
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_partition_by_weights(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent, const FutureMap &weights,
    IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, size_t granularity, Color color,
    const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_weights(ctx, parent, weights, color_space,
                                            granularity, color, provenance);
  }

  if (replay_index_partition()) {
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, color_space, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_weights(
      ctx, parent, weights, color_space, granularity, color, provenance);
  register_index_partition(ip);
  return ip;
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_partition_by_union(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent, IndexPartitionT<DIM, COORD_T> handle1,
    IndexPartitionT<DIM, COORD_T> handle2,
    IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, PartitionKind part_kind,
    Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_union(ctx, parent, handle1, handle2, color_space,
                                          part_kind, color, provenance);
  }

  if (replay_index_partition()) {
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, color_space, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_union(
      ctx, parent, handle1, handle2, color_space, part_kind, color, provenance);
  register_index_partition(ip);
  return ip;
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_partition_by_intersection(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent, IndexPartitionT<DIM, COORD_T> handle1,
    IndexPartitionT<DIM, COORD_T> handle2,
    IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, PartitionKind part_kind,
    Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_intersection(
        ctx, parent, handle1, handle2, color_space, part_kind, color, provenance);
  }

  if (replay_index_partition()) {
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, color_space, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_intersection(
      ctx, parent, handle1, handle2, color_space, part_kind, color, provenance);
  register_index_partition(ip);
  return ip;
}

template <int DIM, typename COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_partition_by_intersection(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent,
    IndexPartitionT<DIM, COORD_T> partition, PartitionKind part_kind, Color color,
    bool dominates, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_intersection(ctx, parent, partition, part_kind, color,
                                                 dominates, provenance);
  }

  if (replay_index_partition()) {
    IndexSpace color_space = lrt->get_index_partition_color_space_name(partition);
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, color_space, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_intersection(
      ctx, parent, partition, part_kind, color, dominates, provenance);
  register_index_partition(ip);
  return ip;
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_partition_by_difference(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent, IndexPartitionT<DIM, COORD_T> handle1,
    IndexPartitionT<DIM, COORD_T> handle2,
    IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, PartitionKind part_kind,
    Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_difference(ctx, parent, handle1, handle2, color_space,
                                               part_kind, color, provenance);
  }

  if (replay_index_partition()) {
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, color_space, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_difference(
      ctx, parent, handle1, handle2, color_space, part_kind, color, provenance);
  register_index_partition(ip);
  return ip;
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
Color Runtime::create_cross_product_partitions(
    Context ctx, IndexPartitionT<DIM, COORD_T> handle1,
    IndexPartitionT<DIM, COORD_T> handle2,
    typename std::map<IndexSpaceT<DIM, COORD_T>, IndexPartitionT<DIM, COORD_T>> &handles,
    PartitionKind part_kind, Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_cross_product_partitions(ctx, handle1, handle2, handles, part_kind,
                                                color, provenance);
  }

  if (replay_index_partition()) {
    IndexSpace color_space = lrt->get_index_partition_color_space_name(handle2);
    Domain domain = lrt->get_index_partition_color_space(handle1);
    for (Domain::DomainPointIterator i(domain); i; ++i) {
      assert(replay_index_partition());
      IndexSpace subspace = lrt->get_index_subspace(handle1, *i);
      IndexPartition sub_ip =
          restore_index_partition(ctx, subspace, color_space, color, provenance);
      color = lrt->get_index_partition_color(sub_ip);
      auto it = handles.find(subspace);
      if (it != handles.end()) {
        it->second = sub_ip;
      }
    }
    return color;
  }

  Color result = lrt->create_cross_product_partitions(ctx, handle1, handle2, handles,
                                                      part_kind, color, provenance);
  Domain domain = lrt->get_index_partition_color_space(handle1);
  for (Domain::DomainPointIterator i(domain); i; ++i) {
    IndexSpace subspace = lrt->get_index_subspace(handle1, *i);
    IndexPartition sub_ip = lrt->get_index_partition(subspace, result);
    register_index_partition(sub_ip);
  }
  return result;
}

template <int DIM, int COLOR_DIM, typename COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_partition_by_restriction(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent,
    IndexSpaceT<COLOR_DIM, COORD_T> color_space,
    Transform<DIM, COLOR_DIM, COORD_T> transform, Rect<DIM, COORD_T> extent,
    PartitionKind part_kind, Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_restriction(ctx, parent, color_space, transform,
                                                extent, part_kind, color, provenance);
  }

  if (replay && partition_tag < state.max_partition_tag) {
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, color_space, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_restriction(
      ctx, parent, color_space, transform, extent, part_kind, color, provenance);
  register_index_partition(ip);
  return ip;
}

template <int DIM, typename COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_partition_by_blockify(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent, Point<DIM, COORD_T> blocking_factor,
    Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_blockify(ctx, parent, blocking_factor, color,
                                             provenance);
  }

  if (replay && partition_tag < state.max_partition_tag) {
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip =
      lrt->create_partition_by_blockify(ctx, parent, blocking_factor, color, provenance);
  register_index_partition(ip);
  return ip;
}

template <int DIM, typename COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_partition_by_blockify(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent, Point<DIM, COORD_T> blocking_factor,
    Point<DIM, COORD_T> origin, Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_blockify(ctx, parent, blocking_factor, origin, color,
                                             provenance);
  }

  if (replay && partition_tag < state.max_partition_tag) {
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_blockify(
      ctx, parent, blocking_factor, origin, color, provenance);
  register_index_partition(ip);
  return ip;
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_partition_by_domain(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent,
    const std::map<Point<COLOR_DIM, COLOR_COORD_T>, DomainT<DIM, COORD_T>> &domains,
    IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, bool perform_intersections,
    PartitionKind part_kind, Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_domain(ctx, parent, domains, color_space,
                                           perform_intersections, part_kind, color,
                                           provenance);
  }

  if (replay_index_partition()) {
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_domain(
      ctx, parent, domains, color_space, perform_intersections, part_kind, color,
      provenance);
  register_index_partition(ip);
  return ip;
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_partition_by_domain(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent, const FutureMap &domain_future_map,
    IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, bool perform_intersections,
    PartitionKind part_kind, Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_domain(ctx, parent, domain_future_map, color_space,
                                           perform_intersections, part_kind, color,
                                           provenance);
  }

  if (replay_index_partition()) {
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_domain(
      ctx, parent, domain_future_map, color_space, perform_intersections, part_kind,
      color, provenance);
  register_index_partition(ip);
  return ip;
}

template <int DIM, typename COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::get_index_partition(
    IndexSpaceT<DIM, COORD_T> parent, Color color) {
  return lrt->get_index_partition(parent, color);
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
DomainT<COLOR_DIM, COLOR_COORD_T> Runtime::get_index_partition_color_space(
    IndexPartitionT<DIM, COORD_T> p) {
  return lrt->get_index_partition_color_space<DIM, COORD_T, COLOR_DIM, COLOR_COORD_T>(p);
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
IndexSpaceT<COLOR_DIM, COLOR_COORD_T> Runtime::get_index_partition_color_space_name(
    IndexPartitionT<DIM, COORD_T> p) {
  return lrt
      ->get_index_partition_color_space_name<DIM, COORD_T, COLOR_DIM, COLOR_COORD_T>(p);
}

template <int DIM, typename COORD_T>
IndexSpaceT<DIM, COORD_T> Runtime::get_parent_index_space(
    IndexPartitionT<DIM, COORD_T> handle) {
  return lrt->get_parent_index_space(handle);
}

template <int DIM, typename COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::get_parent_index_partition(
    IndexSpaceT<DIM, COORD_T> handle) {
  return lrt->get_parent_index_partition(handle);
}

template <int DIM, typename COORD_T>
bool Runtime::safe_cast(Context ctx, Point<DIM, COORD_T> point,
                        LogicalRegionT<DIM, COORD_T> region) {
  return lrt->safe_cast(ctx, point, region);
}

template <int DIM, typename COORD_T>
LogicalRegion Runtime::create_logical_region(Context ctx, IndexSpaceT<DIM, COORD_T> index,
                                             FieldSpace fields, bool task_local,
                                             const char *provenance) {
  return create_logical_region(ctx, static_cast<IndexSpace>(index), fields, task_local,
                               provenance);
}

template <int DIM, typename COORD_T>
LogicalPartitionT<DIM, COORD_T> Runtime::get_logical_partition_by_color(
    LogicalRegionT<DIM, COORD_T> parent, Color c) {
  return lrt->get_logical_partition_by_color(parent, c);
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
LogicalRegionT<DIM, COORD_T> Runtime::get_logical_subregion_by_color(
    LogicalPartitionT<DIM, COORD_T> parent, Point<COLOR_DIM, COLOR_COORD_T> color) {
  return lrt->get_logical_subregion_by_color(parent, color);
}

}  // namespace ResilientLegion
