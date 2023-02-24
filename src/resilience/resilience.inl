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

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
IndexPartitionT<DIM, COORD_T> Runtime::create_equal_partition(
    Context ctx, IndexSpaceT<DIM, COORD_T> parent,
    IndexSpaceT<COLOR_DIM, COLOR_COORD_T> color_space, size_t granularity, Color color,
    const char *provenance) {
  if (!enabled) {
    return lrt->create_equal_partition(ctx, parent, color_space, granularity, color,
                                       provenance);
  }

  if (replay && partition_tag < max_partition_tag) {
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, color_space, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_equal_partition(
      ctx, parent, color_space, granularity, color, provenance);
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

  if (replay && partition_tag < max_partition_tag) {
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

  if (replay && partition_tag < max_partition_tag) {
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
                                                 provenance);
  }

  if (replay && partition_tag < max_partition_tag) {
    IndexSpace color_space = lrt->get_index_partition_color_space_name(partition);
    return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, parent, color_space, color, provenance));
  }

  IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_intersection(
      ctx, parent, partition, part_kind, color, provenance);
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

  if (replay && partition_tag < max_partition_tag) {
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

  if (replay && partition_tag < max_partition_tag) {
    IndexSpace color_space = lrt->get_index_partition_color_space_name(handle2);
    Domain domain = lrt->get_index_partition_color_space(handle1);
    for (Domain::DomainPointIterator i(domain); i; ++i) {
      IndexSpace subspace = lrt->get_index_subspace(handle1, *i);
      IndexPartition sub_ip =
          restore_index_partition(ctx, subspace, color_space, color, provenance);
      color = lrt->get_index_partition_color(sub_ip);
      auto it = handles.find(subspace);
      if (it != handles.end()) {
        it->second = sub_ip;
      }
    }
    assert(partition_tag <= max_partition_tag);
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

}  // namespace ResilientLegion
