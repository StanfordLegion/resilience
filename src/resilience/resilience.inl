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

}  // namespace ResilientLegion
