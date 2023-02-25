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

FutureMap FutureMapSerializer::inflate(Runtime *runtime, Context ctx) const {
  IndexSpace is = domain.inflate(runtime, ctx, NULL);
  std::map<DomainPoint, UntypedBuffer> data;
  for (auto &point : map) {
    auto &buffer = point.second.buffer;
    data[point.first] = UntypedBuffer(buffer.data(), buffer.size());
  }

  Domain d = runtime->lrt->get_index_space_domain(is);
  FutureMap fm(runtime, d, runtime->lrt->construct_future_map(ctx, is, data));
  runtime->lrt->destroy_index_space(ctx, is);
  return fm;
}

IndexSpace IndexSpaceSerializer::inflate(Runtime *runtime, Context ctx,
                                         const char *provenance) const {
  std::vector<Domain> rects;
  for (auto &rect : domain.rects) {
    rects.push_back(Domain(rect));
  }

  return runtime->lrt->create_index_space(ctx, rects, provenance);
}

IndexPartitionSerializer::IndexPartitionSerializer(Runtime *runtime, IndexPartition ip,
                                                   Domain color_space_)
    : color_space(color_space_) {
  for (Domain::DomainPointIterator i(color_space_); i; ++i) {
    IndexSpace is = runtime->lrt->get_index_subspace(ip, *i);
    Domain domain = runtime->lrt->get_index_space_domain(is);
    subspaces.emplace(*i, domain);
  }

  // By the time we get here, disjointness/completeness has probably already been
  // computed, so the cost of querying here should be minimal (and we can save some time
  // on restore by caching the result).
  bool disjoint = runtime->lrt->is_index_partition_disjoint(ip);
  bool complete = runtime->lrt->is_index_partition_complete(ip);
  if (disjoint && complete) {
    kind = LEGION_DISJOINT_COMPLETE_KIND;
  } else if (disjoint && !complete) {
    kind = LEGION_DISJOINT_INCOMPLETE_KIND;
  } else if (!disjoint && complete) {
    kind = LEGION_ALIASED_COMPLETE_KIND;
  } else if (!disjoint && !complete) {
    kind = LEGION_ALIASED_INCOMPLETE_KIND;
  } else {
    assert(false);
  }
}

template <int DIM, typename COORD_T, int COLOR_DIM, typename COLOR_COORD_T>
IndexPartition inflate_partition_2(const IndexPartitionSerializer &ser,
                                   Legion::Runtime *lrt, Context ctx, IndexSpace parent,
                                   IndexSpace color_space, Color color,
                                   const char *provenance) {
  std::map<Point<COLOR_DIM, COLOR_COORD_T>, std::vector<Rect<DIM, COORD_T>>> rectangles;

  for (auto &subspace : ser.subspaces) {
    DomainPoint cp(subspace.first);
    Point<COLOR_DIM, COLOR_COORD_T> color(cp);
    for (auto &r : subspace.second.domain.rects) {
      Domain d(r);
      Rect<DIM, COORD_T> rect(d);
      rectangles[color].push_back(rect);
    }
  }

  // FIXME (Elliott): should try out the sharded version for performance
  return lrt->create_partition_by_rectangles<DIM, COORD_T, COLOR_DIM, COLOR_COORD_T>(
      ctx, static_cast<IndexSpaceT<DIM, COORD_T>>(parent), rectangles,
      static_cast<IndexSpaceT<COLOR_DIM, COLOR_COORD_T>>(color_space),
      false /* perform_intersections */, ser.kind, color, provenance,
      false /* collective */);
}

template <int DIM, typename COORD_T>
IndexPartition inflate_partition_1(const IndexPartitionSerializer &ser,
                                   Legion::Runtime *lrt, Context ctx, IndexSpace parent,
                                   IndexSpace color_space, Color color,
                                   const char *provenance) {
  int color_dim = lrt->get_index_space_domain(color_space).get_dim();
  switch (color_dim) {
#define DIMFUNC(COLOR_DIM)                                        \
  case COLOR_DIM: {                                               \
    return inflate_partition_2<DIM, COORD_T, COLOR_DIM, coord_t>( \
        ser, lrt, ctx, parent, color_space, color, provenance);   \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

IndexPartition inflate_partition_0(const IndexPartitionSerializer &ser,
                                   Legion::Runtime *lrt, Context ctx, IndexSpace parent,
                                   IndexSpace color_space, Color color,
                                   const char *provenance) {
  int dim = lrt->get_index_space_domain(parent).get_dim();
  switch (dim) {
#define DIMFUNC(DIM)                                                                    \
  case DIM: {                                                                           \
    return inflate_partition_1<DIM, coord_t>(ser, lrt, ctx, parent, color_space, color, \
                                             provenance);                               \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

IndexPartition IndexPartitionSerializer::inflate(Runtime *runtime, Context ctx,
                                                 IndexSpace parent,
                                                 IndexSpace color_space_, Color color,
                                                 const char *provenance) const {
  // In most (but not all??) cases, an index space is already available for the color
  // space. We therefore do not need to inflate the serialized color space. But we still
  // serialize the color space because we anticipate that a user-provided color space
  // may not always be available.

  // Some older APIs do not provide a color space, so if it doesn't exist, recreate it:
  if (!color_space_.exists()) {
    color_space_ = color_space.inflate(runtime, ctx, provenance);
  } else {
#ifdef DEBUG_LEGION
    // Sanity check color space from user is same as what we serialized.
    assert(color_space.domain.rects.size() == 1);
    Domain rect(color_space.domain.rects.at(0));
    Domain color_space_domain = runtime->lrt->get_index_space_domain(ctx, color_space_);
    assert(rect == color_space_domain);
#endif
  }

  return inflate_partition_0(*this, runtime->lrt, ctx, parent, color_space_, color,
                             provenance);
}

RegionTreeStateSerializer::RegionTreeStateSerializer(Runtime *runtime,
                                                     LogicalRegion parent,
                                                     const RegionTreeState &state) {
  for (auto &p : state.recent_partitions) {
    IndexPartition ip = p.second.get_index_partition();
    resilient_tag_t ip_tag = runtime->ipartition_tags.at(ip);
    auto &ip_state = runtime->state.ipartition_state.at(ip_tag);
    if (ip_state.destroyed) {
      continue;
    }

    Path lr_path = runtime->compute_region_path(p.first, parent);
    Path lp_path = runtime->compute_partition_path(p.second);
    recent_partitions[lr_path] = lp_path;
  }
}

void RegionTreeStateSerializer::inflate(Runtime *runtime, LogicalRegion parent,
                                        RegionTreeState &state) const {
  assert(state.recent_partitions.empty());
  for (auto &p : recent_partitions) {
    LogicalRegion lr = runtime->lookup_region_path(parent, p.first);
    LogicalPartition lp = runtime->lookup_partition_path(parent, p.second);
    assert(lp.get_index_partition() != IndexPartition::NO_PART);
    state.recent_partitions[lr] = lp;
  }
}
