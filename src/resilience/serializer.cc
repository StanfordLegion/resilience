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
  FutureMap result(d, runtime->lrt->construct_future_map(ctx, is, data));
  runtime->lrt->destroy_index_space(ctx, is);
  return result;
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
  color = runtime->lrt->get_index_partition_color_point(ip);
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

IndexPartition IndexPartitionSerializer::inflate(Runtime *runtime, Context ctx,
                                                 IndexSpace index_space,
                                                 IndexSpace color_space,
                                                 const char *provenance) const {
  MultiDomainPointColoring coloring;
  for (auto &subspace : subspaces) {
    DomainPoint color(subspace.first);
    // Don't inflate the index space: just grab the rects and add them to the coloring.
    for (auto &rect : subspace.second.domain.rects) {
      coloring[color].insert(Domain(rect));
    }
  }

  // In most (but not all??) cases, an index space is already available for the color
  // space. We therefore do not need to inflate the serialized color space. But we still
  // serialize the color space because we anticipate that a user-provided color space
  // may not always be available.

  Domain color_domain = runtime->lrt->get_index_space_domain(ctx, color_space);
  return runtime->lrt->create_index_partition(ctx, index_space, color_domain, coloring,
                                              kind, Point<1>(DomainPoint(color)));
}
