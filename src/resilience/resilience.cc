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

#include <sys/resource.h>
#include <sys/time.h>

#include <algorithm>

#define FILE_AND_LINE (__FILE__ ":" LEGION_MACRO_TO_STRING(__LINE__))

using namespace ResilientLegion;

static Logger log_resilience("resilience");

bool Runtime::config_disable(false);
std::string Runtime::config_prefix;
bool Runtime::config_replay(false);
resilient_tag_t Runtime::config_checkpoint_tag(SIZE_MAX);
size_t Runtime::config_max_instances(3);
long Runtime::config_auto_steps(-1);
bool Runtime::config_measure_replay_time_and_exit(false);
bool Runtime::config_skip_leak_check(false);

TaskID Runtime::write_checkpoint_task_id;
TaskID Runtime::read_checkpoint_task_id;
MapperID Runtime::resilient_mapper_id;

std::vector<ProjectionFunctor *> Runtime::preregistered_projection_functors;

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
      max_checkpoint_tag(0),
      auto_step(0),
      auto_checkpoint_step(0),
      allow_inline_mapping(false),
      shard_space(Legion::IndexSpace::NO_SPACE) {}

Runtime::~Runtime() {
  // Hack: work around https://github.com/StanfordLegion/legion/issues/1415 shutdown crash
  // by issuing one last fence to make sure all the data gets out to disk
  if (enabled) {
    Legion::Future f = lrt->issue_execution_fence(Legion::Runtime::get_context());
    f.get_void_result();
  }

  // Clear all remaining futures

  // Need to clear back-to-front to preserse reference structure
  for (auto rit = futures.rbegin(); rit != futures.rend();) {
    // At this point, all remaining futures should be ones that escaped.
    if (!config_skip_leak_check) {
      auto &state = future_state.at(rit->second);
      assert(state.escaped && state.ref_count == 1);
    }
    // All I wanted for Christmas was to walk a map backwards while erasing elements, but
    // what I got was this?!
    rit = decltype(rit)(futures.erase(std::next(rit).base()));
  }

  // At this point, all remaining future maps should be ones that escaped.
  if (!config_skip_leak_check) {
    for (auto &fm : future_map_state) {
      assert(fm.second.escaped && fm.second.ref_count == 1);
    }
  }
  future_maps.clear();

  if (enabled) {
    lrt->destroy_index_space(Legion::Runtime::get_context(), shard_space, false, false,
                             FILE_AND_LINE);
  }
}

bool Runtime::skip_api_call() {
  bool skip = replay && api_tag < max_api_tag;
  api_tag++;
  return skip;
}

bool Runtime::replay_index_space() const {
  return replay && index_space_tag < max_index_space_tag;
}

IndexSpace Runtime::restore_index_space(Context ctx, const char *provenance) {
  IndexSpaceSerializer ris = state.ispaces.at(index_space_tag);
  IndexSpace is = ris.inflate(this, ctx, provenance);
  assert(ispaces.size() == index_space_tag);
  ispaces.push_back(is);
  index_space_tag++;
  return is;
}

// For the case where the index space is already recomputed, so no reason to bother
// restoring it
void Runtime::restore_index_space_recomputed(IndexSpace is) {
  assert(ispaces.size() == index_space_tag);
  ispaces.push_back(is);
  index_space_tag++;
}

void Runtime::register_index_space(IndexSpace is) {
  assert(ispaces.size() == index_space_tag);
  ispaces.push_back(is);
  // FIXME (Elliott): we do this eagerly because index spaces can be destroyed prior to
  // the checkpoint
  assert(state.ispaces.size() == index_space_tag);
  state.ispaces.emplace_back(lrt->get_index_space_domain(is));
  index_space_tag++;
}

bool Runtime::replay_index_partition() const {
  return replay && partition_tag < max_partition_tag;
}

IndexPartition Runtime::restore_index_partition(Context ctx, IndexSpace index_space,
                                                IndexSpace color_space, Color color,
                                                const char *provenance) {
  if (state.ipartition_state.at(partition_tag).destroyed) {
    IndexPartition ip = IndexPartition::NO_PART;
    ipartitions.push_back(ip);
    partition_tag++;
    return ip;
  }

  IndexPartitionSerializer rip = sharded_state.ipartitions.at(partition_tag);
  IndexPartition ip = rip.inflate(this, ctx, index_space, color_space, color, provenance);
  ipartitions.push_back(ip);
  ipartition_tags[ip] = partition_tag;
  partition_tag++;
  return ip;
}

#ifdef RESILIENCE_CROSS_PRODUCT_BYPASS
// Hacky version for restoring a partition we already recomputed
void Runtime::restore_index_partition_bypass(Context ctx, IndexPartition ip) {
  if (state.ipartition_state.at(partition_tag).destroyed) {
    if (ip != IndexPartition::NO_PART) {
      lrt->destroy_index_partition(ctx, ip, false, true, FILE_AND_LINE);
    }
    ipartitions.push_back(IndexPartition::NO_PART);
    partition_tag++;
  }

  ipartitions.push_back(ip);
  ipartition_tags[ip] = partition_tag;
  partition_tag++;
}
#endif

void Runtime::register_index_partition(IndexPartition ip) {
  ipartitions.push_back(ip);
  ipartition_tags[ip] = partition_tag;
  state.ipartition_state.emplace_back();
  partition_tag++;
}

bool Runtime::replay_future() const { return replay && future_tag < max_future_tag; }

Future Runtime::restore_future() {
  Future f;
  // This is tricky. We want to deserialize the future AT OR BEFORE the current future
  // tag. This is either ourselves, or a future with the same value, OR A COMPLETELY
  // UNRELATED FUTURE. But if the future is unrelated, we know we won't ever use it; the
  // value is dead.
  auto it = state.futures.upper_bound(future_tag);
  if (it != state.futures.begin()) {
    --it;

    auto fit = futures.find(it->first);
    if (fit != futures.end()) {
      future_tag++;
      return fit->second;
    }

    f = it->second.inflate(this);
    futures[future_tag] = f;
    future_tags[f] = future_tag;
  }

  future_tag++;
  return f;
}

void Runtime::register_future(const Future &f) {
  futures[future_tag] = f;
  future_tags[f] = future_tag;
  future_tag++;
}

bool Runtime::replay_future_map() const {
  return replay && future_map_tag < max_future_map_tag;
}

FutureMap Runtime::restore_future_map(Context ctx) {
  FutureMap fm;
  auto it = sharded_state.future_maps.find(future_map_tag);
  if (it != sharded_state.future_maps.end()) {
    fm = it->second.inflate(this, ctx);
    future_maps[future_map_tag] = fm;
    future_map_tags[fm] = future_map_tag;
  }

  future_map_tag++;
  return fm;
}

void Runtime::register_future_map(const FutureMap &fm) {
  future_maps[future_map_tag] = fm;
  future_map_tags[fm] = future_map_tag;
  future_map_tag++;
}

IndexSpace Runtime::create_index_space(Context ctx, const Domain &bounds,
                                       TypeTag type_tag, const char *provenance) {
  if (!enabled) {
    return lrt->create_index_space(ctx, bounds, type_tag, provenance);
  }

  if (replay_index_space()) {
    return restore_index_space(ctx, provenance);
  }

  IndexSpace is = lrt->create_index_space(ctx, bounds, type_tag, provenance);
  register_index_space(is);
  return is;
}

IndexSpace Runtime::create_index_space(Context ctx, size_t dimensions, const Future &f,
                                       TypeTag type_tag, const char *provenance) {
  if (!enabled) {
    return lrt->create_index_space(ctx, dimensions, f, type_tag, provenance);
  }

  if (replay_index_space()) {
    return restore_index_space(ctx, provenance);
  }

  IndexSpace is = lrt->create_index_space(ctx, dimensions, f, type_tag, provenance);
  register_index_space(is);
  return is;
}

IndexSpace Runtime::create_index_space(Context ctx,
                                       const std::vector<DomainPoint> &points,
                                       const char *provenance) {
  if (!enabled) {
    return lrt->create_index_space(ctx, points, provenance);
  }

  if (replay_index_space()) {
    return restore_index_space(ctx, provenance);
  }

  IndexSpace is = lrt->create_index_space(ctx, points, provenance);
  register_index_space(is);
  return is;
}

IndexSpace Runtime::union_index_spaces(Context ctx, const std::vector<IndexSpace> &spaces,
                                       const char *provenance) {
  if (!enabled) {
    return lrt->union_index_spaces(ctx, spaces, provenance);
  }

  if (replay_index_space()) {
    return restore_index_space(ctx, provenance);
  }

  IndexSpace is = lrt->union_index_spaces(ctx, spaces, provenance);
  register_index_space(is);
  return is;
}

IndexSpace Runtime::intersect_index_spaces(Context ctx,
                                           const std::vector<IndexSpace> &spaces,
                                           const char *provenance) {
  if (!enabled) {
    return lrt->intersect_index_spaces(ctx, spaces, provenance);
  }

  if (replay_index_space()) {
    return restore_index_space(ctx, provenance);
  }

  IndexSpace is = lrt->intersect_index_spaces(ctx, spaces, provenance);
  register_index_space(is);
  return is;
}

IndexSpace Runtime::subtract_index_spaces(Context ctx, IndexSpace left, IndexSpace right,
                                          const char *provenance) {
  if (!enabled) {
    return lrt->subtract_index_spaces(ctx, left, right, provenance);
  }

  if (replay_index_space()) {
    return restore_index_space(ctx, provenance);
  }

  IndexSpace is = lrt->subtract_index_spaces(ctx, left, right, provenance);
  register_index_space(is);
  return is;
}

IndexSpace Runtime::create_index_space(Context ctx, size_t max_num_elmts) {
  if (!enabled) {
    return lrt->create_index_space(ctx, max_num_elmts);
  }

  if (replay_index_space()) {
    return restore_index_space(ctx, NULL);
  }

  IndexSpace is = lrt->create_index_space(ctx, max_num_elmts);
  register_index_space(is);
  return is;
}

void Runtime::create_shared_ownership(Context ctx, IndexSpace handle) {
  lrt->create_shared_ownership(ctx, handle);
}

void Runtime::destroy_index_space(Context ctx, IndexSpace handle, const bool unordered,
                                  const bool recurse, const char *provenance) {
  lrt->destroy_index_space(ctx, handle, unordered, recurse, provenance);
}

IndexPartition Runtime::create_index_partition(Context ctx, IndexSpace parent,
                                               const Domain &color_space,
                                               const PointColoring &coloring,
                                               PartitionKind part_kind, Color color,
                                               bool allocable) {
  if (!enabled) {
    return lrt->create_index_partition(ctx, parent, color_space, coloring, part_kind,
                                       color, allocable);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, NULL);
  }

  IndexPartition ip = lrt->create_index_partition(ctx, parent, color_space, coloring,
                                                  part_kind, color, allocable);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_index_partition(Context ctx, IndexSpace parent,
                                               const Coloring &coloring, bool disjoint,
                                               Color color) {
  if (!enabled) {
    return lrt->create_index_partition(ctx, parent, coloring, disjoint, color);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, NULL);
  }

  IndexPartition ip = lrt->create_index_partition(ctx, parent, coloring, disjoint, color);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_index_partition(Context ctx, IndexSpace parent,
                                               const Domain &color_space,
                                               const DomainPointColoring &coloring,
                                               PartitionKind part_kind, Color color) {
  if (!enabled) {
    return lrt->create_index_partition(ctx, parent, color_space, coloring, part_kind,
                                       color);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, NULL);
  }

  IndexPartition ip =
      lrt->create_index_partition(ctx, parent, color_space, coloring, part_kind, color);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_index_partition(Context ctx, IndexSpace parent,
                                               Domain color_space,
                                               const DomainColoring &coloring,
                                               bool disjoint, Color color) {
  if (!enabled) {
    return lrt->create_index_partition(ctx, parent, color_space, coloring, disjoint,
                                       color);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, NULL);
  }

  IndexPartition ip =
      lrt->create_index_partition(ctx, parent, color_space, coloring, disjoint, color);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_index_partition(Context ctx, IndexSpace parent,
                                               const Domain &color_space,
                                               const MultiDomainPointColoring &coloring,
                                               PartitionKind part_kind, Color color) {
  if (!enabled) {
    return lrt->create_index_partition(ctx, parent, color_space, coloring, part_kind,
                                       color);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, NULL);
  }

  IndexPartition ip =
      lrt->create_index_partition(ctx, parent, color_space, coloring, part_kind, color);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_index_partition(Context ctx, IndexSpace parent,
                                               Domain color_space,
                                               const MultiDomainColoring &coloring,
                                               bool disjoint, Color color) {
  if (!enabled) {
    return lrt->create_index_partition(ctx, parent, color_space, coloring, disjoint,
                                       color);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, NULL);
  }

  IndexPartition ip =
      lrt->create_index_partition(ctx, parent, color_space, coloring, disjoint, color);
  register_index_partition(ip);
  return ip;
}

void Runtime::create_shared_ownership(Context ctx, IndexPartition handle) {
  assert(false && "unimplemented");
  abort();
}

void Runtime::destroy_index_partition(Context ctx, IndexPartition handle,
                                      const bool unordered, const bool recurse,
                                      const char *provenance) {
  if (!enabled) {
    lrt->destroy_index_partition(ctx, handle, unordered, recurse, provenance);
    return;
  }

  if (skip_api_call()) return;

  resilient_tag_t ip_tag = ipartition_tags.at(handle);
  IndexPartitionState &ip_state = state.ipartition_state.at(ip_tag);
  ip_state.destroyed = true;
  lrt->destroy_index_partition(ctx, handle, unordered, recurse, provenance);
}

IndexPartition Runtime::create_equal_partition(Context ctx, IndexSpace parent,
                                               IndexSpace color_space, size_t granularity,
                                               Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_equal_partition(ctx, parent, color_space, granularity, color,
                                       provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, color_space, color, provenance);
  }

  IndexPartition ip = lrt->create_equal_partition(ctx, parent, color_space, granularity,
                                                  color, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_weights(
    Context ctx, IndexSpace parent, const std::map<DomainPoint, int> &weights,
    IndexSpace color_space, size_t granularity, Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_weights(ctx, parent, weights, color_space,
                                            granularity, color, provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, color_space, color, provenance);
  }

  IndexPartition ip = lrt->create_partition_by_weights(ctx, parent, weights, color_space,
                                                       granularity, color, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_weights(
    Context ctx, IndexSpace parent, const std::map<DomainPoint, size_t> &weights,
    IndexSpace color_space, size_t granularity, Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_weights(ctx, parent, weights, color_space,
                                            granularity, color, provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, color_space, color, provenance);
  }

  IndexPartition ip = lrt->create_partition_by_weights(ctx, parent, weights, color_space,
                                                       granularity, color, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_weights(Context ctx, IndexSpace parent,
                                                    const FutureMap &weights,
                                                    IndexSpace color_space,
                                                    size_t granularity, Color color,
                                                    const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_weights(ctx, parent, weights, color_space,
                                            granularity, color, provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, color_space, color, provenance);
  }

  IndexPartition ip = lrt->create_partition_by_weights(ctx, parent, weights, color_space,
                                                       granularity, color, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_union(Context ctx, IndexSpace parent,
                                                  IndexPartition handle1,
                                                  IndexPartition handle2,
                                                  IndexSpace color_space,
                                                  PartitionKind part_kind, Color color,
                                                  const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_union(ctx, parent, handle1, handle2, color_space,
                                          part_kind, color, provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, color_space, color, provenance);
  }

  IndexPartition ip = lrt->create_partition_by_union(
      ctx, parent, handle1, handle2, color_space, part_kind, color, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_intersection(
    Context ctx, IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
    IndexSpace color_space, PartitionKind part_kind, Color color,
    const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_intersection(
        ctx, parent, handle1, handle2, color_space, part_kind, color, provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, color_space, color, provenance);
  }

  IndexPartition ip = lrt->create_partition_by_intersection(
      ctx, parent, handle1, handle2, color_space, part_kind, color, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_intersection(Context ctx, IndexSpace parent,
                                                         IndexPartition partition,
                                                         PartitionKind part_kind,
                                                         Color color, bool dominates,
                                                         const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_intersection(ctx, parent, partition, part_kind, color,
                                                 dominates, provenance);
  }

  if (replay_index_partition()) {
    IndexSpace color_space = lrt->get_index_partition_color_space_name(partition);
    return restore_index_partition(ctx, parent, color_space, color, provenance);
  }

  IndexPartition ip = lrt->create_partition_by_intersection(
      ctx, parent, partition, part_kind, color, dominates, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_difference(
    Context ctx, IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
    IndexSpace color_space, PartitionKind part_kind, Color color,
    const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_difference(ctx, parent, handle1, handle2, color_space,
                                               part_kind, color, provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, color_space, color, provenance);
  }

  IndexPartition ip = lrt->create_partition_by_difference(
      ctx, parent, handle1, handle2, color_space, part_kind, color, provenance);
  register_index_partition(ip);
  return ip;
}

Color Runtime::create_cross_product_partitions(
    Context ctx, IndexPartition handle1, IndexPartition handle2,
    std::map<IndexSpace, IndexPartition> &handles, PartitionKind part_kind, Color color,
    const char *provenance) {
  if (!enabled) {
    return lrt->create_cross_product_partitions(ctx, handle1, handle2, handles, part_kind,
                                                color, provenance);
  }

#ifdef RESILIENCE_CROSS_PRODUCT_BYPASS
  if (replay_index_partition()) {
    Color result = lrt->create_cross_product_partitions(ctx, handle1, handle2, handles,
                                                        part_kind, color, provenance);
    Domain domain = lrt->get_index_partition_color_space(handle1);
    for (Domain::DomainPointIterator i(domain); i; ++i) {
      assert(replay_index_partition());
      IndexSpace subspace = lrt->get_index_subspace(handle1, *i);
      IndexPartition sub_ip = lrt->get_index_partition(subspace, result);
      restore_index_partition_bypass(ctx, sub_ip);
    }
    return result;
  }
#else
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
#endif

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

IndexPartition Runtime::create_partition_by_restriction(
    Context ctx, IndexSpace parent, IndexSpace color_space, DomainTransform transform,
    Domain extent, PartitionKind part_kind, Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_restriction(ctx, parent, color_space, transform,
                                                extent, part_kind, color, provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, color_space, color, provenance);
  }

  IndexPartition ip = lrt->create_partition_by_restriction(
      ctx, parent, color_space, transform, extent, part_kind, color, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_blockify(Context ctx, IndexSpace parent,
                                                     DomainPoint blocking_factor,
                                                     Color color,
                                                     const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_blockify(ctx, parent, blocking_factor, color,
                                             provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, provenance);
  }

  IndexPartition ip =
      lrt->create_partition_by_blockify(ctx, parent, blocking_factor, color, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_blockify(Context ctx, IndexSpace parent,
                                                     DomainPoint blocking_factor,
                                                     DomainPoint origin, Color color,
                                                     const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_blockify(ctx, parent, blocking_factor, origin, color,
                                             provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, provenance);
  }

  IndexPartition ip = lrt->create_partition_by_blockify(ctx, parent, blocking_factor,
                                                        origin, color, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_domain(
    Context ctx, IndexSpace parent, const std::map<DomainPoint, Domain> &domains,
    IndexSpace color_space, bool perform_intersections, PartitionKind part_kind,
    Color color, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_domain(ctx, parent, domains, color_space,
                                           perform_intersections, part_kind, color,
                                           provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, provenance);
  }

  IndexPartition ip = lrt->create_partition_by_domain(ctx, parent, domains, color_space,
                                                      perform_intersections, part_kind,
                                                      color, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_domain(Context ctx, IndexSpace parent,
                                                   const FutureMap &domain_future_map,
                                                   IndexSpace color_space,
                                                   bool perform_intersections,
                                                   PartitionKind part_kind, Color color,
                                                   const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_domain(ctx, parent, domain_future_map, color_space,
                                           perform_intersections, part_kind, color,
                                           provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, IndexSpace::NO_SPACE, color, provenance);
  }

  IndexPartition ip = lrt->create_partition_by_domain(ctx, parent, domain_future_map,
                                                      color_space, perform_intersections,
                                                      part_kind, color, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_field(
    Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid,
    IndexSpace color_space, Color color, MapperID id, MappingTagID tag,
    PartitionKind part_kind, UntypedBuffer map_arg, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_field(ctx, handle, parent, fid, color_space, color,
                                          id, tag, part_kind, map_arg, provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, handle.get_index_space(), color_space, color,
                                   provenance);
  }

  IndexPartition ip =
      lrt->create_partition_by_field(ctx, handle, parent, fid, color_space, color, id,
                                     tag, part_kind, map_arg, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_image(
    Context ctx, IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
    FieldID fid, IndexSpace color_space, PartitionKind part_kind, Color color,
    MapperID id, MappingTagID tag, UntypedBuffer map_arg, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_image(ctx, handle, projection, parent, fid,
                                          color_space, part_kind, color, id, tag, map_arg,
                                          provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, handle, color_space, color, provenance);
  }

  IndexPartition ip =
      lrt->create_partition_by_image(ctx, handle, projection, parent, fid, color_space,
                                     part_kind, color, id, tag, map_arg, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_image_range(
    Context ctx, IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
    FieldID fid, IndexSpace color_space, PartitionKind part_kind, Color color,
    MapperID id, MappingTagID tag, UntypedBuffer map_arg, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_image_range(ctx, handle, projection, parent, fid,
                                                color_space, part_kind, color, id, tag,
                                                map_arg, provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, handle, color_space, color, provenance);
  }

  IndexPartition ip = lrt->create_partition_by_image_range(
      ctx, handle, projection, parent, fid, color_space, part_kind, color, id, tag,
      map_arg, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_preimage(
    Context ctx, IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
    FieldID fid, IndexSpace color_space, PartitionKind part_kind, Color color,
    MapperID id, MappingTagID tag, UntypedBuffer map_arg, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_preimage(ctx, projection, handle, parent, fid,
                                             color_space, part_kind, color, id, tag,
                                             map_arg, provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, handle.get_index_space(), color_space, color,
                                   provenance);
  }

  IndexPartition ip =
      lrt->create_partition_by_preimage(ctx, projection, handle, parent, fid, color_space,
                                        part_kind, color, id, tag, map_arg, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_partition_by_preimage_range(
    Context ctx, IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
    FieldID fid, IndexSpace color_space, PartitionKind part_kind, Color color,
    MapperID id, MappingTagID tag, UntypedBuffer map_arg, const char *provenance) {
  if (!enabled) {
    return lrt->create_partition_by_preimage_range(ctx, projection, handle, parent, fid,
                                                   color_space, part_kind, color, id, tag,
                                                   map_arg, provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, handle.get_index_space(), color_space, color,
                                   provenance);
  }

  IndexPartition ip = lrt->create_partition_by_preimage_range(
      ctx, projection, handle, parent, fid, color_space, part_kind, color, id, tag,
      map_arg, provenance);
  register_index_partition(ip);
  return ip;
}

IndexPartition Runtime::create_pending_partition(Context ctx, IndexSpace parent,
                                                 IndexSpace color_space,
                                                 PartitionKind part_kind, Color color,
                                                 const char *provenance) {
  if (!enabled) {
    return lrt->create_pending_partition(ctx, parent, color_space, part_kind, color,
                                         provenance);
  }

  if (replay_index_partition()) {
    return restore_index_partition(ctx, parent, color_space, color, provenance);
  }

  IndexPartition ip = lrt->create_pending_partition(ctx, parent, color_space, part_kind,
                                                    color, provenance);
  register_index_partition(ip);
  return ip;
}

IndexSpace Runtime::create_index_space_union(Context ctx, IndexPartition parent,
                                             const DomainPoint &color,
                                             const std::vector<IndexSpace> &handles,
                                             const char *provenance) {
  if (!enabled) {
    return lrt->create_index_space_union(ctx, parent, color, handles, provenance);
  }

  if (replay_index_space()) {
    IndexSpace is = lrt->get_index_subspace(ctx, parent, color);
    restore_index_space_recomputed(is);
    return is;
  }

  // Note: we may be double-saving in this case (because the index space is also available
  // through the partition), but that seems worth it to avoid overcomplicating the save
  // code.
  IndexSpace is = lrt->create_index_space_union(ctx, parent, color, handles, provenance);
  register_index_space(is);
  return is;
}

IndexSpace Runtime::create_index_space_union(Context ctx, IndexPartition parent,
                                             const DomainPoint &color,
                                             IndexPartition handle,
                                             const char *provenance) {
  if (!enabled) {
    return lrt->create_index_space_union(ctx, parent, color, handle, provenance);
  }

  if (replay_index_space()) {
    IndexSpace is = lrt->get_index_subspace(ctx, parent, color);
    restore_index_space_recomputed(is);
    return is;
  }

  // Note: we may be double-saving in this case (because the index space is also available
  // through the partition), but that seems worth it to avoid overcomplicating the save
  // code.
  IndexSpace is = lrt->create_index_space_union(ctx, parent, color, handle, provenance);
  register_index_space(is);
  return is;
}

IndexSpace Runtime::create_index_space_intersection(
    Context ctx, IndexPartition parent, const DomainPoint &color,
    const std::vector<IndexSpace> &handles, const char *provenance) {
  if (!enabled) {
    return lrt->create_index_space_intersection(ctx, parent, color, handles, provenance);
  }

  if (replay_index_space()) {
    IndexSpace is = lrt->get_index_subspace(ctx, parent, color);
    restore_index_space_recomputed(is);
    return is;
  }

  // Note: we may be double-saving in this case (because the index space is also available
  // through the partition), but that seems worth it to avoid overcomplicating the save
  // code.
  IndexSpace is =
      lrt->create_index_space_intersection(ctx, parent, color, handles, provenance);
  register_index_space(is);
  return is;
}

IndexSpace Runtime::create_index_space_intersection(Context ctx, IndexPartition parent,
                                                    const DomainPoint &color,
                                                    IndexPartition handle,
                                                    const char *provenance) {
  if (!enabled) {
    return lrt->create_index_space_intersection(ctx, parent, color, handle, provenance);
  }

  if (replay_index_space()) {
    IndexSpace is = lrt->get_index_subspace(ctx, parent, color);
    restore_index_space_recomputed(is);
    return is;
  }

  // Note: we may be double-saving in this case (because the index space is also available
  // through the partition), but that seems worth it to avoid overcomplicating the save
  // code.
  IndexSpace is =
      lrt->create_index_space_intersection(ctx, parent, color, handle, provenance);
  register_index_space(is);
  return is;
}

IndexSpace Runtime::create_index_space_difference(Context ctx, IndexPartition parent,
                                                  const DomainPoint &color,
                                                  IndexSpace initial,
                                                  const std::vector<IndexSpace> &handles,
                                                  const char *provenance) {
  if (!enabled) {
    return lrt->create_index_space_difference(ctx, parent, color, initial, handles,
                                              provenance);
  }

  if (replay_index_space()) {
    IndexSpace is = lrt->get_index_subspace(ctx, parent, color);
    restore_index_space_recomputed(is);
    return is;
  }

  // Note: we may be double-saving in this case (because the index space is also available
  // through the partition), but that seems worth it to avoid overcomplicating the save
  // code.
  IndexSpace is = lrt->create_index_space_difference(ctx, parent, color, initial, handles,
                                                     provenance);
  register_index_space(is);
  return is;
}

IndexPartition Runtime::get_index_partition(Context ctx, IndexSpace parent, Color color) {
  return lrt->get_index_partition(ctx, parent, color);
}

IndexPartition Runtime::get_index_partition(Context ctx, IndexSpace parent,
                                            const DomainPoint &color) {
  return lrt->get_index_partition(ctx, parent, color);
}

IndexPartition Runtime::get_index_partition(IndexSpace parent, Color color) {
  return lrt->get_index_partition(parent, color);
}

IndexPartition Runtime::get_index_partition(IndexSpace parent, const DomainPoint &color) {
  return lrt->get_index_partition(parent, color);
}

IndexSpace Runtime::get_index_subspace(Context ctx, IndexPartition p, Color color) {
  return lrt->get_index_subspace(ctx, p, color);
}

IndexSpace Runtime::get_index_subspace(Context ctx, IndexPartition p,
                                       const DomainPoint &color) {
  return lrt->get_index_subspace(ctx, p, color);
}

IndexSpace Runtime::get_index_subspace(IndexPartition p, Color color) {
  return lrt->get_index_subspace(p, color);
}

IndexSpace Runtime::get_index_subspace(IndexPartition p, const DomainPoint &color) {
  return lrt->get_index_subspace(p, color);
}

bool Runtime::has_index_subspace(Context ctx, IndexPartition p,
                                 const DomainPoint &color) {
  return lrt->has_index_subspace(ctx, p, color);
}

bool Runtime::has_index_subspace(IndexPartition p, const DomainPoint &color) {
  return lrt->has_index_subspace(p, color);
}

bool Runtime::has_multiple_domains(Context ctx, IndexSpace handle) {
  return lrt->has_multiple_domains(ctx, handle);
}

bool Runtime::has_multiple_domains(IndexSpace handle) {
  return lrt->has_multiple_domains(handle);
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

Domain Runtime::get_index_partition_color_space(IndexPartition p) {
  return lrt->get_index_partition_color_space(p);
}

IndexSpace Runtime::get_index_partition_color_space_name(Context ctx, IndexPartition p) {
  return lrt->get_index_partition_color_space_name(ctx, p);
}

IndexSpace Runtime::get_index_partition_color_space_name(IndexPartition p) {
  return lrt->get_index_partition_color_space_name(p);
}

bool Runtime::is_index_partition_disjoint(Context ctx, IndexPartition p) {
  return lrt->is_index_partition_disjoint(ctx, p);
}

bool Runtime::is_index_partition_disjoint(IndexPartition p) {
  return lrt->is_index_partition_disjoint(p);
}

bool Runtime::is_index_partition_complete(Context ctx, IndexPartition p) {
  return lrt->is_index_partition_complete(ctx, p);
}

bool Runtime::is_index_partition_complete(IndexPartition p) {
  return lrt->is_index_partition_complete(p);
}

Color Runtime::get_index_space_color(Context ctx, IndexSpace handle) {
  return lrt->get_index_space_color(ctx, handle);
}

DomainPoint Runtime::get_index_space_color_point(Context ctx, IndexSpace handle) {
  return lrt->get_index_space_color_point(ctx, handle);
}

Color Runtime::get_index_space_color(IndexSpace handle) {
  return lrt->get_index_space_color(handle);
}

DomainPoint Runtime::get_index_space_color_point(IndexSpace handle) {
  return lrt->get_index_space_color_point(handle);
}

Color Runtime::get_index_partition_color(Context ctx, IndexPartition handle) {
  return lrt->get_index_partition_color(ctx, handle);
}

DomainPoint Runtime::get_index_partition_color_point(Context ctx, IndexPartition handle) {
  return lrt->get_index_partition_color(ctx, handle);
}

Color Runtime::get_index_partition_color(IndexPartition handle) {
  return lrt->get_index_partition_color(handle);
}

DomainPoint Runtime::get_index_partition_color_point(IndexPartition handle) {
  return lrt->get_index_partition_color(handle);
}

IndexSpace Runtime::get_parent_index_space(Context ctx, IndexPartition handle) {
  return lrt->get_parent_index_space(ctx, handle);
}

IndexSpace Runtime::get_parent_index_space(IndexPartition handle) {
  return lrt->get_parent_index_space(handle);
}

bool Runtime::has_parent_index_partition(Context ctx, IndexSpace handle) {
  return lrt->has_parent_index_partition(ctx, handle);
}

bool Runtime::has_parent_index_partition(IndexSpace handle) {
  return lrt->has_parent_index_partition(handle);
}

IndexPartition Runtime::get_parent_index_partition(Context ctx, IndexSpace handle) {
  return lrt->get_parent_index_partition(ctx, handle);
}

IndexPartition Runtime::get_parent_index_partition(IndexSpace handle) {
  return lrt->get_parent_index_partition(handle);
}

ptr_t Runtime::safe_cast(Context ctx, ptr_t pointer, LogicalRegion region) {
  return lrt->safe_cast(ctx, pointer, region);
}

DomainPoint Runtime::safe_cast(Context ctx, DomainPoint point, LogicalRegion region) {
  return lrt->safe_cast(ctx, point, region);
}

FieldSpace Runtime::create_field_space(Context ctx, const char *provenance) {
  return lrt->create_field_space(ctx, provenance);
}

FieldSpace Runtime::create_field_space(Context ctx,
                                       const std::vector<size_t> &field_sizes,
                                       std::vector<FieldID> &resulting_fields,
                                       CustomSerdezID serdez_id, const char *provenance) {
  return lrt->create_field_space(ctx, field_sizes, resulting_fields, serdez_id,
                                 provenance);
}

FieldSpace Runtime::create_field_space(Context ctx,
                                       const std::vector<Future> &field_sizes,
                                       std::vector<FieldID> &resulting_fields,
                                       CustomSerdezID serdez_id, const char *provenance) {
  std::vector<Legion::Future> field_sizes_;
  for (auto &f : field_sizes) {
    field_sizes_.push_back(f);
  }

  return lrt->create_field_space(ctx, field_sizes_, resulting_fields, serdez_id,
                                 provenance);
}

void Runtime::create_shared_ownership(Context ctx, FieldSpace handle) {
  lrt->create_shared_ownership(ctx, handle);
}

void Runtime::destroy_field_space(Context ctx, FieldSpace handle, const bool unordered,
                                  const char *provenance) {
  lrt->destroy_field_space(ctx, handle, unordered, provenance);
}

size_t Runtime::get_field_size(Context ctx, FieldSpace handle, FieldID fid) {
  return lrt->get_field_size(ctx, handle, fid);
}

size_t Runtime::get_field_size(FieldSpace handle, FieldID fid) {
  return lrt->get_field_size(handle, fid);
}

void Runtime::get_field_space_fields(Context ctx, FieldSpace handle,
                                     std::vector<FieldID> &fields) {
  lrt->get_field_space_fields(ctx, handle, fields);
}

void Runtime::get_field_space_fields(FieldSpace handle, std::vector<FieldID> &fields) {
  lrt->get_field_space_fields(handle, fields);
}

void Runtime::get_field_space_fields(Context ctx, FieldSpace handle,
                                     std::set<FieldID> &fields) {
  lrt->get_field_space_fields(ctx, handle, fields);
}

void Runtime::get_field_space_fields(FieldSpace handle, std::set<FieldID> &fields) {
  lrt->get_field_space_fields(handle, fields);
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

void Runtime::create_shared_ownership(Context ctx, LogicalRegion handle) {
  assert(false && "unimplemented");
  abort();
}

void Runtime::destroy_logical_region(Context ctx, LogicalRegion handle,
                                     const bool unordered, const char *provenance) {
  if (!enabled) {
    lrt->destroy_logical_region(ctx, handle, unordered, provenance);
    return;
  }

  auto region_tag = region_tags.at(handle);
  auto &lr_state = state.region_state.at(region_tag);
  lr_state.destroyed = true;
  lrt->destroy_logical_region(ctx, handle, unordered, provenance);
}

void Runtime::destroy_logical_partition(Context ctx, LogicalPartition handle,
                                        const bool unordered) {
  return lrt->destroy_logical_partition(ctx, handle, unordered);
}

LogicalPartition Runtime::get_logical_partition(Context ctx, LogicalRegion parent,
                                                IndexPartition handle) {
  return lrt->get_logical_partition(ctx, parent, handle);
}

LogicalPartition Runtime::get_logical_partition(LogicalRegion parent,
                                                IndexPartition handle) {
  return lrt->get_logical_partition(parent, handle);
}

LogicalPartition Runtime::get_logical_partition_by_color(Context ctx,
                                                         LogicalRegion parent, Color c) {
  return lrt->get_logical_partition_by_color(ctx, parent, c);
}

LogicalPartition Runtime::get_logical_partition_by_color(Context ctx,
                                                         LogicalRegion parent,
                                                         const DomainPoint &c) {
  return lrt->get_logical_partition_by_color(ctx, parent, c);
}

LogicalPartition Runtime::get_logical_partition_by_color(LogicalRegion parent, Color c) {
  return lrt->get_logical_partition_by_color(parent, c);
}

LogicalPartition Runtime::get_logical_partition_by_color(LogicalRegion parent,
                                                         const DomainPoint &c) {
  return lrt->get_logical_partition_by_color(parent, c);
}

LogicalPartition Runtime::get_logical_partition_by_tree(Context ctx,
                                                        IndexPartition handle,
                                                        FieldSpace fspace,
                                                        RegionTreeID tid) {
  return lrt->get_logical_partition_by_tree(ctx, handle, fspace, tid);
}

LogicalPartition Runtime::get_logical_partition_by_tree(IndexPartition handle,
                                                        FieldSpace fspace,
                                                        RegionTreeID tid) {
  return lrt->get_logical_partition_by_tree(handle, fspace, tid);
}

LogicalRegion Runtime::get_logical_subregion(Context ctx, LogicalPartition parent,
                                             IndexSpace handle) {
  return lrt->get_logical_subregion(ctx, parent, handle);
}

LogicalRegion Runtime::get_logical_subregion(LogicalPartition parent, IndexSpace handle) {
  return lrt->get_logical_subregion(parent, handle);
}

LogicalRegion Runtime::get_logical_subregion_by_color(Context ctx,
                                                      LogicalPartition parent, Color c) {
  return lrt->get_logical_subregion_by_color(ctx, parent, c);
}

LogicalRegion Runtime::get_logical_subregion_by_color(Context ctx,
                                                      LogicalPartition parent,
                                                      const DomainPoint &c) {
  return lrt->get_logical_subregion_by_color(ctx, parent, c);
}

LogicalRegion Runtime::get_logical_subregion_by_color(LogicalPartition parent, Color c) {
  return lrt->get_logical_subregion_by_color(parent, c);
}

LogicalRegion Runtime::get_logical_subregion_by_color(LogicalPartition parent,
                                                      const DomainPoint &c) {
  return lrt->get_logical_subregion_by_color(parent, c);
}

bool Runtime::has_logical_subregion_by_color(Context ctx, LogicalPartition parent,
                                             const DomainPoint &c) {
  return lrt->has_logical_subregion_by_color(ctx, parent, c);
}

bool Runtime::has_logical_subregion_by_color(LogicalPartition parent,
                                             const DomainPoint &c) {
  return lrt->has_logical_subregion_by_color(parent, c);
}

LogicalRegion Runtime::get_logical_subregion_by_tree(Context ctx, IndexSpace handle,
                                                     FieldSpace fspace,
                                                     RegionTreeID tid) {
  return lrt->get_logical_subregion_by_tree(ctx, handle, fspace, tid);
}

LogicalRegion Runtime::get_logical_subregion_by_tree(IndexSpace handle, FieldSpace fspace,
                                                     RegionTreeID tid) {
  return lrt->get_logical_subregion_by_tree(handle, fspace, tid);
}

Color Runtime::get_logical_region_color(Context ctx, LogicalRegion handle) {
  return lrt->get_logical_region_color(ctx, handle);
}

DomainPoint Runtime::get_logical_region_color_point(Context ctx, LogicalRegion handle) {
  return lrt->get_logical_region_color_point(ctx, handle);
}

Color Runtime::get_logical_region_color(LogicalRegion handle) {
  return lrt->get_logical_region_color(handle);
}

DomainPoint Runtime::get_logical_region_color_point(LogicalRegion handle) {
  return lrt->get_logical_region_color_point(handle);
}

Color Runtime::get_logical_partition_color(Context ctx, LogicalPartition handle) {
  return lrt->get_logical_partition_color(ctx, handle);
}

DomainPoint Runtime::get_logical_partition_color_point(Context ctx,
                                                       LogicalPartition handle) {
  return lrt->get_logical_partition_color_point(ctx, handle);
}

Color Runtime::get_logical_partition_color(LogicalPartition handle) {
  return lrt->get_logical_partition_color(handle);
}

DomainPoint Runtime::get_logical_partition_color_point(LogicalPartition handle) {
  return lrt->get_logical_partition_color_point(handle);
}

LogicalRegion Runtime::get_parent_logical_region(Context ctx, LogicalPartition handle) {
  return lrt->get_parent_logical_region(ctx, handle);
}

LogicalRegion Runtime::get_parent_logical_region(LogicalPartition handle) {
  return lrt->get_parent_logical_region(handle);
}

bool Runtime::has_parent_logical_partition(Context ctx, LogicalRegion handle) {
  return lrt->has_parent_logical_partition(ctx, handle);
}

bool Runtime::has_parent_logical_partition(LogicalRegion handle) {
  return lrt->has_parent_logical_partition(handle);
}

LogicalPartition Runtime::get_parent_logical_partition(Context ctx,
                                                       LogicalRegion handle) {
  return lrt->get_parent_logical_partition(ctx, handle);
}

LogicalPartition Runtime::get_parent_logical_partition(LogicalRegion handle) {
  return lrt->get_parent_logical_partition(handle);
}

FieldAllocator Runtime::create_field_allocator(Context ctx, FieldSpace handle) {
  return lrt->create_field_allocator(ctx, handle);
}

bool Runtime::is_partition_eligible(IndexPartition ip) {
  if (ip == IndexPartition::NO_PART) {
    return false;
  }

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

Future Runtime::execute_task(Context ctx, const TaskLauncher &launcher,
                             std::vector<OutputRequirement> *outputs) {
  if (!enabled) {
    return Future(NULL, lrt->execute_task(ctx, launcher));
  }

  assert(outputs == NULL);  // TODO: support output requirements

  if (replay_future()) {
    log_resilience.info() << "execute_task: no-op for replay";
    return restore_future();
  }
  log_resilience.info() << "execute_task: launching task_id " << launcher.task_id;

  assert(!replay || checkpoint_tag >= max_checkpoint_tag);

  for (auto &rr : launcher.region_requirements) {
    track_region_state(rr);
  }

  Future f(this, lrt->execute_task(ctx, launcher));
  register_future(f);
  return f;
}

FutureMap Runtime::execute_index_space(Context ctx, const IndexTaskLauncher &launcher,
                                       std::vector<OutputRequirement> *outputs) {
  if (!enabled) {
    Legion::FutureMap lfm = lrt->execute_index_space(ctx, launcher);
    FutureMap rfm;
    if (launcher.launch_domain == Domain::NO_DOMAIN)
      return FutureMap(NULL, lrt->get_index_space_domain(launcher.launch_space), lfm);
    else
      return FutureMap(NULL, launcher.launch_domain, lfm);
  }

  assert(outputs == NULL);  // TODO: support output requirements

  if (replay_future_map()) {
    log_resilience.info() << "execute_index_space: no-op for replay";
    return restore_future_map(ctx);
  }

  assert(!replay || checkpoint_tag >= max_checkpoint_tag);

  for (auto &rr : launcher.region_requirements) {
    track_region_state(rr);
  }

  Legion::FutureMap fm = lrt->execute_index_space(ctx, launcher);

  FutureMap rfm;
  if (launcher.launch_domain == Domain::NO_DOMAIN)
    rfm = FutureMap(this, lrt->get_index_space_domain(launcher.launch_space), fm);
  else
    rfm = FutureMap(this, launcher.launch_domain, fm);
  register_future_map(rfm);
  return rfm;
}

Future Runtime::execute_index_space(Context ctx, const IndexTaskLauncher &launcher,
                                    ReductionOpID redop, bool deterministic,
                                    std::vector<OutputRequirement> *outputs) {
  if (!enabled) {
    return Future(NULL, lrt->execute_index_space(ctx, launcher, redop, deterministic));
  }

  assert(outputs == NULL);  // TODO: support output requirements

  if (replay_future()) {
    log_resilience.info() << "execute_index_space: no-op for replay";
    return restore_future();
  }

  assert(!replay || checkpoint_tag >= max_checkpoint_tag);

  for (auto &rr : launcher.region_requirements) {
    track_region_state(rr);
  }

  Future f(this, lrt->execute_index_space(ctx, launcher, redop, deterministic));
  register_future(f);
  return f;
}

PhysicalRegion Runtime::map_region(Context ctx, const InlineLauncher &launcher) {
  if (!enabled || allow_inline_mapping) {
    return lrt->map_region(ctx, launcher);
  }

  log_resilience.error() << "Inline mappings are not permitted in checkpointed tasks";
  abort();
}

void Runtime::unmap_region(Context ctx, PhysicalRegion region) {
  return lrt->unmap_region(ctx, region);
}

void Runtime::unmap_all_regions(Context ctx) { lrt->unmap_all_regions(ctx); }

void Runtime::fill_field(Context ctx, LogicalRegion handle, LogicalRegion parent,
                         FieldID fid, const void *value, size_t value_size,
                         Predicate pred) {
  if (!enabled) {
    lrt->fill_field(ctx, handle, parent, fid, value, value_size, pred);
  }

  if (skip_api_call()) return;
  lrt->fill_field(ctx, handle, parent, fid, value, value_size, pred);
}

void Runtime::fill_fields(Context ctx, const FillLauncher &launcher) {
  if (!enabled) {
    lrt->fill_fields(ctx, launcher);
    return;
  }

  if (skip_api_call()) return;
  lrt->fill_fields(ctx, launcher);
}

void Runtime::fill_fields(Context ctx, const IndexFillLauncher &launcher) {
  if (!enabled) {
    lrt->fill_fields(ctx, launcher);
    return;
  }

  if (skip_api_call()) return;
  lrt->fill_fields(ctx, launcher);
}

PhysicalRegion Runtime::attach_external_resource(Context ctx,
                                                 const AttachLauncher &launcher) {
  if (!enabled) {
    return lrt->attach_external_resource(ctx, launcher);
  }

  // FIXME (Elliott): not safe to skip??
  // if (skip_api_call()) return;
  return lrt->attach_external_resource(ctx, launcher);
}

Future Runtime::detach_external_resource(Context ctx, PhysicalRegion region,
                                         const bool flush, const bool unordered,
                                         const char *provenance) {
  if (!enabled) {
    return Future(
        NULL, lrt->detach_external_resource(ctx, region, flush, unordered, provenance));
  }

  // FIXME (Elliott): not safe to skip??
  // if (replay_future()) {
  //   return restore_future();
  // }

  Future f(this,
           lrt->detach_external_resource(ctx, region, flush, unordered, provenance));
  register_future(f);
  return f;
}

void Runtime::issue_copy_operation(Context ctx, const CopyLauncher &launcher) {
  if (skip_api_call()) return;
  lrt->issue_copy_operation(ctx, launcher);
}

void Runtime::issue_copy_operation(Context ctx, const IndexCopyLauncher &launcher) {
  if (skip_api_call()) return;
  lrt->issue_copy_operation(ctx, launcher);
}

Predicate Runtime::create_predicate(Context ctx, const Future &f,
                                    const char *provenance) {
  if (!enabled) {
    return lrt->create_predicate(ctx, f, provenance);
  }

  future_state[f.lft].escaped = true;
  return lrt->create_predicate(ctx, f.lft, provenance);
}

Predicate Runtime::create_predicate(Context ctx, const PredicateLauncher &launcher) {
  if (!enabled) {
    return lrt->create_predicate(ctx, launcher);
  }

  return lrt->create_predicate(ctx, launcher);
}

Predicate Runtime::predicate_not(Context ctx, const Predicate &p,
                                 const char *provenance) {
  return lrt->predicate_not(ctx, p, provenance);
}

Future Runtime::get_predicate_future(Context ctx, const Predicate &p,
                                     const char *provenance) {
  if (!enabled) {
    return Future(NULL, lrt->get_predicate_future(ctx, p, provenance));
  }

  if (replay_future()) {
    return restore_future();
  }

  Future f(this, lrt->get_predicate_future(ctx, p, provenance));
  register_future(f);
  return f;
}

Future Runtime::issue_mapping_fence(Context ctx, const char *provenance) {
  if (!enabled) {
    return Future(NULL, lrt->issue_mapping_fence(ctx, provenance));
  }

  if (replay_future()) {
    return restore_future();
  }

  Future f(this, lrt->issue_mapping_fence(ctx, provenance));
  register_future(f);
  return f;
}

Future Runtime::issue_execution_fence(Context ctx, const char *provenance) {
  if (!enabled) {
    return Future(NULL, lrt->issue_execution_fence(ctx, provenance));
  }

  if (replay_future()) {
    return restore_future();
  }

  Future f(this, lrt->issue_execution_fence(ctx, provenance));
  register_future(f);
  return f;
}

void Runtime::begin_trace(Context ctx, TraceID tid, bool logical_only, bool static_trace,
                          const std::set<RegionTreeID> *managed, const char *provenance) {
  if (skip_api_call()) return;
  lrt->begin_trace(ctx, tid, logical_only, static_trace, managed, provenance);
}

void Runtime::end_trace(Context ctx, TraceID tid, const char *provenance) {
  if (skip_api_call()) return;
  lrt->end_trace(ctx, tid, provenance);
}

TraceID Runtime::generate_dynamic_trace_id(void) {
  return lrt->generate_dynamic_trace_id();
}

TraceID Runtime::generate_library_trace_ids(const char *name, size_t count) {
  return lrt->generate_library_trace_ids(name, count);
}

TraceID Runtime::generate_static_trace_id(void) {
  return Runtime::generate_static_trace_id();
}

Future Runtime::select_tunable_value(Context ctx, TunableID tid, MapperID mapper,
                                     MappingTagID tag, const void *args, size_t argsize) {
  if (!enabled) {
    return Future(NULL, lrt->select_tunable_value(ctx, tid, mapper, tag, args, argsize));
  }

  if (replay_future()) {
    return restore_future();
  }

  Future f(this, lrt->select_tunable_value(ctx, tid, mapper, tag, args, argsize));
  register_future(f);
  return f;
}

Future Runtime::select_tunable_value(Context ctx, const TunableLauncher &launcher) {
  if (!enabled) {
    return Future(NULL, lrt->select_tunable_value(ctx, launcher));
  }

  if (replay_future()) {
    return restore_future();
  }

  Future f(this, lrt->select_tunable_value(ctx, launcher));
  register_future(f);
  return f;
}

Future Runtime::get_current_time(Context ctx, Future precondition) {
  if (!enabled) {
    return Future(NULL, lrt->get_current_time(ctx, precondition));
  }

  if (replay_future()) {
    return restore_future();
  }

  Future f(this, lrt->get_current_time(ctx, precondition));
  register_future(f);
  return f;
}

Future Runtime::get_current_time_in_microseconds(Context ctx, Future precondition) {
  if (!enabled) {
    return Future(NULL, lrt->get_current_time_in_microseconds(ctx, precondition));
  }

  if (replay_future()) {
    return restore_future();
  }

  Future f(this, lrt->get_current_time_in_microseconds(ctx, precondition));
  register_future(f);
  return f;
}

Future Runtime::get_current_time_in_nanoseconds(Context ctx, Future precondition) {
  if (!enabled) {
    return Future(NULL, lrt->get_current_time_in_nanoseconds(ctx, precondition));
  }

  if (replay_future()) {
    return restore_future();
  }

  Future f(this, lrt->get_current_time_in_nanoseconds(ctx, precondition));
  register_future(f);
  return f;
}

Future Runtime::issue_timing_measurement(Context ctx, const TimingLauncher &launcher) {
  if (!enabled) {
    return Future(NULL, lrt->issue_timing_measurement(ctx, launcher));
  }

  if (replay_future()) {
    return restore_future();
  }

  Future f(this, lrt->issue_timing_measurement(ctx, launcher));
  register_future(f);
  return f;
}

Processor Runtime::get_executing_processor(Context ctx) {
  return lrt->get_executing_processor(ctx);
}

void Runtime::attach_semantic_information(TaskID task_id, SemanticTag tag,
                                          const void *buffer, size_t size,
                                          bool is_mutable, bool local_only) {
  return lrt->attach_semantic_information(task_id, tag, buffer, size, is_mutable,
                                          local_only);
}

void Runtime::attach_semantic_information(IndexSpace handle, SemanticTag tag,
                                          const void *buffer, size_t size,
                                          bool is_mutable) {
  return lrt->attach_semantic_information(handle, tag, buffer, size, is_mutable);
}

void Runtime::attach_semantic_information(IndexPartition handle, SemanticTag tag,
                                          const void *buffer, size_t size,
                                          bool is_mutable) {
  return lrt->attach_semantic_information(handle, tag, buffer, size, is_mutable);
}

void Runtime::attach_semantic_information(FieldSpace handle, SemanticTag tag,
                                          const void *buffer, size_t size,
                                          bool is_mutable) {
  return lrt->attach_semantic_information(handle, tag, buffer, size, is_mutable);
}

void Runtime::attach_semantic_information(FieldSpace handle, FieldID fid, SemanticTag tag,
                                          const void *buffer, size_t size,
                                          bool is_mutable) {
  return lrt->attach_semantic_information(handle, fid, tag, buffer, size, is_mutable);
}

void Runtime::attach_semantic_information(LogicalRegion handle, SemanticTag tag,
                                          const void *buffer, size_t size,
                                          bool is_mutable) {
  return lrt->attach_semantic_information(handle, tag, buffer, size, is_mutable);
}

void Runtime::attach_semantic_information(LogicalPartition handle, SemanticTag tag,
                                          const void *buffer, size_t size,
                                          bool is_mutable) {
  return lrt->attach_semantic_information(handle, tag, buffer, size, is_mutable);
}

void Runtime::attach_name(TaskID task_id, const char *name, bool is_mutable,
                          bool local_only) {
  lrt->attach_name(task_id, name, is_mutable, local_only);
}

void Runtime::attach_name(IndexSpace handle, const char *name, bool is_mutable) {
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

void Runtime::attach_name(FieldSpace handle, const char *name, bool is_mutable) {
  lrt->attach_name(handle, name, is_mutable);
}

void Runtime::attach_name(FieldSpace handle, FieldID fid, const char *name,
                          bool is_mutable) {
  lrt->attach_name(handle, fid, name, is_mutable);
}

void Runtime::attach_name(LogicalRegion handle, const char *name, bool is_mutable) {
  lrt->attach_name(handle, name, is_mutable);
}

void Runtime::attach_name(LogicalPartition handle, const char *name, bool is_mutable) {
  lrt->attach_name(handle, name, is_mutable);
}

bool Runtime::retrieve_semantic_information(TaskID task_id, SemanticTag tag,
                                            const void *&result, size_t &size,
                                            bool can_fail, bool wait_until_ready) {
  return lrt->retrieve_semantic_information(task_id, tag, result, size, can_fail,
                                            wait_until_ready);
}

bool Runtime::retrieve_semantic_information(IndexSpace handle, SemanticTag tag,
                                            const void *&result, size_t &size,
                                            bool can_fail, bool wait_until_ready) {
  return lrt->retrieve_semantic_information(handle, tag, result, size, can_fail,
                                            wait_until_ready);
}

bool Runtime::retrieve_semantic_information(IndexPartition handle, SemanticTag tag,
                                            const void *&result, size_t &size,
                                            bool can_fail, bool wait_until_ready) {
  return lrt->retrieve_semantic_information(handle, tag, result, size, can_fail,
                                            wait_until_ready);
}

bool Runtime::retrieve_semantic_information(FieldSpace handle, SemanticTag tag,
                                            const void *&result, size_t &size,
                                            bool can_fail, bool wait_until_ready) {
  return lrt->retrieve_semantic_information(handle, tag, result, size, can_fail,
                                            wait_until_ready);
}

bool Runtime::retrieve_semantic_information(FieldSpace handle, FieldID fid,
                                            SemanticTag tag, const void *&result,
                                            size_t &size, bool can_fail,
                                            bool wait_until_ready) {
  return lrt->retrieve_semantic_information(handle, fid, tag, result, size, can_fail,
                                            wait_until_ready);
}

bool Runtime::retrieve_semantic_information(LogicalRegion handle, SemanticTag tag,
                                            const void *&result, size_t &size,
                                            bool can_fail, bool wait_until_ready) {
  return lrt->retrieve_semantic_information(handle, tag, result, size, can_fail,
                                            wait_until_ready);
}

bool Runtime::retrieve_semantic_information(LogicalPartition handle, SemanticTag tag,
                                            const void *&result, size_t &size,
                                            bool can_fail, bool wait_until_ready) {
  return lrt->retrieve_semantic_information(handle, tag, result, size, can_fail,
                                            wait_until_ready);
}

void Runtime::retrieve_name(TaskID task_id, const char *&result) {
  lrt->retrieve_name(task_id, result);
}

void Runtime::retrieve_name(IndexSpace handle, const char *&result) {
  lrt->retrieve_name(handle, result);
}

void Runtime::retrieve_name(IndexPartition handle, const char *&result) {
  lrt->retrieve_name(handle, result);
}

void Runtime::retrieve_name(FieldSpace handle, const char *&result) {
  lrt->retrieve_name(handle, result);
}

void Runtime::retrieve_name(FieldSpace handle, FieldID fid, const char *&result) {
  lrt->retrieve_name(handle, fid, result);
}

void Runtime::retrieve_name(LogicalRegion handle, const char *&result) {
  lrt->retrieve_name(handle, result);
}

void Runtime::retrieve_name(LogicalPartition handle, const char *&result) {
  lrt->retrieve_name(handle, result);
}

void Runtime::print_once(Context ctx, FILE *f, const char *message) {
  if (skip_api_call()) return;
  lrt->print_once(ctx, f, message);
}

Mapping::MapperRuntime *Runtime::get_mapper_runtime(void) {
  return lrt->get_mapper_runtime();
}

MapperID Runtime::generate_dynamic_mapper_id(void) {
  return lrt->generate_dynamic_mapper_id();
}

MapperID Runtime::generate_library_mapper_ids(const char *name, size_t count) {
  return lrt->generate_library_mapper_ids(name, count);
}

MapperID Runtime::generate_static_mapper_id(void) {
  return Legion::Runtime::generate_static_mapper_id();
}

void Runtime::add_mapper(MapperID map_id, Mapping::Mapper *mapper, Processor proc) {
  lrt->add_mapper(map_id, mapper, proc);
}

void Runtime::replace_default_mapper(Mapping::Mapper *mapper, Processor proc) {
  lrt->replace_default_mapper(mapper, proc);
}

ProjectionID Runtime::generate_dynamic_projection_id(void) {
  return lrt->generate_dynamic_projection_id();
}

ProjectionID Runtime::generate_library_projection_ids(const char *name, size_t count) {
  return lrt->generate_library_projection_ids(name, count);
}

ProjectionID Runtime::generate_static_projection_id(void) {
  return Legion::Runtime::generate_static_projection_id();
}

void Runtime::register_projection_functor(ProjectionID pid, ProjectionFunctor *functor,
                                          bool silence_warnings,
                                          const char *warning_string) {
  lrt->register_projection_functor(pid, functor, silence_warnings, warning_string);
}

void Runtime::preregister_projection_functor(ProjectionID pid,
                                             ProjectionFunctor *functor) {
  preregistered_projection_functors.push_back(functor);
  Legion::Runtime::preregister_projection_functor(pid, functor);
}

ShardingID Runtime::generate_dynamic_sharding_id(void) {
  return lrt->generate_dynamic_sharding_id();
}

ShardingID Runtime::generate_library_sharding_ids(const char *name, size_t count) {
  return lrt->generate_library_sharding_ids(name, count);
}

ShardingID Runtime::generate_static_sharding_id(void) {
  return Legion::Runtime::generate_static_sharding_id();
}

void Runtime::register_sharding_functor(ShardingID sid, ShardingFunctor *functor,
                                        bool silence_warnings,
                                        const char *warning_string) {
  return lrt->register_sharding_functor(sid, functor, silence_warnings, warning_string);
}

void Runtime::preregister_sharding_functor(ShardingID sid, ShardingFunctor *functor) {
  Legion::Runtime::preregister_sharding_functor(sid, functor);
}

ShardingFunctor *Runtime::get_sharding_functor(ShardingID sid) {
  return Legion::Runtime::get_sharding_functor(sid);
}

ReductionOpID Runtime::generate_dynamic_reduction_id(void) {
  return lrt->generate_dynamic_reduction_id();
}

ReductionOpID Runtime::generate_library_reduction_ids(const char *name, size_t count) {
  return lrt->generate_library_reduction_ids(name, count);
}

ReductionOpID Runtime::generate_static_reduction_id(void) {
  return Legion::Runtime::generate_static_reduction_id();
}

void Runtime::register_reduction_op(ReductionOpID redop_id, ReductionOp *op,
                                    SerdezInitFnptr init_fnptr,
                                    SerdezFoldFnptr fold_fnptr, bool permit_duplicates) {
  Legion::Runtime::register_reduction_op(redop_id, op, init_fnptr, fold_fnptr,
                                         permit_duplicates);
}

const ReductionOp *Runtime::get_reduction_op(ReductionOpID redop_id) {
  return Legion::Runtime::get_reduction_op(redop_id);
}

static void write_checkpoint(const Task *task, const std::vector<PhysicalRegion> &regions,
                             Context ctx, Legion::Runtime *runtime) {
  std::string file_name(
      static_cast<const char *>(task->futures.at(0).get_untyped_pointer()),
      task->futures.at(0).get_untyped_size());
  std::string serialized_data(
      static_cast<const char *>(task->futures.at(1).get_untyped_pointer()),
      task->futures.at(1).get_untyped_size());

  log_resilience.info() << "write_checkpoint: file_name " << file_name << " bytes "
                        << serialized_data.size();
  std::ofstream file(file_name, std::ios::binary);
  // This is a hack, but apparently C++ iostream exception messages are useless, so
  // this is what we've got. See: https://codereview.stackexchange.com/a/58130
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

static void read_checkpoint(const void *args, long unsigned arglen, const void *userdata,
                            long unsigned userlen, Realm::Processor p) {
  const Task *task;
  const std::vector<PhysicalRegion> *reg;
  Context ctx;
  Legion::Runtime *runtime;
  Legion::Runtime::legion_task_preamble(args, arglen, p, task, reg, ctx, runtime);

  std::string file_name(
      static_cast<const char *>(task->futures.at(0).get_untyped_pointer()),
      task->futures.at(0).get_untyped_size());

  std::stringstream ss;
  {
    std::ifstream file(file_name, std::ios::binary);
    // This is a hack, but apparently C++ iostream exception messages are useless, so
    // this is what we've got. See: https://codereview.stackexchange.com/a/58130
    if (!file) {
      log_resilience.error() << "unable to open file '" << file_name
                             << "': " << strerror(errno);
      abort();
    }
    ss << file.rdbuf();
    file.close();
    if (!file) {
      log_resilience.error() << "error in closing file '" << file_name
                             << "': " << strerror(errno);
      abort();
    }
  }
  Legion::Runtime::legion_task_postamble(ctx, ss.str().data(), ss.str().size(),
                                         false /*owned*/);
}

void Runtime::register_mapper(Machine machine, Legion::Runtime *rt,
                              const std::set<Processor> &local_procs) {
  for (auto &proc : local_procs) {
    Mapping::ResilientMapper *mapper = new Mapping::ResilientMapper(
        rt->get_mapper_runtime(), machine, proc, "resilient_mapper");
    rt->add_mapper(resilient_mapper_id, mapper, proc);
  }
}

void Runtime::fix_projection_functors(Machine machine, Legion::Runtime *rt,
                                      const std::set<Processor> &local_procs) {
  // Hack: set runtime on all preregistered projection functors, because Legion won't.

  // We just have to leak this. No way to clean it up.
  Runtime *runtime = new Runtime(rt);
  for (auto &fn : preregistered_projection_functors) {
    fn->set_runtime(runtime);
  }
}

static long parse_long(const std::string &flag, const std::string &arg) {
  long result;
  size_t consumed;
  result = std::stol(arg, &consumed);
  if (consumed != arg.size()) {
    log_resilience.error() << "error in parsing flag: " << flag << " " << arg;
    abort();
  }
  return result;
}

static size_t parse_size_t(const std::string &flag, const std::string &arg) {
  long result = parse_long(flag, arg);
  if (result < 0) {
    log_resilience.error() << "error in parsing flag: " << flag << " " << arg
                           << " (value is negative)";
    abort();
  }
  return result;
}

int Runtime::start(int argc, char **argv, bool background, bool supply_default_mapper) {
  // FIXME: filter out these arguments so applications don't need to see them
  for (int i = 1; i < argc; i++) {
    std::string flag(argv[i]);
    if (flag == "-checkpoint:disable") {
      config_disable = true;
    } else if (flag == "-checkpoint:prefix") {
      config_prefix = argv[++i];
      size_t s = config_prefix.size();
      if (s > 0 && config_prefix.find("/", s - 1) != s - 1) {
        config_prefix += "/";
      }
    } else if (flag == "-checkpoint:replay") {
      std::string arg(argv[++i]);
      config_checkpoint_tag = parse_size_t(flag, arg);
      config_replay = true;
    } else if (flag == "-checkpoint:max_instances") {
      std::string arg(argv[++i]);
      config_max_instances = parse_size_t(flag, arg);
    } else if (flag == "-checkpoint:auto_steps") {
      std::string arg(argv[++i]);
      config_auto_steps = parse_long(flag, arg);
    } else if (flag == "-checkpoint:measure_replay_time_and_exit") {
      config_measure_replay_time_and_exit = true;
    } else if (flag == "-checkpoint:skip_leak_check") {
      config_skip_leak_check = true;
    } else if (flag.rfind("-checkpoint:", 0) == 0) {
      log_resilience.error() << "unknown flag: " << flag;
      abort();
    }
  }

  if (config_disable && config_replay) {
    log_resilience.warning()
        << "flags -checkpoint:disable overrides the value of -checkpoint:replay";
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

  {
    read_checkpoint_task_id = Legion::Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(read_checkpoint_task_id, "read_checkpoint");
    ProcessorConstraint pc;
    pc.add_kind(Processor::LOC_PROC);
    pc.add_kind(Processor::IO_PROC);
    registrar.add_constraint(pc);
    registrar.set_leaf();

    CodeDescriptor code_desc(Realm::Type::from_cpp_type<Processor::TaskFuncPtr>());
    code_desc.add_implementation(
        new Realm::FunctionPointerImplementation((void (*)())read_checkpoint));

    Legion::Runtime::preregister_task_variant(
        registrar, code_desc, NULL, 0, "read_checkpoint", LEGION_AUTO_GENERATE_ID,
        LEGION_MAX_RETURN_SIZE, false /*has_return_type_size*/);
  }

  resilient_mapper_id = generate_static_mapper_id();
  Legion::Runtime::add_registration_callback(Runtime::register_mapper);

  Legion::Runtime::add_registration_callback(Runtime::fix_projection_functors);

  return Legion::Runtime::start(argc, argv, background, supply_default_mapper);
}

void Runtime::initialize(int *argc, char ***argv, bool filter) {
  // We parse our own flags in Runtime::start()
  Legion::Runtime::initialize(argc, argv, filter);
}

int Runtime::wait_for_shutdown(void) { return Legion::Runtime::wait_for_shutdown(); }

void Runtime::set_return_code(int return_code) {
  return Legion::Runtime::set_return_code(return_code);
}

void Runtime::set_top_level_task_id(TaskID top_id) {
  Legion::Runtime::set_top_level_task_id(top_id);
}

size_t Runtime::get_maximum_dimension(void) {
  return Legion::Runtime::get_maximum_dimension();
}

static void callback_wrapper(const RegistrationCallbackArgs &args) {
  auto callback = *static_cast<RegistrationCallbackFnptr *>(args.buffer.get_ptr());
  Runtime new_runtime_(args.runtime);
  Runtime *new_runtime = &new_runtime_;
  callback(args.machine, new_runtime, args.local_procs);
}

void Runtime::add_registration_callback(RegistrationCallbackFnptr callback, bool dedup,
                                        size_t dedup_tag) {
  auto fptr = &callback;
  UntypedBuffer buffer(fptr, sizeof(fptr));
  Legion::Runtime::add_registration_callback(callback_wrapper, buffer, dedup, dedup_tag);
}

void Runtime::set_registration_callback(RegistrationCallbackFnptr callback) {
  auto fptr = &callback;
  UntypedBuffer buffer(fptr, sizeof(fptr));
  // FIXME (Elliott): Legion doesn't support set_registration_callback on the wrapped
  // type, so just have to pass it to add here...
  Legion::Runtime::add_registration_callback(callback_wrapper, buffer);
}

const InputArgs &Runtime::get_input_args(void) {
  return Legion::Runtime::get_input_args();
}
LayoutConstraintID Runtime::register_layout(const LayoutConstraintRegistrar &registrar) {
  return lrt->register_layout(registrar);
}

void Runtime::release_layout(LayoutConstraintID layout_id) {
  lrt->release_layout(layout_id);
}

LayoutConstraintID Runtime::preregister_layout(const LayoutConstraintRegistrar &registrar,
                                               LayoutConstraintID layout_id) {
  return Legion::Runtime::preregister_layout(registrar, layout_id);
}

TaskID Runtime::generate_dynamic_task_id(void) {
  return lrt->Runtime::generate_dynamic_task_id();
}

TaskID Runtime::generate_library_task_ids(const char *name, size_t count) {
  return lrt->generate_library_task_ids(name, count);
}

TaskID Runtime::generate_static_task_id(void) {
  return Legion::Runtime::generate_static_task_id();
}

VariantID Runtime::register_task_variant(const TaskVariantRegistrar &registrar,
                                         const CodeDescriptor &codedesc,
                                         const void *user_data, size_t user_len,
                                         size_t return_type_size, VariantID vid,
                                         bool has_return_type_size) {
  return lrt->register_task_variant(registrar, codedesc, user_data, user_len,
                                    return_type_size, vid, has_return_type_size);
}

VariantID Runtime::preregister_task_variant(
    const TaskVariantRegistrar &registrar, const CodeDescriptor &codedesc,
    const void *user_data, size_t user_len, const char *task_name, VariantID vid,
    size_t return_type_size, bool has_return_type_size, bool check_task_id) {
  return Legion::Runtime::preregister_task_variant(
      registrar, codedesc, user_data, user_len, task_name, vid, return_type_size,
      has_return_type_size, check_task_id);
}

void Runtime::legion_task_preamble(const void *data, size_t datalen, Processor p,
                                   const Task *&task,
                                   const std::vector<PhysicalRegion> *&reg, Context &ctx,
                                   Runtime *&runtime) {
  Legion::Runtime *runtime_;
  Legion::Runtime::legion_task_preamble(data, datalen, p, task, reg, ctx, runtime_);
  runtime = new Runtime(runtime_);
}

void Runtime::legion_task_postamble(Runtime *runtime, Context ctx, const void *retvalptr,
                                    size_t retvalsize, bool owned,
                                    Realm::RegionInstance inst, const void *metadataptr,
                                    size_t metadatasize) {
  Legion::Runtime::legion_task_postamble(ctx, retvalptr, retvalsize, owned, inst,
                                         metadataptr, metadatasize);
  delete runtime;
}

ShardID Runtime::get_shard_id(Context ctx, bool I_know_what_I_am_doing) {
  return lrt->get_shard_id(ctx, I_know_what_I_am_doing);
}

size_t Runtime::get_num_shards(Context ctx, bool I_know_what_I_am_doing) {
  return lrt->get_num_shards(ctx, I_know_what_I_am_doing);
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

    resilient_tag_t tag = ipartition_tags.at(partition.get_index_partition());
    auto &ip_state = state.ipartition_state.at(tag);
    if (ip_state.destroyed) {
      // If this is a destroyed partition, can't save it.
      continue;
    }

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
  if (!covering_set.regions.empty()) {
    log_resilience.warning()
        << "Computed a covering set for logical region " << r
        << "that includes leaf regions. This may indicate that there are no "
           "complete-disjoint partitions of the region.";
  }
  for (auto &partition : covering_set.partitions) {
    log_resilience.info() << "  Partition: " << partition;
  }
}

Path Runtime::compute_region_path(LogicalRegion lr, LogicalRegion parent) {
  if (lr == parent) {
    return Path();
  }

  assert(lrt->has_parent_logical_partition(lr));
  LogicalPartition lp = lrt->get_parent_logical_partition(lr);
  DomainPoint color = lrt->get_logical_region_color_point(lr);

  return Path(this, lp.get_index_partition(), color);
}

Path Runtime::compute_partition_path(LogicalPartition lp) {
  return Path(this, lp.get_index_partition());
}

LogicalRegion Runtime::lookup_region_path(LogicalRegion root, const Path &path) {
  if (!path.partition) {
    return root;
  }

  IndexPartition ip = ipartitions.at(path.partition_tag);
  LogicalPartition lp = lrt->get_logical_partition(root, ip);
  assert(!path.subregion_color.is_null());

  return lrt->get_logical_subregion_by_color(lp, path.subregion_color);
}

LogicalPartition Runtime::lookup_partition_path(LogicalRegion root, const Path &path) {
  assert(path.partition);
  IndexPartition ip = ipartitions.at(path.partition_tag);
  return lrt->get_logical_partition(root, ip);
}

void Runtime::restore_region(Context ctx, LogicalRegion lr, LogicalRegion parent,
                             LogicalRegion cpy, const std::vector<FieldID> &fids,
                             resilient_tag_t tag, const PathSerializer &path) {
  // FIXME (Elliott): can't restrict: https://github.com/StanfordLegion/legion/issues/1427
  AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, cpy, false /*restricted*/,
                    false /*mapped*/);

  std::string file_name;
  {
    std::stringstream ss;
    ss << config_prefix << "checkpoint." << checkpoint_tag << ".lr." << tag << "." << path
       << ".dat";
    file_name = ss.str();
  }

  log_resilience.info() << "restore_region: file_name " << file_name;
  al.attach_file(file_name.c_str(), fids, LEGION_FILE_READ_ONLY);

  PhysicalRegion pr = lrt->attach_external_resource(ctx, al);

  CopyLauncher cl(Predicate::TRUE_PRED, resilient_mapper_id);
  cl.add_copy_requirements(RegionRequirement(cpy, READ_ONLY, EXCLUSIVE, cpy),
                           RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));

  for (auto &fid : fids) {
    cl.add_src_field(0, fid);
    cl.add_dst_field(0, fid);
  }

  lrt->issue_copy_operation(ctx, cl);
  lrt->detach_external_resource(ctx, pr);
}

void Runtime::restore_partition(Context ctx, LogicalPartition lp, LogicalRegion parent,
                                LogicalRegion cpy, const std::vector<FieldID> &fids,
                                resilient_tag_t tag, const PathSerializer &path) {
  IndexPartition ip = lp.get_index_partition();
  LogicalPartition cpy_lp = lrt->get_logical_partition(cpy, ip);

  IndexAttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, true /*restricted*/);

  Domain domain = lrt->get_index_partition_color_space(ip);

  ShardID shard = lrt->get_shard_id(ctx, true);
  size_t num_shards = lrt->get_num_shards(ctx, true);
  size_t num_points = domain.get_volume();
  size_t point_start = num_points * shard / num_shards;
  size_t point_stop = num_points * (shard + 1) / num_shards;
  size_t point_idx = 0;

  log_resilience.info() << "restore_partition: shard " << shard << " num " << num_shards
                        << " point " << point_idx << " start " << point_start << " stop "
                        << point_stop;

  // Doing this in two steps so we don't invalidate file_names while iterating.
  std::vector<DomainPoint> points;
  std::vector<std::string> file_names;
  for (Domain::DomainPointIterator i(domain); i; ++i) {
    if (point_idx < point_start || point_idx >= point_stop) {
      point_idx++;
      continue;
    }

    std::stringstream ss;
    DomainPointSerializer dps(*i);
    ss << config_prefix << "checkpoint." << checkpoint_tag << ".lp." << tag << "." << path
       << "_" << dps << ".dat";
    points.push_back(*i);
    file_names.emplace_back(ss.str());

    point_idx++;
  }

  for (size_t idx = 0; idx < points.size(); ++idx) {
    DomainPoint &p = points.at(idx);
    std::string &file_name = file_names.at(idx);

    log_resilience.info() << "restore_partition: lp " << lp << " subregion color " << p
                          << " file_name " << file_name;

    LogicalRegion cpy_subregion = lrt->get_logical_subregion_by_color(cpy_lp, p);
    al.attach_file(cpy_subregion, file_name.c_str(), fids, LEGION_FILE_READ_ONLY);
  }
  al.privilege_fields.insert(fids.begin(), fids.end());

  ExternalResources res = lrt->attach_external_resources(ctx, al);

  IndexCopyLauncher cl(domain, Predicate::TRUE_PRED, resilient_mapper_id);
  constexpr ProjectionID identity = 0;
  cl.add_copy_requirements(
      RegionRequirement(cpy_lp, identity, READ_ONLY, EXCLUSIVE, cpy),
      RegionRequirement(lp, identity, WRITE_DISCARD, EXCLUSIVE, parent));

  for (auto &fid : fids) {
    cl.add_src_field(0, fid);
    cl.add_dst_field(0, fid);
  }

  lrt->issue_copy_operation(ctx, cl);
  lrt->detach_external_resources(ctx, res);
}

void Runtime::restore_region_content(Context ctx, LogicalRegion lr) {
  resilient_tag_t tag = region_tags.at(lr);
  auto &lr_state = state.region_state.at(tag);
  const SavedSet &saved_set = lr_state.saved_set;

  log_resilience.info() << "restore_region_content: region from checkpoint, tag " << tag;
  LogicalRegion cpy = lrt->create_logical_region(
      ctx, lr.get_index_space(), lr.get_field_space(), false, FILE_AND_LINE);

  std::vector<FieldID> fids;
  lrt->get_field_space_fields(lr.get_field_space(), fids);

  for (auto &path_ser : saved_set.partitions) {
    Path path(path_ser);
    LogicalPartition partition = lookup_partition_path(lr, path);
    restore_partition(ctx, partition, lr, cpy, fids, tag, path_ser);
  }

  for (auto &path_ser : saved_set.regions) {
    Path path(path_ser);
    LogicalRegion subregion = lookup_region_path(lr, path);
    restore_region(ctx, subregion, lr, cpy, fids, tag, path_ser);
  }

  lrt->destroy_logical_region(ctx, cpy, false, FILE_AND_LINE);
}

void Runtime::save_region(Context ctx, LogicalRegion lr, LogicalRegion parent,
                          LogicalRegion cpy, const std::vector<FieldID> &fids,
                          resilient_tag_t tag, const PathSerializer &path,
                          Predicate pred) {
  std::string file_name;
  {
    std::stringstream ss;
    ss << config_prefix << "checkpoint." << checkpoint_tag << ".lr." << tag << "." << path
       << ".dat";
    file_name = ss.str();
  }

  log_resilience.info() << "save_region: lr " << lr << " file_name " << file_name;

  LogicalRegion cpy_lr = lrt->get_logical_subregion_by_tree(
      lr.get_index_space(), cpy.get_field_space(), cpy.get_tree_id());

  // FIXME (Elliott): can't restrict: https://github.com/StanfordLegion/legion/issues/1427
  // FIXME (Elliott): can't predicate AttachLauncher
  // https://github.com/StanfordLegion/legion/issues/1434
  AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy_lr, cpy, false /*restricted*/,
                    false /*mapped*/);
  al.attach_file(file_name.c_str(), fids, LEGION_FILE_CREATE);

  PhysicalRegion pr = lrt->attach_external_resource(ctx, al);

  CopyLauncher cl(pred, resilient_mapper_id);
  cl.add_copy_requirements(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, parent),
                           RegionRequirement(cpy_lr, WRITE_DISCARD, EXCLUSIVE, cpy));

  for (auto &fid : fids) {
    cl.add_src_field(0, fid);
    cl.add_dst_field(0, fid);
  }

  lrt->issue_copy_operation(ctx, cl);
  lrt->detach_external_resource(ctx, pr);
}

void Runtime::save_partition(Context ctx, LogicalPartition lp, LogicalRegion parent,
                             LogicalRegion cpy, const std::vector<FieldID> &fids,
                             resilient_tag_t tag, const PathSerializer &path,
                             Predicate pred) {
  IndexPartition ip = lp.get_index_partition();
  LogicalPartition cpy_lp = lrt->get_logical_partition(cpy, ip);

  // FIXME (Elliott): can't predicate AttachLauncher
  // https://github.com/StanfordLegion/legion/issues/1434
  IndexAttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, true /*restricted*/);
  Domain domain = lrt->get_index_partition_color_space(ip);

  ShardID shard = lrt->get_shard_id(ctx, true);
  size_t num_shards = lrt->get_num_shards(ctx, true);
  size_t num_points = domain.get_volume();
  size_t point_start = num_points * shard / num_shards;
  size_t point_stop = num_points * (shard + 1) / num_shards;
  size_t point_idx = 0;

  // Doing this in two steps so we don't invalidate file_names while iterating.
  std::vector<DomainPoint> points;
  std::vector<std::string> file_names;
  for (Domain::DomainPointIterator i(domain); i; ++i) {
    if (point_idx < point_start || point_idx >= point_stop) {
      point_idx++;
      continue;
    }

    std::stringstream ss;
    DomainPointSerializer dps(*i);
    ss << config_prefix << "checkpoint." << checkpoint_tag << ".lp." << tag << "." << path
       << "_" << dps << ".dat";
    points.push_back(*i);
    file_names.emplace_back(ss.str());

    point_idx++;
  }

  for (size_t idx = 0; idx < points.size(); ++idx) {
    DomainPoint &p = points.at(idx);
    std::string &file_name = file_names.at(idx);

    log_resilience.info() << "save_partition: lp " << lp << " subregion color " << p
                          << " file_name " << file_name;

    LogicalRegion cpy_subregion = lrt->get_logical_subregion_by_color(cpy_lp, p);
    al.attach_file(cpy_subregion, file_name.c_str(), fids, LEGION_FILE_CREATE);
  }
  al.privilege_fields.insert(fids.begin(), fids.end());

  ExternalResources res = lrt->attach_external_resources(ctx, al);

  IndexCopyLauncher cl(domain, pred, resilient_mapper_id);
  constexpr ProjectionID identity = 0;
  cl.add_copy_requirements(
      RegionRequirement(lp, identity, READ_ONLY, EXCLUSIVE, parent),
      RegionRequirement(cpy_lp, identity, WRITE_DISCARD, EXCLUSIVE, cpy));

  for (auto &fid : fids) {
    cl.add_src_field(0, fid);
    cl.add_dst_field(0, fid);
  }

  lrt->issue_copy_operation(ctx, cl);
  lrt->detach_external_resources(ctx, res);
}

void Runtime::save_region_content(Context ctx, LogicalRegion lr, Predicate pred) {
  CoveringSet covering_set;
  compute_covering_set(lr, covering_set);

  resilient_tag_t tag = region_tags.at(lr);
  auto &lr_state = state.region_state.at(tag);
  SavedSet &saved_set = lr_state.saved_set;
  saved_set.partitions.clear();
  saved_set.regions.clear();

  LogicalRegion cpy = lrt->create_logical_region(
      ctx, lr.get_index_space(), lr.get_field_space(), false, FILE_AND_LINE);

  std::vector<FieldID> fids;
  lrt->get_field_space_fields(lr.get_field_space(), fids);

  for (auto &partition : covering_set.partitions) {
    Path path = compute_partition_path(partition);
    PathSerializer path_ser(path);
    save_partition(ctx, partition, lr, cpy, fids, tag, path_ser, pred);
    saved_set.partitions.emplace_back(path_ser);
  }

  for (auto &subregion : covering_set.regions) {
    Path path = compute_region_path(subregion, lr);
    PathSerializer path_ser(path);
    save_region(ctx, subregion, lr, cpy, fids, tag, path_ser, pred);
    saved_set.regions.emplace_back(path_ser);
  }

  lrt->destroy_logical_region(ctx, cpy, false, FILE_AND_LINE);
}

void Runtime::checkpoint(Context ctx, Predicate pred) {
  if (config_disable) return;

  if (!enabled) {
    log_resilience.error()
        << "error: must enable checkpointing with runtime->enable_checkpointing()";
    abort();
  }

  if (replay && checkpoint_tag == max_checkpoint_tag - 1) {
    // This is the checkpoint we originally saved. Restore all region data at this point.
    log_resilience.info() << "In checkpoint: restoring regions from tag "
                          << checkpoint_tag;

    // The predicate had better be true in this case
    if (pred != Predicate::TRUE_PRED &&
        !lrt->get_predicate_future(ctx, pred, FILE_AND_LINE).get_result<bool>()) {
      log_resilience.error()
          << "Attempting to restore from checkpoint that was predicated false";
      abort();
    }

    for (resilient_tag_t i = 0; i < regions.size(); ++i) {
      auto &lr = regions.at(i);
      auto &lr_state = state.region_state.at(i);
      if (!lr_state.destroyed) {
        restore_region_content(ctx, lr);
        auto &tree_state = region_tree_state.at(i);
        state.region_tree_state.at(i).inflate(this, lr, tree_state);
      }
    }

    if (config_measure_replay_time_and_exit) {
      lrt->issue_execution_fence(ctx);
      Legion::Future replay_stop = lrt->get_current_time_in_nanoseconds(ctx);
      long long start = replay_start.get_result<long long>();
      long long stop = replay_stop.get_result<long long>();
      double duration = (stop - start) / 1e9;
      lrt->log_once(ctx, log_resilience.print() << "Checkpoint replay finished in "
                                                << duration << " seconds");
      exit(0);
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

  unsigned long long start_time = Realm::Clock::current_time_in_nanoseconds();

  for (size_t i = 0; i < regions.size(); ++i) {
    auto &lr = regions.at(i);
    auto &lr_state = state.region_state.at(i);

    if (lr_state.destroyed) {
      continue;
    }
    save_region_content(ctx, lr, pred);
  }

  log_resilience.info() << "Saved all logical regions!";

  // Synchornize checkpoint state
  // Note: this is incremental!
  resilient_tag_t last_future_tag = state.max_future_tag;
  resilient_tag_t last_future_map_tag = state.max_future_map_tag;
  resilient_tag_t last_partition_tag = state.max_partition_tag;

  state.max_api_tag = api_tag;
  state.max_future_tag = future_tag;
  state.max_future_map_tag = future_map_tag;
  state.max_index_space_tag = index_space_tag;
  state.max_region_tag = region_tag;
  state.max_partition_tag = partition_tag;
  state.max_checkpoint_tag = checkpoint_tag + 1;
  state.num_shards = lrt->get_num_shards(ctx, true);

  // FIXME (Elliott): there is a linear time algorithm for this based on walking both
  // iterators at once
  for (auto it = state.futures.begin(); it != state.futures.end();) {
    if (futures.count(it->first) == 0) {
      state.futures.erase(it++);
    } else {
      ++it;
    }
  }
  // After deleting dead futures, we may find that we can further compress the remaining.
  for (auto it = state.futures.begin(); it != state.futures.end();) {
    if (it != state.futures.begin()) {
      auto last_it = std::prev(it);
      if (it->second == last_it->second) {
        state.futures.erase(it++);
        continue;
      }
    }
    ++it;
  }
  for (auto it = futures.lower_bound(last_future_tag); it != futures.end();) {
    // Run-length encode futures by only inserting futures when we see a new value.
    FutureSerializer fs(it->second);
    auto rit = state.futures.rbegin();
    if (rit == state.futures.rend() || fs != rit->second) {
      state.futures[it->first] = fs;
      ++it;
    } else {
      // We're going to reuse the old value, so move all the state over to that
      // entry. Note: because we can't make references to the new (to-be-deleted) value go
      // away, we double-track the state (on old and new entries). We only actually delete
      // when its references go to zero.
      bool need_delete;
      {
        auto &last = futures.at(rit->first);
        auto &last_state = future_state.at(last);
        auto &state = future_state.at(it->second);
        last_state.escaped = last_state.escaped || state.escaped;
        need_delete = state.ref_count == 1;
        // No point doing this if we're about to delete it anyway
        if (!need_delete) {
          last_state.ref_count += state.ref_count;
          state.moved_to = rit->first;
        }
      }
      if (need_delete) {
        futures.erase(it++);
      } else {
        ++it;
      }
    }
  }

  // FIXME (Elliott): there is a linear time algorithm for this based on walking both
  // iterators at once
  for (auto it = sharded_state.future_maps.begin();
       it != sharded_state.future_maps.end();) {
    if (future_maps.count(it->first) == 0) {
      sharded_state.future_maps.erase(it++);
    } else {
      ++it;
    }
  }
  for (auto it = future_maps.lower_bound(last_future_map_tag); it != future_maps.end();
       ++it) {
    sharded_state.future_maps[it->first] = FutureMapSerializer(this, ctx, it->second);
  }

  // Index spaces have already been handled eagerly, see register_index_space

  for (auto it = sharded_state.ipartitions.begin();
       it != sharded_state.ipartitions.end();) {
    auto &ip_state = state.ipartition_state.at(it->first);
    if (ip_state.destroyed) {
      sharded_state.ipartitions.erase(it++);
    } else {
      ++it;
    }
  }
  // Partition table does not include deleted entries, but do the best we can to avoid
  // useless work.
  for (resilient_tag_t i = last_partition_tag; i < ipartitions.size(); ++i) {
    auto &ip_state = state.ipartition_state.at(i);
    if (ip_state.destroyed) continue;

    auto &ip = ipartitions.at(i);
    Domain color_space = lrt->get_index_partition_color_space(ip);
    sharded_state.ipartitions[i] = IndexPartitionSerializer(this, ctx, ip, color_space);
  }

  // Unlike the others, this state is mutable, so we need to recreate it in full.
  state.region_tree_state.clear();
  for (size_t i = 0; i < regions.size(); ++i) {
    auto &lr_state = state.region_state.at(i);
    if (lr_state.destroyed) {
      state.region_tree_state.emplace_back();
      continue;
    }

    auto &lr = regions.at(i);
    auto &tree_state = region_tree_state.at(i);
    state.region_tree_state.push_back(RegionTreeStateSerializer(this, lr, tree_state));
  }

  // Sanity checks
  {
    auto rit = state.futures.rbegin();
    if (rit != state.futures.rend()) {
      assert(rit->first < state.max_future_tag);
    }
  }
  {
    auto rit = sharded_state.future_maps.rbegin();
    if (rit != sharded_state.future_maps.rend()) {
      assert(rit->first < state.max_future_map_tag);
    }
  }
  assert(state.max_region_tag == state.region_state.size());
  assert(state.max_index_space_tag == state.ispaces.size());
  assert(state.max_partition_tag == state.ipartition_state.size());
  assert(state.max_checkpoint_tag == checkpoint_tag + 1);

  // Serialize state
  size_t serialized_data_size;
  {
    std::string serialized_data;
    {
      std::stringstream ss;
      {
#ifdef DEBUG_LEGION
        cereal::XMLOutputArchive oarchive(ss);
#else
        cereal::BinaryOutputArchive oarchive(ss);
#endif
        oarchive(CEREAL_NVP(state));
      }
      serialized_data = ss.str();
    }
    serialized_data_size = serialized_data.size();

    std::string file_name;
    {
      std::stringstream ss;
      ss << config_prefix << "checkpoint." << checkpoint_tag << ".dat";
      file_name = ss.str();
    }

    Legion::Future file_name_f =
        Legion::Future::from_untyped_pointer(lrt, file_name.data(), file_name.size());
    Legion::Future serialized_data_f = Legion::Future::from_untyped_pointer(
        lrt, serialized_data.data(), serialized_data.size());

    Legion::TaskLauncher launcher(write_checkpoint_task_id, TaskArgument(), pred,
                                  resilient_mapper_id, 0, UntypedBuffer(), FILE_AND_LINE);
    launcher.add_future(file_name_f);
    launcher.add_future(serialized_data_f);
    lrt->execute_task(ctx, launcher);
  }

  // Serialize sharded_state
  size_t sharded_serialized_data_size;
  {
    std::string serialized_data;
    {
      std::stringstream ss;
      {
#ifdef DEBUG_LEGION
        cereal::XMLOutputArchive oarchive(ss);
#else
        cereal::BinaryOutputArchive oarchive(ss);
#endif
        oarchive(CEREAL_NVP(sharded_state));
      }
      serialized_data = ss.str();
    }
    sharded_serialized_data_size = serialized_data.size();

    ShardID shard = lrt->get_shard_id(ctx, true);

    std::string file_name;
    {
      std::stringstream ss;
      ss << config_prefix << "checkpoint." << checkpoint_tag << ".shard." << shard
         << ".dat";
      file_name = ss.str();
    }

    std::map<DomainPoint, UntypedBuffer> file_name_map;
    file_name_map[shard] = UntypedBuffer(file_name.data(), file_name.size());

    std::map<DomainPoint, UntypedBuffer> serialized_data_map;
    serialized_data_map[shard] =
        UntypedBuffer(serialized_data.data(), serialized_data.size());

    Legion::FutureMap file_name_fm =
        lrt->construct_future_map(ctx, shard_space, file_name_map, true /*collective*/,
                                  0 /*sid*/, false /*implicit_sharding*/, FILE_AND_LINE);

    Legion::FutureMap serialized_data_fm = lrt->construct_future_map(
        ctx, shard_space, serialized_data_map, true /*collective*/, 0 /*sid*/,
        false /*implicit_sharding*/, FILE_AND_LINE);

    Legion::IndexTaskLauncher launcher(
        write_checkpoint_task_id, shard_space, TaskArgument(), ArgumentMap(), pred,
        false /*must*/, resilient_mapper_id, 0, UntypedBuffer(), FILE_AND_LINE);
    launcher.point_futures.push_back(file_name_fm);
    launcher.point_futures.push_back(serialized_data_fm);
    lrt->execute_index_space(ctx, launcher);
  }

  unsigned long long stop_time = Realm::Clock::current_time_in_nanoseconds();
  double elapsed_time = (stop_time - start_time) / 1e9;

  size_t maxrss = 0;  // kilobytes (= 1024 bytes)
  {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
      maxrss = usage.ru_maxrss;
    }
  }

  log_resilience.print() << "Serialized checkpoint " << checkpoint_tag << " in "
                         << elapsed_time << " seconds (primary: " << serialized_data_size
                         << " bytes, " << state.futures.size() << " futures, "
                         << state.region_state.size() << " regions, "
                         << state.ispaces.size() << " ispaces), "
                         << " (sharded: " << sharded_serialized_data_size << " bytes, "
                         << sharded_state.future_maps.size() << " future_maps, "
                         << state.ipartition_state.size()
                         << " ipartitions), RSS = " << maxrss << " KiB";

  checkpoint_tag++;
}

void Runtime::auto_checkpoint(Context ctx, Predicate pred) {
  if (config_disable) return;

  if (!enabled) {
    log_resilience.error()
        << "error: must enable checkpointing with runtime->enable_checkpointing()";
    abort();
  }

  if (config_auto_steps <= 0 || auto_step >= auto_checkpoint_step + config_auto_steps) {
    log_resilience.info() << "auto_checkpoint: triggering checkpoint (step " << auto_step
                          << " checkpoint " << auto_checkpoint_step << " config_steps "
                          << config_auto_steps << ")";
    // FIXME (Elliott): this assumes pred is monotonic, i.e., it will always be true
    // followed by false (and not return to true again)
    checkpoint(ctx, pred);
    auto_checkpoint_step = auto_step;
  }

  auto_step++;
}

void Runtime::enable_checkpointing(Context ctx) {
  if (config_disable) return;

  bool first_time = !enabled;
  enabled = true;
  if (!first_time) return;

  if (config_measure_replay_time_and_exit) {
    replay_start = lrt->get_current_time_in_nanoseconds(ctx);
  }

  shard_space = lrt->create_index_space(
      ctx, Rect<1>(0, lrt->get_num_shards(ctx, true) - 1), FILE_AND_LINE);

  // These values get parsed in Runtime::start
  replay = config_replay;
  resilient_tag_t load_checkpoint_tag = config_checkpoint_tag;

  log_resilience.info() << "In enable_checkpointing: replay " << replay
                        << " load_checkpoint_tag " << load_checkpoint_tag;

  if (replay) {
    // Read primary checkpoint
    // Use a task so we do this I/O only once
    std::string serialized_data;
    {
      std::string file_name;
      {
        std::stringstream ss;
        ss << config_prefix << "checkpoint." << load_checkpoint_tag << ".dat";
        file_name = ss.str();
      }

      Legion::Future file_name_f =
          Legion::Future::from_untyped_pointer(lrt, file_name.data(), file_name.size());

      Legion::TaskLauncher launcher(read_checkpoint_task_id, TaskArgument(),
                                    Predicate::TRUE_PRED, resilient_mapper_id, 0,
                                    UntypedBuffer(), FILE_AND_LINE);
      launcher.add_future(file_name_f);
      Legion::Future f = lrt->execute_task(ctx, launcher);
      serialized_data = std::string((const char *)f.get_untyped_pointer(true /*silent*/),
                                    f.get_untyped_size());
    }

    {
      std::stringstream ss(serialized_data);
#ifdef DEBUG_LEGION
      cereal::XMLInputArchive iarchive(ss);
#else
      cereal::BinaryInputArchive iarchive(ss);
#endif
      iarchive(CEREAL_NVP(state));
    }

    // Currently we can only replay on the same number of shards
    assert(state.num_shards == lrt->get_num_shards(ctx, true));

    // Read shard checkpoint
    // Don't bother with a task, we're going to block on the result anyway
    {
      ShardID shard = lrt->get_shard_id(ctx, true);

      std::string file_name;
      {
        std::stringstream ss;
        ss << config_prefix << "checkpoint." << load_checkpoint_tag << ".shard." << shard
           << ".dat";
        file_name = ss.str();
      }

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
      iarchive(CEREAL_NVP(sharded_state));
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
    assert(state.max_region_tag == state.region_state.size());
    assert(state.max_index_space_tag == state.ispaces.size());
    assert(state.max_partition_tag == state.ipartition_state.size());
    assert(state.max_checkpoint_tag == load_checkpoint_tag + 1);

    // Restore state
    // Note: most of this happens lazily
    max_api_tag = state.max_api_tag;
    max_future_tag = state.max_future_tag;
    max_future_map_tag = state.max_future_map_tag;
    max_index_space_tag = state.max_index_space_tag;
    max_region_tag = state.max_region_tag;
    max_partition_tag = state.max_partition_tag;
    max_checkpoint_tag = state.max_checkpoint_tag;
  }
}

void Runtime::allow_unsafe_inline_mapping(bool I_know_what_I_am_doing) {
  if (!I_know_what_I_am_doing) {
    log_resilience.error() << "allow_unsafe_inline_mapping is unsafe and should be "
                              "avoided unless you know what you're doing";
    abort();
  }

  allow_inline_mapping = true;
}

bool Runtime::is_replaying_checkpoint(bool I_know_what_I_am_doing) {
  if (!I_know_what_I_am_doing) {
    log_resilience.error() << "is_replaying_checkpoint is unsafe and should be avoided "
                              "unless you know what you're doing";
    abort();
  }

  return replay && checkpoint_tag < max_checkpoint_tag;
}
