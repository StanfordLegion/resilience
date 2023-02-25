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

#ifndef RESILIENCE_SERIALIZER_H
#define RESILIENCE_SERIALIZER_H

#include "legion.h"
#include "resilience/future.h"
#include "resilience/types.h"

namespace ResilientLegion {

class Runtime;
class RegionTreeState;

typedef size_t resilient_tag_t;

class DomainPointSerializer {
public:
  DomainPoint p;

  DomainPointSerializer() = default;
  DomainPointSerializer(DomainPoint p_) : p(p_) {}

  operator DomainPoint() const { return p; }

  bool operator<(const DomainPointSerializer &o) const { return p < o.p; }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(p.dim), CEREAL_NVP(p.point_data));
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
    ar(CEREAL_NVP(lo), CEREAL_NVP(hi));
  }
};

class DomainSerializer {
public:
  int dim;
  std::vector<DomainRectSerializer> rects;

  DomainSerializer() = default;

  DomainSerializer(Domain domain) {
    dim = domain.get_dim();

    if (!domain.is_valid()) return;

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
  void add_rect(DomainPoint lo_, DomainPoint hi_) { rects.emplace_back(lo_, hi_); }

public:
  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(dim), CEREAL_NVP(rects));
  }
};

class IndexSpaceSerializer {
public:
  DomainSerializer domain;

  IndexSpaceSerializer() = default;
  IndexSpaceSerializer(Domain d) : domain(d) {}

  IndexSpace inflate(Runtime *runtime, Context ctx, const char *provenance) const;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(domain));
  }
};

class FutureSerializer {
public:
  std::vector<uint8_t> buffer;

  FutureSerializer() = default;
  FutureSerializer(const Future &f) : FutureSerializer(f.lft) {}
  FutureSerializer(const Legion::Future &f) {
    const uint8_t *ptr = static_cast<const uint8_t *>(f.get_untyped_pointer());
    size_t size = f.get_untyped_size();
    std::vector<uint8_t>(ptr, ptr + size).swap(buffer);
  }

  Future inflate(Runtime *runtime) const {
    return Future(runtime,
                  Legion::Future::from_untyped_pointer(buffer.data(), buffer.size()));
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(buffer));
  }
};

class FutureMapSerializer {
public:
  IndexSpaceSerializer domain;
  std::map<DomainPointSerializer, FutureSerializer> map;

  FutureMapSerializer() = default;
  FutureMapSerializer(const FutureMap &fm) : domain(fm.domain) {
    for (Domain::DomainPointIterator i(fm.domain); i; ++i) {
      map[*i] = fm.lfm.get_future(*i);
    }
  }

  FutureMap inflate(Runtime *runtime, Context ctx) const;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(domain), CEREAL_NVP(map));
  }
};

class IndexPartitionSerializer {
public:
  IndexSpaceSerializer color_space;
  std::map<DomainPointSerializer, IndexSpaceSerializer> subspaces;
  PartitionKind kind;

  IndexPartitionSerializer() = default;
  IndexPartitionSerializer(Runtime *runtime, IndexPartition ip, Domain color_space_);

  IndexPartition inflate(Runtime *runtime, Context ctx, IndexSpace parent,
                         IndexSpace color_space, Color color,
                         const char *provenance) const;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(color_space), CEREAL_NVP(subspaces), CEREAL_NVP(kind));
  }
};

class IndexPartitionState {
public:
  bool destroyed;

  IndexPartitionState() : destroyed(false) {}

  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(destroyed));
  }
};

class Path {
public:
  bool partition, subregion;
  resilient_tag_t partition_tag;  // Set if partition
  DomainPoint subregion_color;    // Set if partition && subregion

  // Root region
  Path()
      : partition(false),
        subregion(false),
        partition_tag(0),
        subregion_color(DomainPoint::nil()) {}

  // Partition
  Path(Runtime *runtime, IndexPartition partition);

  // Subregion
  Path(Runtime *runtime, IndexPartition partition, const DomainPoint &subregion_color);

  Path(bool partition_, bool subregion_, resilient_tag_t partition_tag_,
       const DomainPoint &subregion_color_)
      : partition(partition_),
        subregion(subregion_),
        partition_tag(partition_tag_),
        subregion_color(subregion_color_) {}
};

class PathSerializer {
public:
  bool partition, subregion;
  resilient_tag_t partition_tag;          // Set if partition
  DomainPointSerializer subregion_color;  // Set if partition && subregion

  PathSerializer() = default;

  PathSerializer(const Path &path)
      : partition(path.partition),
        subregion(path.subregion),
        partition_tag(path.partition_tag),
        subregion_color(path.subregion_color) {}

  operator Path() const {
    return Path(partition, subregion, partition_tag, DomainPoint(subregion_color));
  }

  bool operator<(const PathSerializer &path) const {
    if (partition < path.partition) return true;
    if (path.partition < partition) return false;
    if (subregion < path.subregion) return true;
    if (path.subregion < subregion) return false;
    if (partition_tag < path.partition_tag) return true;
    if (path.partition_tag < partition_tag) return false;
    if (subregion_color < path.subregion_color) return true;
    return false;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(partition), CEREAL_NVP(subregion), CEREAL_NVP(partition_tag),
       CEREAL_NVP(subregion_color));
  }
};

class SavedSet {
public:
  std::vector<PathSerializer> partitions;
  std::vector<PathSerializer> regions;

  SavedSet() = default;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(partitions), CEREAL_NVP(regions));
  }
};

class LogicalRegionState {
public:
  bool destroyed;
  SavedSet saved_set;

  LogicalRegionState() : destroyed(false) {}

  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(destroyed), CEREAL_NVP(saved_set));
  }
};

class RegionTreeStateSerializer {
public:
  std::map<PathSerializer, PathSerializer> recent_partitions;

  RegionTreeStateSerializer() = default;
  RegionTreeStateSerializer(Runtime *runtime, LogicalRegion parent,
                            const RegionTreeState &state);

  void inflate(Runtime *runtime, LogicalRegion parent, RegionTreeState &state) const;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(recent_partitions));
  }
};

class CheckpointState {
public:
  std::map<resilient_tag_t, FutureSerializer> futures;
  std::map<resilient_tag_t, FutureMapSerializer> future_maps;
  std::vector<IndexSpaceSerializer> ispaces;
  std::map<resilient_tag_t, IndexPartitionSerializer> ipartitions;
  std::vector<IndexPartitionState> ipartition_state;
  std::vector<LogicalRegionState> region_state;
  std::vector<RegionTreeStateSerializer> region_tree_state;
  resilient_tag_t max_api_tag, max_future_tag, max_future_map_tag, max_index_space_tag,
      max_region_tag, max_partition_tag, max_checkpoint_tag;

  CheckpointState()
      : max_api_tag(0),
        max_future_tag(0),
        max_future_map_tag(0),
        max_index_space_tag(0),
        max_region_tag(0),
        max_partition_tag(0),
        max_checkpoint_tag(0) {}

  template <class Archive>
  void serialize(Archive &ar) {
    ar(CEREAL_NVP(futures), CEREAL_NVP(future_maps), CEREAL_NVP(ispaces),
       CEREAL_NVP(ipartitions), CEREAL_NVP(ipartition_state), CEREAL_NVP(region_state),
       CEREAL_NVP(region_tree_state), CEREAL_NVP(max_api_tag), CEREAL_NVP(max_future_tag),
       CEREAL_NVP(max_future_map_tag), CEREAL_NVP(max_region_tag),
       CEREAL_NVP(max_index_space_tag), CEREAL_NVP(max_partition_tag),
       CEREAL_NVP(max_checkpoint_tag));
  }
};

}  // namespace ResilientLegion

#endif  // RESILIENCE_SERIALIZER_H
