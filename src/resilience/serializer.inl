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

class IndexSpaceSerializer {
public:
  DomainSerializer domain;

  IndexSpaceSerializer() = default;
  IndexSpaceSerializer(Domain d) : domain(d) {}

  IndexSpace inflate(Runtime *runtime, Context ctx) const;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(domain);
  }
};

class FutureSerializer {
public:
  std::vector<uint8_t> buffer;

  FutureSerializer() = default;
  FutureSerializer(const Future &f) {
    const uint8_t *ptr = static_cast<const uint8_t *>(f.get_untyped_pointer());
    size_t size = f.get_untyped_size();
    std::vector<uint8_t>(ptr, ptr + size).swap(buffer);
  }

  operator Future() const {
    return Future(Legion::Future::from_untyped_pointer(buffer.data(), buffer.size()));
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(buffer);
  }
};

class FutureMapSerializer {
public:
  IndexSpaceSerializer domain;
  std::map<DomainPointSerializer, FutureSerializer> map;

  FutureMapSerializer() = default;
  FutureMapSerializer(const FutureMap &fm) : domain(fm.domain) {
    for (Domain::DomainPointIterator i(fm.domain); i; ++i) {
      map[*i] = fm.get_future(*i);
    }
  }

  FutureMap inflate(Runtime *runtime, Context ctx) const;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(domain, map);
  }
};

}  // namespace ResilientLegion
