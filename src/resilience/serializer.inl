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

inline Path::Path(Runtime* runtime, IndexPartition partition_)
    : partition(true),
      subregion(false),
      partition_tag(runtime->ipartition_tags.at(partition_)),
      subregion_color(DomainPoint::nil()) {}

inline Path::Path(Runtime* runtime, IndexPartition partition_,
                  const DomainPoint& subregion_color_)
    : partition(true),
      subregion(true),
      partition_tag(runtime->ipartition_tags.at(partition_)),
      subregion_color(subregion_color_) {}

inline std::ostream& operator<<(std::ostream& os, const DomainPointSerializer& dps) {
  const DomainPoint& dp = dps.p;
  switch (dp.dim) {
    case 0: {
      os << dp.point_data[0];
      break;
    }
    case 1: {
      os << dp.point_data[0];
      break;
    }
#if LEGION_MAX_DIM >= 2
    case 2: {
      os << dp.point_data[0] << '-' << dp.point_data[1];
      break;
    }
#endif
#if LEGION_MAX_DIM >= 3
    case 3: {
      os << dp.point_data[0] << '-' << dp.point_data[1] << '-' << dp.point_data[2];
      break;
    }
#endif
#if LEGION_MAX_DIM >= 4
    case 4: {
      os << dp.point_data[0] << '-' << dp.point_data[1] << '-' << dp.point_data[2] << '-'
         << dp.point_data[3];
      break;
    }
#endif
#if LEGION_MAX_DIM >= 5
    case 5: {
      os << dp.point_data[0] << '-' << dp.point_data[1] << '-' << dp.point_data[2] << '-'
         << dp.point_data[3] << '-' << dp.point_data[4];
      break;
    }
#endif
#if LEGION_MAX_DIM >= 6
    case 6: {
      os << dp.point_data[0] << '-' << dp.point_data[1] << '-' << dp.point_data[2] << '-'
         << dp.point_data[3] << '-' << dp.point_data[4] << '-' << dp.point_data[5];
      break;
    }
#endif
#if LEGION_MAX_DIM >= 7
    case 7: {
      os << dp.point_data[0] << '-' << dp.point_data[1] << '-' << dp.point_data[2] << '-'
         << dp.point_data[3] << '-' << dp.point_data[4] << '-' << dp.point_data[5] << '-'
         << dp.point_data[6];
      break;
    }
#endif
#if LEGION_MAX_DIM >= 8
    case 8: {
      os << dp.point_data[0] << '-' << dp.point_data[1] << '-' << dp.point_data[2] << '-'
         << dp.point_data[3] << '-' << dp.point_data[4] << '-' << dp.point_data[5] << '-'
         << dp.point_data[6] << '-' << dp.point_data[7];
      break;
    }
#endif
#if LEGION_MAX_DIM >= 9
    case 9: {
      os << dp.point_data[0] << '-' << dp.point_data[1] << '-' << dp.point_data[2] << '-'
         << dp.point_data[3] << '-' << dp.point_data[4] << '-' << dp.point_data[5] << '-'
         << dp.point_data[6] << '-' << dp.point_data[7] << '-' << dp.point_data[8];
      break;
    }
#endif
    default:
      assert(0);
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Path& path) {
  if (path.partition) {
    os << path.partition_tag;
    if (path.subregion) {
      os << '_' << path.subregion_color;
    }
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const PathSerializer& path) {
  if (path.partition) {
    os << path.partition_tag;
    if (path.subregion) {
      os << '_' << path.subregion_color;
    }
  }
  return os;
}

}  // namespace ResilientLegion
