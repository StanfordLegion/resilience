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
  IndexSpace is = domain.inflate(runtime, ctx);
  std::map<DomainPoint, UntypedBuffer> data;
  for (auto &point : map) {
    auto &buffer = point.second.buffer;
    data[point.first] = UntypedBuffer(buffer.data(), buffer.size());
  }

  Domain d = runtime->lrt->get_index_space_domain(is);
  return FutureMap(d, runtime->lrt->construct_future_map(ctx, is, data));
}
