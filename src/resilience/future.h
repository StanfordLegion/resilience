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

#ifndef RESILIENCE_FUTURE_H
#define RESILIENCE_FUTURE_H

#include "legion.h"
#include "resilience/types.h"

namespace ResilientLegion {

class FutureMap;
class FutureMapSerializer;
class Runtime;

class Future {
public:
  Legion::Future lft;

  Future() = default;
  Future(const Future &f) : lft(f.lft) {}

  template <class T>
  inline T get_result(bool silence_warnings = false,
                      const char *warning_string = NULL) const {
    return lft.get_result<T>(silence_warnings, warning_string);
  }

  void get_void_result(bool silence_warnings = false,
                       const char *warning_string = NULL) const {
    lft.get_void_result(silence_warnings, warning_string);
  }

  const void *get_untyped_pointer(bool silence_warnings = false,
                                  const char *warning_string = NULL) const {
    return lft.get_untyped_pointer(silence_warnings, warning_string);
  }

  size_t get_untyped_size(void) const { return lft.get_untyped_size(); }

  static Future from_untyped_pointer(Runtime *runtime, const void *buffer, size_t bytes);

  template <typename T>
  static Future from_value(Runtime *runtime, const T &value);

private:
  Future(const Legion::Future &lft_) : lft(lft_) {}

  // This is dangerous because we can't track the Future liveness after conversion
  operator Legion::Future() const { return lft; }

  friend class FutureMap;
  friend class FutureSerializer;
  friend class Runtime;
};

class FutureMap {
public:
  Legion::Domain domain;
  Legion::FutureMap lfm;

  FutureMap() = default;
  FutureMap(const FutureMap &fm) : domain(fm.domain), lfm(fm.lfm) {}

  template <typename T>
  T get_result(const DomainPoint &point, bool silence_warnings = false,
               const char *warning_string = NULL) const {
    return lfm.get_result<T>(point, silence_warnings, warning_string);
  }

  Future get_future(const DomainPoint &point) const {
    // FIXME (Elliott): Do we need to track this?
    return Future(lfm.get_future(point));
  }

  void get_void_result(const DomainPoint &point, bool silence_warnings = false,
                       const char *warning_string = NULL) const {
    lfm.get_void_result(point, silence_warnings, warning_string);
  }

  void wait_all_results(bool silence_warnings = false,
                        const char *warning_string = NULL) const {
    lfm.wait_all_results(silence_warnings, warning_string);
  }

private:
  FutureMap(const Legion::Domain &domain_, const Legion::FutureMap &lfm_)
      : domain(domain_), lfm(lfm_) {}

  // This is dangerous because we can't track the Future liveness after conversion
  operator Legion::FutureMap() const { return lfm; }

  friend class FutureMapSerializer;
  friend class Runtime;
};

}  // namespace ResilientLegion

#endif  // RESILIENCE_FUTURE_H
