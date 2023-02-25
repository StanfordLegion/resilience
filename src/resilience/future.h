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

// Wrappers for Legion futures

class FillLauncher;
class FutureMap;
class FutureMapSerializer;
class IndexFillLauncher;
class IndexTaskLauncher;
class Runtime;
class TaskLauncher;
class TimingLauncher;
class TunableLauncher;

class Future {
public:
  Runtime *runtime;
  Legion::Future lft;

  Future() : runtime(NULL) {}
  Future(const Future &f) : runtime(f.runtime), lft(f.lft) {}

  template <class T>
  inline T get_result(bool silence_warnings = false,
                      const char *warning_string = NULL) const;

  inline void get_void_result(bool silence_warnings = false,
                              const char *warning_string = NULL) const;

  inline const void *get_untyped_pointer(bool silence_warnings = false,
                                         const char *warning_string = NULL) const;
  inline size_t get_untyped_size(void) const;

  static Future from_untyped_pointer(Runtime *runtime, const void *buffer, size_t bytes);

  template <typename T>
  static Future from_value(Runtime *runtime, const T &value);

  bool operator<(const Future &o) const { return lft < o.lft; }

private:
  // This is dangerous because we can't track the Future liveness after conversion
  operator Legion::Future() const { return lft; }

  Future(Runtime *runtime_, const Legion::Future &lft_) : runtime(runtime_), lft(lft_) {}

  friend class FillLauncher;
  friend class FutureMap;
  friend class FutureSerializer;
  friend class IndexFillLauncher;
  friend class IndexTaskLauncher;
  friend class Runtime;
  friend class TaskLauncher;
  friend class TimingLauncher;
  friend class TunableLauncher;
};

class FutureMap {
public:
  Runtime *runtime;
  Legion::Domain domain;
  Legion::FutureMap lfm;

  FutureMap() : runtime(NULL) {}
  FutureMap(const FutureMap &fm) : runtime(fm.runtime), domain(fm.domain), lfm(fm.lfm) {}

  template <typename T>
  T get_result(const DomainPoint &point, bool silence_warnings = false,
               const char *warning_string = NULL) const;

  inline Future get_future(const DomainPoint &point) const;

  inline void get_void_result(const DomainPoint &point, bool silence_warnings = false,
                              const char *warning_string = NULL) const;

  inline void wait_all_results(bool silence_warnings = false,
                               const char *warning_string = NULL) const;

private:
  FutureMap(Runtime *runtime_, const Legion::Domain &domain_,
            const Legion::FutureMap &lfm_)
      : runtime(runtime_), domain(domain_), lfm(lfm_) {}

  // This is dangerous because we can't track the Future liveness after conversion
  operator Legion::FutureMap() const { return lfm; }

  friend class FillLauncher;
  friend class FutureMapSerializer;
  friend class IndexFillLauncher;
  friend class IndexTaskLauncher;
  friend class Runtime;
  friend class TaskLauncher;
  friend class TimingLauncher;
  friend class TunableLauncher;
};

}  // namespace ResilientLegion

#endif  // RESILIENCE_FUTURE_H
