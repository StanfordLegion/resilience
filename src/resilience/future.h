/* Copyright 2024 Stanford University
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

struct FillLauncher;
class FutureMap;
class FutureMapSerializer;
struct IndexFillLauncher;
struct IndexTaskLauncher;
class Runtime;
struct TaskLauncher;
struct TimingLauncher;
struct TunableLauncher;

class Future {
public:
  inline Future();
  inline Future(const Future &f);
  inline Future(Future &&f);

#ifndef RESILIENCE_AUDIT_FUTURE_API
  // IMPORTANT: For users only. DO NOT USE INTERNALLY. For details see implementation.
  inline Future(const Legion::Future &f);
#endif  // RESILIENCE_AUDIT_FUTURE_API

  inline ~Future();

  inline Future &operator=(const Future &f);
  inline Future &operator=(Future &&f);

  template <class T>
  inline T get_result(bool silence_warnings = false,
                      const char *warning_string = NULL) const;

  inline void get_void_result(bool silence_warnings = false,
                              const char *warning_string = NULL) const;

  inline bool is_empty(bool block = false, bool silence_warnings = false,
                       const char *warning_string = NULL) const;
  inline bool is_ready(bool subscribe = false) const;

  inline const void *get_untyped_pointer(bool silence_warnings = false,
                                         const char *warning_string = NULL) const;
  inline size_t get_untyped_size(void) const;

  static Future from_untyped_pointer(Runtime *runtime, const void *buffer, size_t bytes);

  template <typename T>
  static Future from_value(Runtime *runtime, const T &value);

  bool operator<(const Future &o) const { return lft < o.lft; }

private:
  // This is dangerous because we can't track liveness after conversion
  operator Legion::Future() const { return lft; }

  inline Future(Runtime *r, const Legion::Future &f);

  inline void increment_ref();
  inline void decrement_ref();
  inline void mark_escaped() const;

private:
  Runtime *runtime;
  Legion::Future lft;

  friend struct FillLauncher;
  friend class FutureMap;
  friend class FutureSerializer;
  friend struct IndexFillLauncher;
  friend struct IndexTaskLauncher;
  friend class Runtime;
  friend struct TaskLauncher;
  friend struct TimingLauncher;
  friend struct TunableLauncher;
  friend Legion::Future c_obj_convert(const Future &f);
  friend Future c_obj_convert(Runtime *r, const Legion::Future &f);
};

class FutureMap {
public:
  inline FutureMap();
  inline FutureMap(const FutureMap &fm);
  inline FutureMap(FutureMap &&fm);

  inline ~FutureMap();

  inline FutureMap &operator=(const FutureMap &fm);
  inline FutureMap &operator=(FutureMap &&fm);

  template <typename T>
  T get_result(const DomainPoint &point, bool silence_warnings = false,
               const char *warning_string = NULL) const;

  inline Future get_future(const DomainPoint &point) const;

  inline void get_void_result(const DomainPoint &point, bool silence_warnings = false,
                              const char *warning_string = NULL) const;

  inline void wait_all_results(bool silence_warnings = false,
                               const char *warning_string = NULL) const;

private:
  inline FutureMap(Runtime *runtime_, const Legion::Domain &domain_,
                   const Legion::FutureMap &lfm_);

  // This is dangerous because we can't track liveness after conversion
  operator Legion::FutureMap() const { return lfm; }

  inline void increment_ref();
  inline void decrement_ref();
  inline void mark_escaped() const;

private:
  Runtime *runtime;
  Legion::Domain domain;
  Legion::FutureMap lfm;

  friend struct FillLauncher;
  friend class FutureMapSerializer;
  friend struct IndexFillLauncher;
  friend struct IndexTaskLauncher;
  friend class Runtime;
  friend struct TaskLauncher;
  friend struct TimingLauncher;
  friend struct TunableLauncher;
  friend Legion::FutureMap c_obj_convert(const FutureMap &fm);
};

}  // namespace ResilientLegion

#endif  // RESILIENCE_FUTURE_H
