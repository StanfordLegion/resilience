// -*- mode: c++ -*-
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

LEGION_DISABLE_DEPRECATED_WARNINGS

namespace ResilientLegion {

inline Future::Future() : runtime(NULL) {}

inline Future::Future(const Future &f) : runtime(f.runtime), lft(f.lft) {
  increment_ref();
}

inline Future::Future(Future &&f) : runtime(f.runtime), lft(f.lft) {
  f.runtime = NULL;
  f.lft = Legion::Future();
}

inline Future::Future(Runtime *r, const Legion::Future &f) : runtime(r), lft(f) {
  assert(!runtime || runtime->enabled);
  increment_ref();
}

#ifndef RESILIENCE_AUDIT_FUTURE_API
inline Future::Future(const Legion::Future &f) : runtime(NULL), lft(f) {
  // IMPORTANT: this assumes the future is coming from outside the task context (e.g., as
  // a task parameter). We make no attempt to track it, and therefore there is no runtime
  // at all here.

  // **DO NOT CALL THIS INTERNALLY.** It is intended to be used by EXTERNAL USERS ONLY to
  // provide shim support for said futures.
}
#endif  // RESILIENCE_AUDIT_FUTURE_API

inline Future::~Future() { decrement_ref(); }

inline Future &Future::operator=(const Future &f) {
  decrement_ref();
  runtime = f.runtime;
  lft = f.lft;
  increment_ref();
  return *this;
}

inline Future &Future::operator=(Future &&f) {
  decrement_ref();
  runtime = f.runtime;
  lft = f.lft;
  f.runtime = NULL;
  f.lft = Legion::Future();
  return *this;
}

inline void Future::increment_ref() {
  if (!runtime) return;

  auto &state = runtime->future_state[lft];
  if (state.moved_to != SIZE_MAX) {
    auto &moved = runtime->futures.at(state.moved_to);
    moved.increment_ref();
  }
  state.ref_count++;
}

inline void Future::decrement_ref() {
  if (!runtime) return;

  auto state = runtime->future_state.find(lft);
  assert(state != runtime->future_state.end());

  resilient_tag_t moved_to = state->second.moved_to;
  if (moved_to != SIZE_MAX) {
    auto &moved = runtime->futures.at(state->second.moved_to);
    moved.decrement_ref();
  }

  size_t count = --state->second.ref_count;
  // There is always at least one reference in the runtime. When we reach it, we know
  // there are no longer any user references.

  // But! If the future escaped, we still need to track it.
  if (count == 1 && (!state->second.escaped || moved_to != SIZE_MAX)) {
    resilient_tag_t tag = runtime->future_tags.at(lft);
    runtime->futures.erase(tag);  // Should decrement ref count to zero.
    assert(state->second.ref_count == 0);
    runtime->future_tags.erase(lft);
    runtime->future_state.erase(state);
  }
}

inline void Future::mark_escaped() const {
  if (!runtime) return;

  auto &state = runtime->future_state.at(lft);
  // Do this the first time only for efficiency
  if (!state.escaped && state.moved_to != SIZE_MAX) {
    auto &moved = runtime->futures.at(state.moved_to);
    moved.mark_escaped();
  }
  state.escaped = true;
}

template <class T>
inline T Future::get_result(bool silence_warnings, const char *warning_string) const {
  mark_escaped();
  return lft.get_result<T>(silence_warnings, warning_string);
}

inline void Future::get_void_result(bool silence_warnings,
                                    const char *warning_string) const {
  lft.get_void_result(silence_warnings, warning_string);
}

inline bool Future::is_empty(bool block, bool silence_warnings,
                             const char *warning_string) const {
  return lft.is_empty(block, silence_warnings, warning_string);
}

inline bool Future::is_ready(bool subscribe) const { return lft.is_ready(subscribe); }

inline const void *Future::get_untyped_pointer(bool silence_warnings,
                                               const char *warning_string) const {
  mark_escaped();
  return lft.get_untyped_pointer(silence_warnings, warning_string);
}

inline size_t Future::get_untyped_size(void) const {
  mark_escaped();
  return lft.get_untyped_size();
}

template <typename T>
Future Future::from_value(Runtime *runtime, const T &value) {
  // There is no reason to ever register a Future::from_value. Even if checkpointing is
  // enabled, the user will provide the value to us on replay.
  return Future(NULL, Legion::Future::from_value<T>(value));
}

inline FutureMap::FutureMap() : runtime(NULL) {}

inline FutureMap::FutureMap(const FutureMap &fm)
    : runtime(fm.runtime), domain(fm.domain), lfm(fm.lfm) {
  increment_ref();
}

inline FutureMap::FutureMap(FutureMap &&fm)
    : runtime(fm.runtime), domain(fm.domain), lfm(fm.lfm) {
  fm.runtime = NULL;
  fm.domain = Domain::NO_DOMAIN;
  fm.lfm = Legion::FutureMap();
}

inline FutureMap::FutureMap(Runtime *r, const Legion::Domain &d,
                            const Legion::FutureMap &fm)
    : runtime(r), domain(d), lfm(fm) {
  increment_ref();
}

inline FutureMap::~FutureMap() { decrement_ref(); }

inline FutureMap &FutureMap::operator=(const FutureMap &fm) {
  decrement_ref();
  runtime = fm.runtime;
  domain = fm.domain;
  lfm = fm.lfm;
  increment_ref();
  return *this;
}

inline FutureMap &FutureMap::operator=(FutureMap &&fm) {
  decrement_ref();
  runtime = fm.runtime;
  domain = fm.domain;
  lfm = fm.lfm;
  fm.runtime = NULL;
  fm.domain = Domain::NO_DOMAIN;
  fm.lfm = Legion::FutureMap();
  return *this;
}

inline void FutureMap::increment_ref() {
  if (!runtime) return;

  auto &state = runtime->future_map_state[lfm];
  state.ref_count++;
}

inline void FutureMap::decrement_ref() {
  if (!runtime) return;

  auto state = runtime->future_map_state.find(lfm);
  assert(state != runtime->future_map_state.end());
  size_t count = --state->second.ref_count;
  // There is always at least one reference in the runtime. When we reach it, we know
  // there are no longer any user references.

  // But! If the future escaped, we still need to track it.
  if (count == 1 && !state->second.escaped) {
    resilient_tag_t tag = runtime->future_map_tags.at(lfm);
    runtime->future_maps.erase(tag);  // Should decrement ref count to zero.
    assert(state->second.ref_count == 0);
    runtime->future_map_tags.erase(lfm);
    runtime->future_map_state.erase(state);
  }
}

inline void FutureMap::mark_escaped() const {
  if (!runtime) return;

  auto &state = runtime->future_map_state.at(lfm);
  state.escaped = true;
}

template <typename T>
T FutureMap::get_result(const DomainPoint &point, bool silence_warnings,
                        const char *warning_string) const {
  mark_escaped();
  return lfm.get_result<T>(point, silence_warnings, warning_string);
}

inline Future FutureMap::get_future(const DomainPoint &point) const {
  if (!runtime) return Future(NULL, lfm.get_future(point));

  // We track this as a Future, so no need to track the FutureMap at this point
  if (!runtime->enabled) {
    return Future(runtime, lfm.get_future(point));
  }

  if (runtime->replay_future()) {
    return runtime->restore_future();
  }

  Future f(runtime, lfm.get_future(point));
  runtime->register_future(f);
  return f;
}

inline void FutureMap::get_void_result(const DomainPoint &point, bool silence_warnings,
                                       const char *warning_string) const {
  lfm.get_void_result(point, silence_warnings, warning_string);
}

inline void FutureMap::wait_all_results(bool silence_warnings,
                                        const char *warning_string) const {
  lfm.wait_all_results(silence_warnings, warning_string);
}

}  // namespace ResilientLegion

LEGION_REENABLE_DEPRECATED_WARNINGS
