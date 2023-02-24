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

template <class T>
inline T Future::get_result(bool silence_warnings, const char *warning_string) const {
  return lft.get_result<T>(silence_warnings, warning_string);
}

inline void Future::get_void_result(bool silence_warnings,
                                    const char *warning_string) const {
  lft.get_void_result(silence_warnings, warning_string);
}

inline const void *Future::get_untyped_pointer(bool silence_warnings,
                                               const char *warning_string) const {
  return lft.get_untyped_pointer(silence_warnings, warning_string);
}

inline size_t Future::get_untyped_size(void) const { return lft.get_untyped_size(); }

template <typename T>
Future Future::from_value(Runtime *runtime, const T &value) {
  if (!runtime->enabled) {
    return Legion::Future::from_value<T>(runtime->lrt, value);
  }

  if (runtime->replay_future()) {
    return runtime->restore_future();
  }

  Future f = Legion::Future::from_value<T>(runtime->lrt, value);
  runtime->register_future(f);
  return f;
}

}  // namespace ResilientLegion
