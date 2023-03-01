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


#ifndef __RESILIENT_LEGION_C_UTIL_H__
#define __RESILIENT_LEGION_C_UTIL_H__

/**
 * \file resilience_c_util.h
 * Legion C API: C++ Conversion Utilities
 */

#include "legion/legion_c_util.h"
#include "resilience.h"
#include "resilience/resilience_c.h"

namespace ResilientLegion {

    using Legion::CContext;
    using Legion::TaskMut;

    class ResilientCObjectWrapper {
    public:

#ifdef __ICC
// icpc complains about "error #858: type qualifier on return type is meaningless"
// but it's pretty annoying to get this macro to handle all the cases right
#pragma warning (push)
#pragma warning (disable: 858)
#endif
#ifdef __PGIC__
#pragma warning (push)
#pragma diag_suppress 191
#pragma diag_suppress 816
#endif
#define NEW_OPAQUE_WRAPPER(T_, T)                                       \
      static T_ wrap(T t) {                                             \
        T_ t_;                                                          \
        t_.impl = static_cast<void *>(t);                               \
        return t_;                                                      \
      }                                                                 \
      static const T_ wrap_const(const T t) {                           \
        T_ t_;                                                          \
        t_.impl = const_cast<void *>(static_cast<const void *>(t));     \
        return t_;                                                      \
      }                                                                 \
      static T unwrap(T_ t_) {                                          \
        return static_cast<T>(t_.impl);                                 \
      }                                                                 \
      static const T unwrap_const(const T_ t_) {                        \
        return static_cast<const T>(t_.impl);                           \
      }

      NEW_OPAQUE_WRAPPER(resilient_legion_runtime_t, Runtime *);
      NEW_OPAQUE_WRAPPER(resilient_legion_future_t, Future *);
      NEW_OPAQUE_WRAPPER(resilient_legion_future_map_t, FutureMap *);
      NEW_OPAQUE_WRAPPER(resilient_legion_task_launcher_t, TaskLauncher *);
      NEW_OPAQUE_WRAPPER(resilient_legion_index_launcher_t, IndexTaskLauncher *);
      NEW_OPAQUE_WRAPPER(resilient_legion_fill_launcher_t, FillLauncher *);
      NEW_OPAQUE_WRAPPER(resilient_legion_index_fill_launcher_t, IndexFillLauncher *);
#undef NEW_OPAQUE_WRAPPER
#ifdef __ICC
// icpc complains about "error #858: type qualifier on return type is meaningless"
// but it's pretty annoying to get this macro to handle all the cases right
#pragma warning (pop)
#endif
#ifdef __PGIC__
#pragma warning (pop)
#endif
    };
};

#endif // __RESILIENT_LEGION_C_UTIL_H__
