#include "legion.h"

namespace ResilientLegion {

class ResilientFuture {
 public:
  Legion::Future lft;

  ResilientFuture(Legion::Future lft_) : lft(lft_) {}

  ResilientFuture() : lft(Legion::Future()) {}

  template<typename T>
  inline T get_result(bool silence_warnings = false,
                      const char *warning_string = NULL) const
  {
    return lft.get_result<T>();
  }
};

class ResilientRuntime {
 public:
  ResilientRuntime(Legion::Runtime *lrt_) : lrt(lrt_) {}

  ResilientFuture execute_task(Legion::Context ctx, Legion::TaskLauncher launcher) {
    return lrt->execute_task(ctx, launcher);
  }

  ResilientFuture get_current_time(Legion::Context ctx,
                                   ResilientFuture precondition = ResilientFuture())
  {
    return lrt->get_current_time(ctx, precondition.lft);
  }

 private:
  Legion::Runtime *lrt;
};
}
