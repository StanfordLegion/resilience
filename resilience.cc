#include <cassert>
#include "legion.h"

namespace ResilientLegion {

class ResilientRuntime;

class ResilientFuture {
 public:
  int tag;
  Legion::Future lft;

  ResilientFuture(Legion::Future lft_) : lft(lft_) {}

  ResilientFuture() : lft(Legion::Future()) {}

  template<typename T>
  inline T get_result(std::vector<std::pair<ResilientFuture, void *>> state_fts) const
  {
    T result = lft.get_result<T>();
    T *cpy = new T;
    *cpy = result;
    assert(state_fts.at(tag).second == nullptr);
    state_fts[tag].second = cpy;
    return result;
  }
};

class ResilientRuntime {
 public:
  ResilientRuntime(Legion::Runtime *lrt_) : lrt(lrt_) {}

  // Leaky
  std::vector<std::pair<ResilientFuture, void *>> state_fts;

  ResilientFuture execute_task(Legion::Context ctx, Legion::TaskLauncher launcher) {
    ResilientFuture ft = lrt->execute_task(ctx, launcher);
    ft.tag = state_fts.size();
    state_fts.push_back(std::make_pair(ft, nullptr));
    return ft;
  }

  ResilientFuture get_current_time(Legion::Context ctx,
                                   ResilientFuture precondition = ResilientFuture())
  {
    ResilientFuture ft = lrt->get_current_time(ctx, precondition.lft);
    ft.tag = state_fts.size();
    state_fts.push_back(std::make_pair(ft, nullptr));
    return ft;
  }

 private:
  Legion::Runtime *lrt;
};
}
