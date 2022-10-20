#include <cassert>
#include <iostream>
#include <cereal/types/vector.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>
#include <fstream>
#include "legion.h"

namespace ResilientLegion {

class ResilientFuture {
 public:
  int tag;
  Legion::Future lft;

  ResilientFuture(Legion::Future lft_) : lft(lft_) {}

  ResilientFuture() : lft(Legion::Future()) {}

  template<typename T>
  inline T get_result(std::vector<std::shared_ptr<int>> &results) const
  {
    T result = lft.get_result<T>();
    std::shared_ptr<int> cpy = std::make_shared<T>();
    *cpy = result;
    results.at(tag) = cpy;
    // results.at(tag) = cpy;
    // auto cpy = std::shared_ptr<void>(result);
    // T *cpy = new T;
    // *cpy = result;
    // assert(results.at(tag) == nullptr);
    // results.at(tag) = cpy;
    // results.at(tag) = std::make_shared
    return result;
  }
};

class ResilientRuntime {
 public:
  // Leaky
  std::vector<std::shared_ptr<int>> results;

  ResilientRuntime(Legion::Runtime *lrt_) : lrt(lrt_) {}

  ResilientFuture execute_task(Legion::Context ctx, Legion::TaskLauncher launcher) {
    ResilientFuture ft = lrt->execute_task(ctx, launcher);
    ft.tag = results.size();
    results.push_back(nullptr);
    return ft;
  }

  ResilientFuture get_current_time(Legion::Context ctx,
                                   ResilientFuture precondition = ResilientFuture())
  {
    ResilientFuture ft = lrt->get_current_time(ctx, precondition.lft);
    ft.tag = results.size();
    results.push_back(nullptr);
    return ft;
  }

  template<class Archive>
  void serialize(Archive &ar)
  {
    ar(results);
  }

  void checkpoint()
  {
    std::ofstream file("checkpost.legion");
    {
      // Change to binary later
      cereal::XMLOutputArchive oarchive(file);
      oarchive(*this);
    }
    file.close();
  }

 private:
  Legion::Runtime *lrt;
};
}
