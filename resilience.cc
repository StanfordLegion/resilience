#include <cassert>
#include <cstdio>
#include <iostream>
#include <cereal/types/vector.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>
#include <fstream>
#include "legion.h"

namespace ResilientLegion {

class ResilientFuture {
 public:
  long unsigned int tag;
  Legion::Future lft;

  ResilientFuture(Legion::Future lft_) : lft(lft_) {}

  ResilientFuture() : lft(Legion::Future()) {}

  template<typename T>
  inline T get_result(std::vector<std::shared_ptr<int>> &results) const
  {
    if (tag < results.size() && results[tag] != nullptr)
    {
      return *results[tag];
    }

    T result = lft.get_result<T>();
    std::shared_ptr<int> cpy = std::make_shared<T>();
    *cpy = result;
    results.at(tag) = cpy;
    return result;
  }
};

class ResilientRuntime {
 public:
  long unsigned int curr_tag;
  // Leaky
  std::vector<std::shared_ptr<int>> results;

  ResilientRuntime(Legion::Runtime *lrt_, Legion::InputArgs args)
    : curr_tag(0), lrt(lrt_)
  {
    replay = false;
    for (int i = 1; i < args.argc; i++)
      if (strstr(args.argv[i], "-replay"))
        replay = true;

    if (replay)
    {
      std::ifstream file("checkpost.legion");
      cereal::XMLInputArchive iarchive(file);
      iarchive(*this);
    }
  }

  ResilientFuture execute_task(Legion::Context ctx, Legion::TaskLauncher launcher) {
    if (replay && curr_tag < results.size() && results[curr_tag] != nullptr)
    {
      ResilientFuture empty = ResilientFuture();
      empty.tag = curr_tag++;
      return empty;
    }
    ResilientFuture ft = lrt->execute_task(ctx, launcher);
    ft.tag = curr_tag++;
    results.push_back(nullptr);
    return ft;
  }

  ResilientFuture get_current_time(Legion::Context ctx,
                                   ResilientFuture precondition = ResilientFuture())
  {
    ResilientFuture ft = lrt->get_current_time(ctx, precondition.lft);
    ft.tag = curr_tag++;
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
  bool replay;
};
}
