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
  inline T get_result(std::vector<std::vector<char>> &results, bool replay) const
  {
    if (replay && tag < results.size() && !results[tag].empty())
    {
      T *tmp = reinterpret_cast<T*>(&results[tag][0]);
      return *tmp;
    }

    const void *ptr = lft.get_untyped_pointer();
    size_t size = lft.get_untyped_size();
    char *buf = (char *)ptr;
    std::vector<char> result(buf, buf + size);
    results[tag] = result;
    return lft.get_result<T>();
  }
};

class ResilientRuntime {
 public:
  std::vector<std::vector<char>> results;
  bool replay;

  ResilientRuntime(Legion::Runtime *lrt_) : curr_tag(0), lrt(lrt_)
  {
    Legion::InputArgs args = Legion::Runtime::get_input_args();
    replay = false;
    for (int i = 1; i < args.argc; i++)
      if (strstr(args.argv[i], "-replay"))
        replay = true;

    if (replay)
    {
      std::ifstream file("checkpost.legion");
      cereal::XMLInputArchive iarchive(file);
      iarchive(*this);
      file.close();
    }
  }

  ResilientFuture execute_task(Legion::Context ctx, Legion::TaskLauncher launcher) {
    if (replay && curr_tag < results.size() && !results[curr_tag].empty())
    {
      ResilientFuture empty = ResilientFuture();
      empty.tag = curr_tag++;
      return empty;
    }
    ResilientFuture ft = lrt->execute_task(ctx, launcher);
    ft.tag = curr_tag++;
    results.push_back(std::vector<char>());
    return ft;
  }

  ResilientFuture get_current_time(Legion::Context ctx,
                                   ResilientFuture precondition = ResilientFuture())
  {
    if (replay && curr_tag < results.size() && !results[curr_tag].empty())
    {
      ResilientFuture empty = ResilientFuture();
      empty.tag = curr_tag++;
      return empty;
    }
    ResilientFuture ft = lrt->get_current_time(ctx, precondition.lft);
    ft.tag = curr_tag++;
    results.push_back(std::vector<char>());
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
  long unsigned int curr_tag;
  Legion::Runtime *lrt;
};
}
