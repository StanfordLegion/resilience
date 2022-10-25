#include "legion.h"

namespace ResilientLegion {

class ResilientFuture {
 public:
  long unsigned int tag;
  Legion::Future lft;

  ResilientFuture(Legion::Future lft_) : lft(lft_) {}

  ResilientFuture() : lft(Legion::Future()) {}

  template<class T>
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

  ResilientRuntime(Legion::Runtime *);

  ResilientFuture execute_task(Legion::Context, Legion::TaskLauncher);

  ResilientFuture get_current_time(Legion::Context, ResilientFuture = Legion::Future());

  template<class Archive>
  void serialize(Archive &ar)
  {
    ar(results);
  }

  void checkpoint();

 private:
  long unsigned int curr_tag;
  Legion::Runtime *lrt;
};
}
