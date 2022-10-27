#include "legion.h"

using namespace Legion;

namespace ResilientLegion {

class ResilientFuture {
 public:
  long unsigned int tag;
  Future lft;

  ResilientFuture(Future lft_) : lft(lft_) {}

  ResilientFuture() : lft(Future()) {}

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
  std::vector<LogicalRegion> lrs;
  bool replay;

  ResilientRuntime(Runtime *);

  ResilientFuture execute_task(Context, TaskLauncher);

  ResilientFuture get_current_time(Context, ResilientFuture = Future());

  template<int DIM, typename COORD_T>
  IndexSpaceT<DIM, COORD_T>
    create_index_space(Context ctx, const Rect<DIM, COORD_T> &bounds)
  {
    return lrt->create_index_space(ctx, bounds);
  }

  FieldSpace create_field_space(Context ctx);

  FieldAllocator create_field_allocator(Context ctx, FieldSpace handle);

  template<int DIM, typename COORD_T>
  LogicalRegion
    create_logical_region(Context ctx,
                          IndexSpaceT<DIM, COORD_T> index,
                          FieldSpace fields)
  {
    lrs.push_back(lrt->create_logical_region(ctx, index, fields));
    return lrs[0];
  }

  PhysicalRegion map_region(Context ctx, const InlineLauncher &launcher);

  void unmap_region(Context ctx, PhysicalRegion region);

  template<class Archive>
  void serialize(Archive &ar)
  {
    ar(results);
  }

  void checkpoint(Context ctx);

 private:
  long unsigned int curr_tag;
  Runtime *lrt;
};
}
