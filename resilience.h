#include "legion.h"

using namespace Legion;

namespace ResilientLegion {

class ResilientFuture
{
 public:
  long unsigned int tag;
  Future lft;

  ResilientFuture(Future lft_) : lft(lft_) {}

  ResilientFuture() : lft(Future()) {}

  template<class T>
  inline T get_result(std::vector<std::vector<char>> &futures, bool replay) const
  {
    if (replay && tag < futures.size() && !futures[tag].empty())
    {
      T *tmp = reinterpret_cast<T*>(&futures[tag][0]);
      return *tmp;
    }
  
    const void *ptr = lft.get_untyped_pointer();
    size_t size = lft.get_untyped_size();
    char *buf = (char *)ptr;
    std::vector<char> result(buf, buf + size);
    futures[tag] = result;
    return lft.get_result<T>();
  }
};

/* 1-D for now */
class ResilientDomainPoint
{
 public:
  unsigned point;

  ResilientDomainPoint() = default;

  ResilientDomainPoint(unsigned pt) : point(pt) {}

  ResilientDomainPoint(DomainPoint d)
  {
    point = d.get_point<1>();
  }

  bool operator<(const ResilientDomainPoint &rdp) const
  {
    return this->point < rdp.point;
  }

  template<class Archive>
  void serialize(Archive &ar)
  {
    ar(point);
  }
};

class ResilientDomain
{
 public:
  std::vector<std::array<ResilientDomainPoint, 2>> raw_rects;

  ResilientDomain() = default;

  ResilientDomain(Domain domain)
  {
    for (RectInDomainIterator<1> i(domain); i(); i++)
    {
      raw_rects.push_back({ (DomainPoint) i->lo, (DomainPoint) i->hi });
    }
  }

  template<class Archive>
  void serialize(Archive &ar)
  {
    ar(raw_rects);
  }
};

class ResilientIndexSpace
{
 public:
  ResilientDomain domain;

  ResilientIndexSpace() = default;

  ResilientIndexSpace(Domain d) : domain(d) {}

  template<class Archive>
  void serialize(Archive &ar)
  {
    ar(domain);
  }
};

class ResilientIndexPartition
{
 public:
  ResilientIndexSpace color_space;
  std::map<ResilientDomainPoint, ResilientIndexSpace> map;

  ResilientIndexPartition() = default;

  template<class Archive>
  void serialize(Archive &ar)
  {
    ar(color_space, map);
  }
};

class ResilientRuntime
{
 public:
  std::vector<std::vector<char>> futures;
  std::vector<LogicalRegion> regions;
  std::vector<ResilientIndexPartition> partitions;
  bool replay;

  ResilientRuntime(Runtime *);

  ResilientFuture execute_task(Context, TaskLauncher, bool flag = false);

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
    if (replay)
    {
      // Create empty lr from index and fields
      // Check if file corresponding to this region_tag (assuming 0 for now) exists and is non-empty.
      // Create another empty lr and attach it to the file.
      //   Since we are not returning this one, we don't need to launch a sub-task.
      // Issue a copy operation.
      // Return the first lr.

      std::cout << "Reconstructing logical region from checkpoint\n";
      LogicalRegion lr = lrt->create_logical_region(ctx, index, fields);
      LogicalRegion cpy = lrt->create_logical_region(ctx, index, fields);

      // Query this from args instead
      // Need to extend this for N-D regions
      std::vector<FieldID> fids = { 0 };
      AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, cpy);

      char file_name[20];
      // Fix this for multiple logical regions
      sprintf(file_name, "lr.%d.checkpoint", 0);
      al.attach_file(file_name, fids, LEGION_FILE_READ_ONLY);

      PhysicalRegion pr = lrt->attach_external_resource(ctx, al);

      CopyLauncher cl;
      cl.add_copy_requirements(RegionRequirement(cpy, READ_ONLY, EXCLUSIVE, cpy),
                               RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));

      cl.add_src_field(0, 0);
      cl.add_dst_field(0, 0);

      // Index launch this?
      lrt->issue_copy_operation(ctx, cl);
      {
        Future f = lrt->detach_external_resource(ctx, pr);
        f.get_void_result(true);
      }
      return lr;
    }
    LogicalRegion lr = lrt->create_logical_region(ctx, index, fields);
    regions.push_back(lr);
    return lr;
  }

  PhysicalRegion map_region(Context ctx, const InlineLauncher &launcher);

  void unmap_region(Context ctx, PhysicalRegion region);

  IndexPartition create_equal_partition(Context ctx, IndexSpace parent, IndexSpace color_space);

  LogicalPartition get_logical_partition(Context ctx, LogicalRegion parent, IndexPartition handle);

  LogicalRegion get_logical_subregion_by_color(Context ctx, LogicalPartition parent, Color c);

  template<typename T>
  void fill_field(Context ctx, LogicalRegion handle, LogicalRegion parent,
    FieldID fid, const T &value, Predicate pred = Predicate::TRUE_PRED)
  {
    if (replay && future_tag < futures.size())
    {
      assert(futures[future_tag].empty());
      std::cout << "Nooping this fill\n";
      future_tag++;
      return;
    }
    lrt->fill_field<T>(ctx, handle, parent, fid, value);
    future_tag++;
    futures.push_back(std::vector<char>());
  }

  void save_logical_region(Context ctx, LogicalRegion &lr, const char *file_name);

  template<class Archive>
  void serialize(Archive &ar)
  {
    ar(futures, partitions);
  }

  void checkpoint(Context ctx);

 private:
  long unsigned int future_tag, partition_tag;
  Runtime *lrt;
};
}
