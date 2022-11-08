#include "legion.h"

using namespace Legion;

namespace ResilientLegion
{
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

class ResilientFutureMap
{
 public:
  FutureMap fm;

  ResilientFutureMap(FutureMap fm_) : fm(fm_) {}

  void wait_all_results()
  {
    fm.wait_all_results();
  }
};

/* 1-D for now */
class ResilientDomainPoint
{
 public:
  /* I think this should be long long instead? */
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

  void attach_name(FieldSpace handle, const char *name, bool is_mutable = false);

  void attach_name(FieldSpace handle, FieldID fid, const char *name, bool is_mutable = false);

  void attach_name(IndexSpace handle, const char *name, bool is_mutable = false);

  void attach_name(LogicalRegion handle, const char *name, bool is_mutable = false);

  void attach_name(IndexPartition handle, const char *name, bool is_mutable = false);

  void issue_execution_fence(Context ctx, const char *provenance = NULL);

  ResilientFuture execute_task(Context, TaskLauncher, bool flag = false);

  FutureMap execute_index_space(Context, const IndexTaskLauncher &launcher);

  ResilientFuture get_current_time(Context, ResilientFuture = Future());

  ResilientFuture get_current_time_in_microseconds(Context, ResilientFuture = Future());

  template<int DIM, typename COORD_T>
  IndexSpaceT<DIM, COORD_T>
    create_index_space(Context ctx, const Rect<DIM, COORD_T> &bounds)
  {
    return lrt->create_index_space(ctx, bounds);
  }

  IndexSpace create_index_space_union(Context ctx, IndexPartition parent, const DomainPoint &color, const std::vector<IndexSpace> &handles);

  IndexSpace create_index_space_union(Context ctx, IndexPartition parent, const DomainPoint &color, IndexPartition handle);

  IndexSpace create_index_space_difference(Context ctx, IndexPartition parent, const DomainPoint &color, IndexSpace initial, const std::vector<IndexSpace> &handles);

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

      /* Everything is 1-D for now */
      std::vector<FieldID> fids;
      lrt->get_field_space_fields(fields, fids);
      AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, cpy);

      char file_name[20];
      sprintf(file_name, "lr.%ld.checkpoint", region_tag++);
      al.attach_file(file_name, fids, LEGION_FILE_READ_ONLY);

      PhysicalRegion pr = lrt->attach_external_resource(ctx, al);

      CopyLauncher cl;
      cl.add_copy_requirements(RegionRequirement(cpy, READ_ONLY, EXCLUSIVE, cpy),
                               RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));

      for (auto &id : fids)
      {
        /* Verify that the first index is ok */
        cl.add_src_field(0, id);
        cl.add_dst_field(0, id);
      }

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

  LogicalRegion create_logical_region(Context ctx, IndexSpace index, FieldSpace fields, bool task_local = false, const char *provenance = NULL);

  PhysicalRegion map_region(Context ctx, const InlineLauncher &launcher);

  void unmap_region(Context ctx, PhysicalRegion region);

  void destroy_index_space(Context ctx, IndexSpace handle);

  void destroy_field_space(Context ctx, FieldSpace handle);

  void destroy_logical_region(Context ctx, LogicalRegion handle);

  void destroy_index_partition(Context ctx, IndexPartition handle);

  IndexPartition create_equal_partition(Context ctx, IndexSpace parent, IndexSpace color_space);

  IndexPartition create_pending_partition(Context ctx, IndexSpace parent, IndexSpace color_space);

  Color create_cross_product_partitions(Context ctx, IndexPartition handle1, IndexPartition handle2, std::map<IndexSpace, IndexPartition> &handles);

  IndexPartition create_partition_by_field(Context ctx, LogicalRegion handle, LogicalRegion parent, FieldID fid, IndexSpace color_space);

  IndexPartition create_partition_by_image(Context ctx, IndexSpace handle, LogicalPartition projection, LogicalRegion parent, FieldID fid, IndexSpace color_space);

  IndexPartition create_partition_by_preimage(Context ctx, IndexPartition projection, LogicalRegion handle, LogicalRegion parent, FieldID fid, IndexSpace color_space);

  IndexPartition create_partition_by_difference(Context ctx, IndexSpace parent, IndexPartition handle1, IndexPartition handle2, IndexSpace color_space);

  LogicalPartition get_logical_partition(Context ctx, LogicalRegion parent, IndexPartition handle);

  LogicalPartition get_logical_partition(LogicalRegion parent, IndexPartition handle);

  LogicalPartition get_logical_partition_by_tree(IndexPartition handle, FieldSpace fspace, RegionTreeID tid);

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

  void save_index_partition(Context ctx, IndexSpace &color_space, IndexPartition &ip);

  IndexPartition restore_index_partition(Context ctx, const IndexSpace &index_space, IndexSpace &color_space);

  template<class Archive>
  void serialize(Archive &ar)
  {
    ar(futures, partitions);
  }

  void checkpoint(Context ctx);

 private:
  long unsigned int future_tag, region_tag, partition_tag;
  Runtime *lrt;
};
}
