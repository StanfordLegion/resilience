#include <cstdio>
#include <cstring>
#include <iostream>
#include <cassert>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/array.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>
#include "legion.h"

using namespace Legion;

namespace ResilientLegion
{

class ResilientFuture
{
 public:
  Future lft;
  std::vector<char> result;
  bool empty; /* Problematic with predicates? */
  bool is_fill;

  ResilientFuture(Future lft_) : lft(lft_), empty(false), is_fill(false) {}
  ResilientFuture() : lft(Future()), empty(true), is_fill(false) {}

  void setup_for_checkpoint()
  {
    if (is_fill) return;

    const void *ptr = lft.get_untyped_pointer();
    size_t size = lft.get_untyped_size();
    char *buf = (char *)ptr;
    std::vector<char> tmp(buf, buf + size);
    result = tmp;
  }

  /* Did this have to be declared const? */
  template<class T>
  inline T get_result()
  {
    assert(!is_fill);
    if (!result.empty())
    {
      return *reinterpret_cast<T*>(&result[0]);
    }
    const void *ptr = lft.get_untyped_pointer();
    char *buf = (char *)ptr;
    std::vector<char> tmp(buf, buf + sizeof(T));
    result = tmp;
    return *static_cast<const T*>(ptr);
  }

  template<class Archive>
  void serialize(Archive &ar)
  {
    ar(empty, is_fill, result);
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
      raw_rects.push_back({
        static_cast<DomainPoint>(i->lo),
        static_cast<DomainPoint>(i->hi)
      });
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
  IndexPartition ip;
  ResilientIndexSpace color_space;
  std::map<ResilientDomainPoint, ResilientIndexSpace> map;

  ResilientIndexPartition() = default;
  ResilientIndexPartition(IndexPartition ip_) : ip(ip_) {}

  void setup_for_checkpoint(Context ctx, Runtime *lrt);

  template<class Archive>
  void serialize(Archive &ar)
  {
    ar(color_space, map);
  }
};

class ResilientFutureMap
{
 public:
  FutureMap fm;
  Domain d;
  std::map<ResilientDomainPoint, std::vector<char>> map;

  ResilientFutureMap() = default;

  ResilientFutureMap(FutureMap fm_, Domain d_) : fm(fm_), d(d_) {}

  void setup_for_checkpoint()
  {
    for (PointInDomainIterator<1> i(d); i(); i++)
    {
      Future ft = fm.get_future(*i);
      const void *ptr = ft.get_untyped_pointer();
      size_t size = ft.get_untyped_size();
      char *buf = (char *)ptr;
      std::vector<char> result(buf, buf + size);
      map[static_cast<DomainPoint>(*i)] = result;
    }
  }

  template<typename T>
  T get_result(const DomainPoint &point, bool replay)
  {
    if (replay)
    {
      T *tmp = reinterpret_cast<T *>(&map[point][0]);
      return *tmp;
    }
    return fm.get_result<T>(point);
  }

  void wait_all_results(bool replay)
  {
    if (replay)
      return;
    fm.wait_all_results();
  }

  template<class Archive>
  void serialize(Archive &ar)
  {
    ar(map);
  }
};

class ResilientRuntime
{
 public:
  std::vector<ResilientFuture> futures;
  std::vector<LogicalRegion> regions; /* Not persistent */
  std::vector<ResilientIndexPartition> partitions;
  std::vector<ResilientFutureMap> future_maps;
  bool replay;
  long unsigned int future_tag, future_map_tag, region_tag, partition_tag;
  long unsigned max_future_tag, max_future_map_tag, max_partition_tag;

  ResilientRuntime(Context ctx, Runtime *);

  void attach_name(FieldSpace handle, const char *name, bool is_mutable = false);

  void attach_name(FieldSpace handle, FieldID fid, const char *name, bool is_mutable = false);

  void attach_name(IndexSpace handle, const char *name, bool is_mutable = false);

  void attach_name(LogicalRegion handle, const char *name, bool is_mutable = false);

  void attach_name(IndexPartition handle, const char *name, bool is_mutable = false);

  void issue_execution_fence(Context ctx, const char *provenance = NULL);

  ResilientFuture execute_task(Context, TaskLauncher);

  ResilientFutureMap execute_index_space(Context, const IndexTaskLauncher &launcher);

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

  LogicalRegion create_logical_region(Context ctx, IndexSpace index, FieldSpace fields, bool task_local = false, const char *provenance = NULL);

  template<int DIM, typename COORD_T>
  LogicalRegion
    create_logical_region(Context ctx,
                          IndexSpaceT<DIM, COORD_T> index,
                          FieldSpace fields)
  {
    return create_logical_region(ctx, static_cast<IndexSpace>(index),
      static_cast<FieldSpace>(fields));
  }

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

  template<int DIM, int COLOR_DIM, typename COORD_T>
  IndexPartitionT<DIM, COORD_T> create_partition_by_restriction(Context ctx, IndexSpaceT<DIM, COORD_T> parent, IndexSpaceT<COLOR_DIM, COORD_T> color_space, Transform<DIM, COLOR_DIM, COORD_T> transform, Rect<DIM, COORD_T> extent)
  {
    if (replay)
      return static_cast<IndexPartitionT<DIM, COORD_T>>(
        restore_index_partition(ctx, static_cast<IndexSpace>(parent),
          static_cast<IndexSpace>(color_space)));

    IndexPartitionT<DIM, COORD_T> ip = lrt->create_partition_by_restriction(ctx,
      parent, color_space, transform, extent);
    partitions.push_back(static_cast<ResilientIndexPartition>(ip));
    return ip;
  }

  LogicalPartition get_logical_partition(Context ctx, LogicalRegion parent, IndexPartition handle);

  LogicalPartition get_logical_partition(LogicalRegion parent, IndexPartition handle);

  LogicalPartition get_logical_partition_by_tree(IndexPartition handle, FieldSpace fspace, RegionTreeID tid);

  LogicalRegion get_logical_subregion_by_color(Context ctx, LogicalPartition parent, Color c);

  template<typename T>
  void fill_field(Context ctx, LogicalRegion handle, LogicalRegion parent,
    FieldID fid, const T &value, Predicate pred = Predicate::TRUE_PRED)
  {
    if (replay && future_tag < max_future_tag)
    {
      std::cout << "Nooping this fill\n";
      future_tag++;
      return;
    }
    lrt->fill_field<T>(ctx, handle, parent, fid, value);
    future_tag++;
    /* We have to push something into the vector here because future_tag gets
     * out of sync with the vector otherwise. And the user never sees this
     * ResilientFuture so we're fine. */
    ResilientFuture ft;
    ft.is_fill = true;
    futures.push_back(ft);
  }

  void save_logical_region(Context ctx, LogicalRegion &lr, const char *file_name);

  void save_index_partition(Context ctx, IndexSpace color_space, IndexPartition ip);

  IndexPartition restore_index_partition(Context ctx, IndexSpace index_space, IndexSpace color_space);

  template<class Archive>
  void serialize(Archive &ar)
  {
    ar(max_future_tag, max_future_map_tag, futures, future_maps, partitions);
  }

  void checkpoint(Context ctx);

 private:
  Runtime *lrt;
};
}
