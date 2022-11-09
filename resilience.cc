#include <cstdio>
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
#include "resilience.h"
#include "legion.h"

using namespace Legion;
using namespace ResilientLegion;

ResilientRuntime::ResilientRuntime(Runtime *lrt_)
  : future_tag(0), region_tag(0), partition_tag(0), lrt(lrt_)
{
  InputArgs args = Runtime::get_input_args();
  replay = false;
  for (int i = 1; i < args.argc; i++)
    if (strstr(args.argv[i], "-replay"))
      replay = true;

  if (replay)
  {
    std::ifstream file("checkpoint.dat");
    cereal::XMLInputArchive iarchive(file);
    iarchive(*this);
    file.close();
  }
}

void ResilientRuntime::attach_name(FieldSpace handle, const char *name, bool is_mutable)
{
  lrt->attach_name(handle, name, is_mutable);
}

void ResilientRuntime::attach_name(FieldSpace handle, FieldID fid, const char *name, bool is_mutable)
{
  lrt->attach_name(handle, fid, name, is_mutable);
}

void ResilientRuntime::attach_name(IndexSpace handle, const char *name, bool is_mutable)
{
  lrt->attach_name(handle, name, is_mutable);
}

void ResilientRuntime::attach_name(LogicalRegion handle, const char *name, bool is_mutable)
{
  lrt->attach_name(handle, name, is_mutable);
}

void ResilientRuntime::attach_name(IndexPartition handle, const char *name, bool is_mutable)
{
  lrt->attach_name(handle, name, is_mutable);
}

void ResilientRuntime::issue_execution_fence(Context ctx, const char *provenance)
{
  lrt->issue_execution_fence(ctx, provenance);
}

FutureMap ResilientRuntime::execute_index_space(Context ctx, const IndexTaskLauncher &launcher)
{
  return lrt->execute_index_space(ctx, launcher);
}

ResilientFuture ResilientRuntime::execute_task(Context ctx, TaskLauncher launcher, bool flag)
{
  if (replay && future_tag < max_future_tag)
  {
    std::cout << "No-oping task.\n";
    /* It is ok to return an empty ResilentFuture because get_result knows to
     * fetch the actual result from ResilientRuntime.futures by looking at the
     * tag. get_result should never be called on an empty Future.
     */
    ResilientFuture empty = ResilientFuture();
    empty.tag = future_tag++;
    return empty;
  }
  std::cout << "Executing task.\n";
  ResilientFuture ft = lrt->execute_task(ctx, launcher);
  ft.tag = future_tag++;
  futures.push_back(std::vector<char>());
  future_handles.push_back(ft);
  return ft;
}

ResilientFuture ResilientRuntime::get_current_time(Context ctx,
                                                   ResilientFuture precondition)
{
  if (replay && future_tag < futures.size())
  {
    assert(!futures[future_tag].empty());
    ResilientFuture empty = ResilientFuture();
    empty.tag = future_tag++;
    return empty;
  }
  ResilientFuture ft = lrt->get_current_time(ctx, precondition.lft);
  ft.tag = future_tag++;
  futures.push_back(std::vector<char>());
  future_handles.push_back(ft);
  return ft;
}

ResilientFuture ResilientRuntime::get_current_time_in_microseconds(
  Context ctx, ResilientFuture precondition)
{
  if (replay && future_tag < futures.size())
  {
    assert(!futures[future_tag].empty());
    ResilientFuture empty = ResilientFuture();
    empty.tag = future_tag++;
    return empty;
  }
  ResilientFuture ft = lrt->get_current_time_in_microseconds(ctx, precondition.lft);
  ft.tag = future_tag++;
  futures.push_back(std::vector<char>());
  future_handles.push_back(ft);
  return ft;
}

FieldSpace ResilientRuntime::create_field_space(Context ctx)
{
  return lrt->create_field_space(ctx);
}

FieldAllocator ResilientRuntime::create_field_allocator(
  Context ctx, FieldSpace handle)
{
  return lrt->create_field_allocator(ctx, handle);
}

LogicalRegion ResilientRuntime::create_logical_region(Context ctx, IndexSpace index, FieldSpace fields, bool task_local, const char *provenance)
{
  /* Just copying for now */
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

/* Inline mappings need to be disallowed */
PhysicalRegion ResilientRuntime::map_region(
  Context ctx, const InlineLauncher &launcher)
{
  return lrt->map_region(ctx, launcher);
}

void ResilientRuntime::unmap_region(
  Context ctx, PhysicalRegion region)
{
  return lrt->unmap_region(ctx, region);
}

void ResilientRuntime::destroy_index_space(Context ctx, IndexSpace handle)
{
  lrt->destroy_index_space(ctx, handle);
}

void ResilientRuntime::destroy_field_space(Context ctx, FieldSpace handle)
{
  lrt->destroy_field_space(ctx, handle);
}

void ResilientRuntime::destroy_logical_region(Context ctx, LogicalRegion handle)
{
  lrt->destroy_logical_region(ctx, handle);
}

void ResilientRuntime::destroy_index_partition(Context ctx, IndexPartition handle)
{
  lrt->destroy_index_partition(ctx, handle);
}

IndexSpace ResilientRuntime::create_index_space_union(Context ctx, IndexPartition parent, const DomainPoint &color, const std::vector<IndexSpace> &handles)
{
  return lrt->create_index_space_union(ctx, parent, color, handles);
}

IndexSpace ResilientRuntime::create_index_space_union(Context ctx, IndexPartition parent, const DomainPoint &color, IndexPartition handle)
{
  return lrt->create_index_space_union(ctx, parent, color, handle);
}

IndexSpace ResilientRuntime::create_index_space_difference(Context ctx, IndexPartition parent, const DomainPoint &color, IndexSpace initial, const std::vector<IndexSpace> &handles)
{
  return lrt->create_index_space_difference(ctx, parent, color, initial, handles);
}

Rect<1> make_rect(std::array<ResilientDomainPoint, 2> raw_rect)
{
  Rect<1> rect(raw_rect[0].point, raw_rect[1].point);
  return rect;
}

void ResilientRuntime::save_index_partition(Context ctx,
  IndexSpace color_space, IndexPartition ip)
{
  Domain color_domain = lrt->get_index_space_domain(ctx, color_space);

  ResilientIndexPartition rip;
  rip.color_space = color_domain; /* Implicit conversion */

  /* For rect in color space
   *   For point in rect
   *     Get the index space under this point
   *     Stuff everything into a ResilientIndexPartition
   */
  for (RectInDomainIterator<1> i(color_domain); i(); i++)
  {
    for (PointInRectIterator<1> j(*i); j(); j++)
    {
      IndexSpace sub_is = lrt->get_index_subspace(ctx, ip, (unsigned int) *j);
      if (sub_is == IndexSpace::NO_SPACE)
        continue;
      ResilientIndexSpace sub_ris(lrt->get_index_space_domain(ctx, sub_is));
      rip.map[(DomainPoint) *j] = sub_ris;
    }
  }
  partitions.push_back(rip);
}

IndexPartition ResilientRuntime::restore_index_partition(
  Context ctx, const IndexSpace &index_space, IndexSpace &color_space)
{
  assert(partition_tag < partitions.size());
  ResilientIndexPartition rip = partitions[partition_tag++];
  MultiDomainPointColoring *mdpc = new MultiDomainPointColoring();

  /* For rect in color space
   *   For point in rect
   *     Get the index space under this point
   *     For rect in index space
   *       Insert into mdpc at point
   */
  for (auto &raw_rect : rip.color_space.domain.raw_rects)
  {
    for (PointInRectIterator<1> i(make_rect(raw_rect)); i(); i++)
    {
      ResilientIndexSpace ris = rip.map[(DomainPoint) *i];
      for (auto &raw_rect_ris : ris.domain.raw_rects)
      {
        (*mdpc)[*i].insert(make_rect(raw_rect_ris));
      }
    }
  }

  /* Assuming the domain cannot change */
  Domain color_domain = lrt->get_index_space_domain(ctx, color_space);
  IndexPartition ip = lrt->create_index_partition(ctx, index_space, color_domain, *mdpc);
  return ip;
}

IndexPartition ResilientRuntime::create_equal_partition(
  Context ctx, IndexSpace parent, IndexSpace color_space)
{
  if (replay)
    return restore_index_partition(ctx, parent, color_space);

  IndexPartition ip = lrt->create_equal_partition(ctx, parent, color_space); 
  partition_handles.push_back(ip);
  // save_index_partition(ctx, color_space, ip);
  return ip;
}

IndexPartition ResilientRuntime::create_pending_partition(
  Context ctx, IndexSpace parent, IndexSpace color_space)
{
  if (replay)
    return restore_index_partition(ctx, parent, color_space);

  IndexPartition ip = lrt->create_pending_partition(ctx, parent, color_space); 
  partition_handles.push_back(ip);
  // save_index_partition(ctx, color_space, ip);
  return ip;
  // return lrt->create_pending_partition(ctx, parent, color_space); 
}

IndexPartition ResilientRuntime::create_partition_by_field(Context ctx,
  LogicalRegion handle, LogicalRegion parent, FieldID fid, IndexSpace color_space)
{
  if (replay)
    return restore_index_partition(ctx, handle.get_index_space(), color_space);

  IndexPartition ip = lrt->create_partition_by_field(ctx, handle, parent, fid, color_space);
  partition_handles.push_back(ip);
  // save_index_partition(ctx, color_space, ip);
  return ip;
}

IndexPartition ResilientRuntime::create_partition_by_image(
  Context ctx, IndexSpace handle, LogicalPartition projection,
  LogicalRegion parent, FieldID fid, IndexSpace color_space)
{
  if (replay)
    return restore_index_partition(ctx, handle, color_space);

  IndexPartition ip = lrt->create_partition_by_image(ctx, handle, projection, parent, fid, color_space);
  partition_handles.push_back(ip);
  // save_index_partition(ctx, color_space, ip);
  return ip;
}

IndexPartition ResilientRuntime::create_partition_by_preimage(
  Context ctx, IndexPartition projection, LogicalRegion handle,
  LogicalRegion parent, FieldID fid, IndexSpace color_space)
{
  if (replay)
    return restore_index_partition(ctx, handle.get_index_space(), color_space);

  IndexPartition ip = lrt->create_partition_by_preimage(ctx, projection, handle, parent, fid, color_space);
  partition_handles.push_back(ip);
  // save_index_partition(ctx, color_space, ip);
  return ip;
}

IndexPartition ResilientRuntime::create_partition_by_difference(
  Context ctx, IndexSpace parent, IndexPartition handle1,
  IndexPartition handle2, IndexSpace color_space)
{
  if (replay)
    return restore_index_partition(ctx, parent, color_space);

  IndexPartition ip = lrt->create_partition_by_difference(ctx, parent, handle1, handle2, color_space);
  partition_handles.push_back(ip);
  // save_index_partition(ctx, color_space, ip);
  return ip;
}

Color ResilientRuntime::create_cross_product_partitions(Context ctx, IndexPartition handle1, IndexPartition handle2, std::map<IndexSpace, IndexPartition> &handles)
{
  return lrt->create_cross_product_partitions(ctx, handle1, handle2, handles);
}

LogicalPartition ResilientRuntime::get_logical_partition(
  Context ctx, LogicalRegion parent, IndexPartition handle)
{
  return lrt->get_logical_partition(ctx, parent, handle);
}

LogicalPartition ResilientRuntime::get_logical_partition(
  LogicalRegion parent, IndexPartition handle)
{
  return lrt->get_logical_partition(parent, handle);
}

LogicalPartition ResilientRuntime::get_logical_partition_by_tree(IndexPartition handle, FieldSpace fspace, RegionTreeID tid)
{
  return lrt->get_logical_partition_by_tree(handle, fspace, tid);
}

LogicalRegion ResilientRuntime::get_logical_subregion_by_color(
  Context ctx, LogicalPartition parent, Color c)
{
  return lrt->get_logical_subregion_by_color(ctx, parent, c);
}

bool generate_disk_file(const char *file_name)
{
  int fd = open(file_name, O_CREAT | O_WRONLY, 0666);
  if (fd < 0)
  {
    perror("open");
    return false;
  }
  close(fd);
  return true;
}

void ResilientRuntime::save_logical_region(
  Context ctx, LogicalRegion &lr, const char *file_name)
{
  bool ok = generate_disk_file(file_name);
  assert(ok);

  LogicalRegion cpy = lrt->create_logical_region(ctx,
                        lr.get_index_space(), lr.get_field_space());

  std::vector<FieldID> fids;
  lrt->get_field_space_fields(lr.get_field_space(), fids);

  AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, cpy);
  al.attach_file(file_name, fids, LEGION_FILE_READ_WRITE);

  PhysicalRegion pr = lrt->attach_external_resource(ctx, al);

  CopyLauncher cl;
  cl.add_copy_requirements(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr),
                           RegionRequirement(cpy, READ_WRITE, EXCLUSIVE, cpy));

  for (auto &id : fids)
  {
    cl.add_src_field(0, id);
    cl.add_dst_field(0, id);
  }

  // Index launch this?
  lrt->issue_copy_operation(ctx, cl);

  {
    Future f = lrt->detach_external_resource(ctx, pr);
    f.get_void_result(true);
  }
}

void ResilientRuntime::checkpoint(Context ctx)
{
  /* Need to support multiple checkpoints */
  if (replay) return;

  char file_name[20];
  int counter = 0;
  for (auto &lr : regions)
  {
    sprintf(file_name, "lr.%d.checkpoint", counter++);
    save_logical_region(ctx, lr, file_name);
  }

  max_future_tag = future_tag;

  /* This should be a task instead */
  for (long unsigned i = 0; i < futures.size(); i++)
  {
    /* Because fills don't return a Future, we push an empty ResilientFuture
     * into future_handles form fills (and only fills).
     */
    if (futures[i].empty() && !future_handles[i].empty)
    {
      const void *ptr = future_handles[i].lft.get_untyped_pointer();
      size_t size = future_handles[i].lft.get_untyped_size();
      char *buf = (char *)ptr;
      std::vector<char> result(buf, buf + size);
      futures[i] = result;
      // assert(!futures[i].empty());
    }
  }

  for (auto &ip : partition_handles)
  {
    save_index_partition(ctx, lrt->get_index_partition_color_space_name(ctx, ip), ip);
  }

  std::ofstream file("checkpoint.dat");
  {
    // Change to binary later
    cereal::XMLOutputArchive oarchive(file);
    oarchive(*this);
  }
  file.close();
}
