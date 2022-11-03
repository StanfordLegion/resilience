#include <cstdio>
#include <iostream>
#include <cassert>
#include <cereal/types/vector.hpp>
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
  : future_tag(0), lrt(lrt_)
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

ResilientFuture ResilientRuntime::execute_task(Context ctx, TaskLauncher launcher, bool flag)
{
  if (replay && future_tag < futures.size() && (!futures[future_tag].empty() || flag))
  {
    std::cout << "Nooping this task\n";
    ResilientFuture empty = ResilientFuture();
    empty.tag = future_tag++;
    return empty;
  }
  std::cout << "Executing this task\n";
  ResilientFuture ft = lrt->execute_task(ctx, launcher);
  ft.tag = future_tag++;
  futures.push_back(std::vector<char>());
  return ft;
}

ResilientFuture ResilientRuntime::get_current_time(Context ctx,
                                                   ResilientFuture precondition)
{
  if (replay && future_tag < futures.size() && !futures[future_tag].empty())
  {
    ResilientFuture empty = ResilientFuture();
    empty.tag = future_tag++;
    return empty;
  }
  ResilientFuture ft = lrt->get_current_time(ctx, precondition.lft);
  ft.tag = future_tag++;
  futures.push_back(std::vector<char>());
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

IndexPartition ResilientRuntime::create_equal_partition(
  Context ctx, IndexSpace parent, IndexSpace color_space)
{
  auto ip = lrt->create_equal_partition(ctx, parent, color_space); 
  partitions.push_back(ip);
  return ip;
}

LogicalPartition ResilientRuntime::get_logical_partition(
  Context ctx, LogicalRegion parent, IndexPartition handle)
{
  return lrt->get_logical_partition(ctx, parent, handle);
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
  Context ctx, LogicalRegion &lr, const char* file_name)
{
  bool ok = generate_disk_file(file_name);
  assert(ok);

  LogicalRegion cpy = lrt->create_logical_region(ctx,
                        lr.get_index_space(), lr.get_field_space());

  std::vector<FieldID> fids = { 0 };
  AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, cpy);
  al.attach_file(file_name, fids, LEGION_FILE_READ_WRITE);

  PhysicalRegion pr = lrt->attach_external_resource(ctx, al);

  CopyLauncher cl;
  cl.add_copy_requirements(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr),
                           RegionRequirement(cpy, READ_WRITE, EXCLUSIVE, cpy));

  cl.add_src_field(0, 0);
  cl.add_dst_field(0, 0);

  // Index launch this?
  lrt->issue_copy_operation(ctx, cl);

  {
    Future f = lrt->detach_external_resource(ctx, pr);
    f.get_void_result(true);
  }
}

void ResilientRuntime::checkpoint(Context ctx)
{
  if (replay) return;

  char file_name[20];
  int counter = 0;
  for (auto &lr : regions)
  {
    sprintf(file_name, "lr.%d.checkpoint", counter++);
    save_logical_region(ctx, lr, file_name);
  }

  FieldSpace fspace = lrt->create_field_space(ctx);
  {
    FieldAllocator fal = lrt->create_field_allocator(ctx, fspace);
    fal.allocate_field(sizeof(int), 0);
  }

  counter = 0;
  for (IndexPartition &ip : partitions)
  {
    // For each partition, create (colors + 1) logical regions: one from its
    // color space, and the rest from the index space corresponding to each
    // color.

    IndexSpace cspace = lrt->get_index_partition_color_space_name(ctx, ip);
    LogicalRegion lr = lrt->create_logical_region(ctx, cspace, fspace);
    lrt->fill_field<int>(ctx, lr, lr, 0, 0);

    sprintf(file_name, "p.%d.checkpoint", counter++);
    save_logical_region(ctx, lr, file_name);

    // How do I iterate over an arbitrary region here?
    Rect<1> domain = lrt->get_index_space_domain(cspace);
    for (PointInRectIterator<1> pir(domain); pir(); pir++)
    {
      IndexSpace ispace = lrt->get_index_subspace(ctx, ip, (unsigned int)*pir);
      LogicalRegion lr = lrt->create_logical_region(ctx, ispace, fspace);
      lrt->fill_field<int>(ctx, lr, lr, 0, 0);
      sprintf(file_name, "p.%d.checkpoint", counter++);
      save_logical_region(ctx, lr, file_name);
    } 
  }

  std::ofstream file("checkpoint.dat");
  {
    // Change to binary later
    cereal::XMLOutputArchive oarchive(file);
    oarchive(*this);
  }
  file.close();
}
