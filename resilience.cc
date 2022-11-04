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
  if (replay)
  {
    // Fix the indexing
    ResilientPartition rp = partitions[0];

    Rect<1> color_domain(rp.color_space.front(), rp.color_space.back());
    IndexSpace color_space = lrt->create_index_space(ctx, color_domain);    
    MultiDomainPointColoring *mdpc = new MultiDomainPointColoring();
    for (auto &color : rp.color_space)
    {
      for (auto &sub_region : rp.sub_regions[color])
      {
        Rect<1> r(sub_region, sub_region);
        (*mdpc)[color].insert(r);
      }
    }
    IndexPartition ip = lrt->create_index_partition(ctx, parent, color_domain, *mdpc);
    return ip;
  }

  IndexPartition ip = lrt->create_equal_partition(ctx, parent, color_space); 

  // Save the index space for each color
  ResilientPartition rp;
  Rect<1> domain = lrt->get_index_space_domain(color_space);
  for (PointInRectIterator<1> pir(domain); pir(); pir++)
  {
    unsigned int point = (unsigned int) *pir;
    rp.color_space.push_back(point);

    IndexSpace ispace = lrt->get_index_subspace(ctx, ip, point);
    Rect<1> sub_domain = lrt->get_index_space_domain(ispace);
    std::vector<unsigned int> tmp;
    for (PointInRectIterator<1> sub_pir(sub_domain); sub_pir(); sub_pir++)
    {
      // Assuming no color is empty!
      // Todo: Change to std::map
      tmp.push_back((unsigned int) *sub_pir);
    }
    rp.sub_regions.push_back(tmp);
  }
  partitions.push_back(rp);
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

  std::ofstream file("checkpoint.dat");
  {
    // Change to binary later
    cereal::XMLOutputArchive oarchive(file);
    oarchive(*this);
  }
  file.close();
}
