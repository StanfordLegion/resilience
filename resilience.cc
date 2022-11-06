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

Rect<1> make_rect(std::array<ResilientDomainPoint, 2> raw_rect)
{
  Rect<1> rect(raw_rect[0].point, raw_rect[1].point);
  return rect;
}

IndexPartition ResilientRuntime::create_equal_partition(
  Context ctx, IndexSpace parent, IndexSpace color_space)
{
  if (replay)
  {
    /* Fix indexing */
    ResilientIndexPartition rip = partitions[0];
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
    IndexPartition ip = lrt->create_index_partition(ctx, parent, color_domain, *mdpc);
    return ip;
  }

  IndexPartition ip = lrt->create_equal_partition(ctx, parent, color_space); 
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
      ResilientIndexSpace sub_ris(lrt->get_index_space_domain(ctx, sub_is));
      rip.map[(DomainPoint) *j] = sub_ris;
    }
  }
  partitions.push_back(rip);
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
