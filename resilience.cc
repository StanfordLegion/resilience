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

ResilientRuntime::ResilientRuntime(Runtime *lrt_) : curr_tag(0), lrt(lrt_)
{
  InputArgs args = Runtime::get_input_args();
  replay = false;
  for (int i = 1; i < args.argc; i++)
    if (strstr(args.argv[i], "-replay"))
      replay = true;

  if (replay)
  {
    std::ifstream file("results.checkpoint");
    cereal::XMLInputArchive iarchive(file);
    iarchive(*this);
    file.close();
  }
}

ResilientFuture ResilientRuntime::execute_task(Context ctx, TaskLauncher launcher) {
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

ResilientFuture ResilientRuntime::get_current_time(Context ctx,
                                                   ResilientFuture precondition)
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

bool generate_disk_file(const char *file_name)
{
  // strip off any filename prefix starting with a colon
  {
    const char *pos = strchr(file_name, ':');
    if(pos) file_name = pos + 1;
  }

  // create the file if needed
  int fd = open(file_name, O_CREAT | O_WRONLY, 0666);
  if(fd < 0) {
    perror("open");
    return false;
  }

  // now close the file - the Legion runtime will reopen it on the attach
  close(fd);
  return true;
}

void ResilientRuntime::checkpoint(Context ctx)
{
  // Save lr
  LogicalRegion lr = lrs[0];
  LogicalRegion cpy = lrt->create_logical_region(ctx,
                         lr.get_index_space(), lr.get_field_space());

  bool ok = generate_disk_file("lr.checkpoint");
  assert(ok);

  std::vector<FieldID> fids = { 0 };
  AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, cpy);
  al.attach_file("lr.checkpoint", fids, LEGION_FILE_READ_WRITE);

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

  std::ofstream file("results.checkpoint");
  {
    // Change to binary later
    cereal::XMLOutputArchive oarchive(file);
    oarchive(*this);
  }
  file.close();
}
