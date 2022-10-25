#include <cstdio>
#include <cereal/types/vector.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>
#include <fstream>
#include "resilience.h"
#include "legion.h"

using namespace ResilientLegion;

ResilientRuntime::ResilientRuntime(Legion::Runtime *lrt_) : curr_tag(0), lrt(lrt_)
{
  Legion::InputArgs args = Legion::Runtime::get_input_args();
  replay = false;
  for (int i = 1; i < args.argc; i++)
    if (strstr(args.argv[i], "-replay"))
      replay = true;

  if (replay)
  {
    std::ifstream file("checkpost.legion");
    cereal::XMLInputArchive iarchive(file);
    iarchive(*this);
    file.close();
  }
}

ResilientFuture ResilientRuntime::execute_task(Legion::Context ctx, Legion::TaskLauncher launcher) {
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

ResilientFuture ResilientRuntime::get_current_time(Legion::Context ctx,
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

void ResilientRuntime::checkpoint()
{
  std::ofstream file("checkpost.legion");
  {
    // Change to binary later
    cereal::XMLOutputArchive oarchive(file);
    oarchive(*this);
  }
  file.close();
}