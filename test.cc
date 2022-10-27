#include <iostream>
#include "resilience.h"
#include "legion.h"

using namespace Legion;

void check(const Task *task,
           const std::vector<PhysicalRegion> &regions,
           Context ctx, Runtime *runtime)
{
  auto pr = regions[0];
  const FieldAccessor<READ_ONLY, int, 1> acc(pr, 0);
  const Rect<1> domain(0, 10);
  for (PointInRectIterator<1> pir(domain); pir(); pir++)
  {
    std::cout << "Data from checkpoint " << acc[*pir] << std::endl;
  }
}

void top_level_alt(const Task *task,
           const std::vector<PhysicalRegion> &regions,
           Context ctx, Runtime *runtime)
{
  int N = 10;
  const Rect<1> domain(0, N);
  IndexSpaceT<1> ispace = runtime->create_index_space(ctx, domain);
  FieldSpace fspace = runtime->create_field_space(ctx);
  {
    FieldAllocator fal = runtime->create_field_allocator(ctx, fspace);
    fal.allocate_field(sizeof(int), 0);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, ispace, fspace);

  std::vector<FieldID> fids = { 0 };
  AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, lr, lr);
  al.attach_file("lr.checkpoint", fids, LEGION_FILE_READ_ONLY);

  PhysicalRegion pr = runtime->attach_external_resource(ctx, al);
  pr.wait_until_valid();

  TaskLauncher launcher(1, TaskArgument());
  launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
  launcher.add_field(0, 0);
  runtime->execute_task(ctx, launcher);

  {
    Future f = runtime->detach_external_resource(ctx, pr);
    f.get_void_result(true);
  }
}

// void top_level(const Task *task,
//                const std::vector<PhysicalRegion> &regions,
//                Context ctx, Runtime *runtime_)
// {
//   using namespace ResilientLegion;
//   ResilientRuntime runtime__(runtime_);
//   ResilientRuntime *runtime = &runtime__;
//   
//   int N = 10;
//   const Rect<1> domain(0, N);
//   IndexSpaceT<1> ispace = runtime->create_index_space(ctx, domain);
//   FieldSpace fspace = runtime->create_field_space(ctx);
//   {
//     FieldAllocator fal = runtime->create_field_allocator(ctx, fspace);
//     fal.allocate_field(sizeof(int), 0);
//   }
//   LogicalRegion lr = runtime->create_logical_region(ctx, ispace, fspace);
// 
//   RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
//   req.add_field(0);
//   InlineLauncher il(req);
//   PhysicalRegion pr = runtime->map_region(ctx, il);
//   pr.wait_until_valid();
// 
//   const FieldAccessor<READ_WRITE, int, 1> acc(pr, 0);
//   for (PointInRectIterator<1> pir(domain); pir(); pir++)
//   {
//     acc[*pir] = *pir;
//   }
// 
//   runtime->unmap_region(ctx, pr);
// 
//   runtime->checkpoint(ctx);
// 
//   std::cout << "Done!" << std::endl;
// }

int main(int argc, char **argv)
{
   Runtime::set_top_level_task_id(0);
   {
     TaskVariantRegistrar registrar(0, "top_level_alt");
     registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
     Runtime::preregister_task_variant<top_level_alt>(registrar, "top_level_alt");
   }
   {
     TaskVariantRegistrar registrar(1, "check");
     registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
     Runtime::preregister_task_variant<check>(registrar, "check");
   }

//   Runtime::set_top_level_task_id(0);
//   {
//     TaskVariantRegistrar registrar(0, "top_level");
//     registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
//     Runtime::preregister_task_variant<top_level>(registrar, "top_level");
//   }
  return Runtime::start(argc, argv);
}
