#include "resilience.h"

using namespace ResilientLegion;

Runtime::Runtime(Legion::Runtime *lrt_)
  : is_checkpoint(false), future_tag(0), future_map_tag(0),
    index_space_tag(0), region_tag(0), partition_tag(0), checkpoint_tag(0), lrt(lrt_)
{
  // FIXME: This is problematic now because we are constructing this object everywhere.
  InputArgs args = Legion::Runtime::get_input_args();
  replay = false;

  bool check = false;

  for (int i = 1; i < args.argc; i++)
  {
    if (strstr(args.argv[i], "-replay"))
      replay = true;
    if (strstr(args.argv[i], "-cpt"))
    {
    /* Ideally we'd go through a preset directory to find the latest
     * checkpoint. For now we require the user to tell us which checkpoint file
     * they want to use.
     */
      check = true;
      max_checkpoint_tag = atoi(args.argv[i + 1]);
    }
  }

  if (replay)
  {
    assert(check);
    char file_name[60];
    sprintf(file_name, "checkpoint.%ld.dat", max_checkpoint_tag);
    std::ifstream file(file_name);
    cereal::XMLInputArchive iarchive(file);
    iarchive(*this);
    file.close();
  }
}

void Runtime::attach_name(FieldSpace handle, const char *name, bool is_mutable)
{
  lrt->attach_name(handle, name, is_mutable);
}

void Runtime::attach_name(FieldSpace handle, FieldID fid, const char *name, bool is_mutable)
{
  lrt->attach_name(handle, fid, name, is_mutable);
}

void Runtime::attach_name(IndexSpace handle, const char *name, bool is_mutable)
{
  lrt->attach_name(handle, name, is_mutable);
}

void Runtime::attach_name(LogicalRegion handle, const char *name, bool is_mutable)
{
  lrt->attach_name(handle, name, is_mutable);
}

void Runtime::attach_name(IndexPartition handle, const char *name, bool is_mutable)
{
  // if (replay && partition_tag < max_partition_tag)??
  for (auto &rip : partitions)
    if (rip.ip == handle && !rip.is_valid)
      return;
  lrt->attach_name(handle, name, is_mutable);
}

void Runtime::issue_execution_fence(Context ctx, const char *provenance)
{
  lrt->issue_execution_fence(ctx, provenance);
}

void callback_wrapper(const RegistrationCallbackArgs &args)
{
  auto FUNC = *static_cast<
    void (**)(Machine, Runtime *, const std::set<Processor> &)>(args.buffer.get_ptr());
  Runtime new_runtime_(args.runtime);
  Runtime *new_runtime = &new_runtime_;
  FUNC(args.machine, new_runtime, args.local_procs);
}

void Runtime::add_registration_callback(
  void (*FUNC)(Machine, Runtime *, const std::set<Processor> &),
  bool dedup, size_t dedup_tag)
{
  auto fptr = &FUNC;
  UntypedBuffer buffer(fptr, sizeof(fptr));
  Legion::Runtime::add_registration_callback(callback_wrapper, buffer, dedup, dedup_tag);
}

const InputArgs& Runtime::get_input_args(void)
{
  return Legion::Runtime::get_input_args();
}

void Runtime::set_top_level_task_id(TaskID top_id)
{
  Legion::Runtime::set_top_level_task_id(top_id);
}

LayoutConstraintID Runtime::preregister_layout(const LayoutConstraintRegistrar &registrar,
  LayoutConstraintID layout_id)
{
  return Legion::Runtime::preregister_layout(registrar, layout_id);
}

int Runtime::start(int argc, char **argv, bool background, bool supply_default_mapper)
{
  return Legion::Runtime::start(argc, argv, background, supply_default_mapper);
}

void FutureMap::wait_all_results(Runtime *runtime)
{
  /* What if this FutureMap occured after the checkpoint?! */
  if (runtime->is_checkpoint && runtime->replay)
    return;
  fm.wait_all_results();
}

FutureMap Runtime::execute_index_space(Context ctx,
  const IndexTaskLauncher &launcher)
{
  if (replay && future_map_tag < max_future_map_tag)
  {
    std::cout << "No-oping index launch\n";
    return future_maps[future_map_tag++];
  }

  Legion::FutureMap fm = lrt->execute_index_space(ctx, launcher);

  FutureMap rfm;
  if (launcher.launch_domain == Domain::NO_DOMAIN)
    rfm = FutureMap(fm, lrt->get_index_space_domain(launcher.launch_space));
  else
    rfm = FutureMap(fm, launcher.launch_domain);

  future_maps.push_back(rfm);
  future_map_tag++;
  return rfm;
}

Future Runtime::execute_task(Context ctx, TaskLauncher launcher)
{
  if (replay && future_tag < max_future_tag)
  {
    std::cout << "No-oping task.\n";
    /* It is ok to return an empty ResilentFuture because get_result knows to
     * fetch the actual result from Runtime.futures by looking at the
     * tag. get_result should never be called on an empty Future.
     */
    return futures[future_tag++];
  }
  std::cout << "Executing task.\n";
  Future ft = lrt->execute_task(ctx, launcher);
  future_tag++;
  futures.push_back(ft);
  return ft;
}

Domain Runtime::get_index_space_domain(Context ctx, IndexSpace handle)
{
  return lrt->get_index_space_domain(ctx, handle);
}

Domain Runtime::get_index_space_domain(IndexSpace handle)
{
  return lrt->get_index_space_domain(handle);
}

Future Runtime::get_current_time(
  Context ctx, Future precondition)
{
  if (replay && future_tag < max_future_tag)
  {
    /* Unlike an arbitrary task, get_current_time and friends are guaranteed to
     * return a non-void Future.
     */
    assert(!futures[future_tag].empty);
    return futures[future_tag++];
  }
  Future ft = lrt->get_current_time(ctx, precondition.lft);
  future_tag++;
  futures.push_back(ft);
  return ft;
}

Future Runtime::get_current_time_in_microseconds(
  Context ctx, Future precondition)
{
  if (replay && future_tag < max_future_tag)
  {
    assert(!futures[future_tag].empty);
    return futures[future_tag++];
  }
  Future ft = lrt->get_current_time_in_microseconds(ctx, precondition.lft);
  future_tag++;
  futures.push_back(ft);
  return ft;
}

FieldSpace Runtime::create_field_space(Context ctx)
{
  return lrt->create_field_space(ctx);
}

FieldAllocator Runtime::create_field_allocator(
  Context ctx, FieldSpace handle)
{
  return lrt->create_field_allocator(ctx, handle);
}

LogicalRegion Runtime::create_logical_region(Context ctx, IndexSpace index, FieldSpace fields, bool task_local, const char *provenance)
{
  if (replay)
  {
    /* Create empty lr from index and fields
     * Check if file corresponding to this region_tag (assuming 0 for now) exists and is non-empty.
     * Create another empty lr and attach it to the file.
     *   Since we are not returning this one, we don't need to launch a sub-task.
     * Issue a copy operation.
     * Return the first lr.
     */

    std::cout << "Reconstructing logical region from checkpoint\n";
    LogicalRegion lr = lrt->create_logical_region(ctx, index, fields);
    LogicalRegion cpy = lrt->create_logical_region(ctx, index, fields);

    /* Everything is 1-D for now */
    std::vector<FieldID> fids;
    lrt->get_field_space_fields(fields, fids);
    AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, cpy);

    char file_name[60];
    sprintf(file_name, "checkpoint.%ld.lr.%ld.dat", checkpoint_tag, region_tag++);
    al.attach_file(file_name, fids, LEGION_FILE_READ_ONLY);

    PhysicalRegion pr = lrt->attach_external_resource(ctx, al);

    CopyLauncher cl;
    cl.add_copy_requirements(RegionRequirement(cpy, READ_ONLY, EXCLUSIVE, cpy),
                             RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));

    for (auto &id : fids)
    {
      cl.add_src_field(0, id);
      cl.add_dst_field(0, id);
    }

    /* Convert to index launch */
    lrt->issue_copy_operation(ctx, cl);
    {
      Legion::Future f = lrt->detach_external_resource(ctx, pr);
      f.get_void_result(true);
    }
    return lr;
  }
  LogicalRegion lr = lrt->create_logical_region(ctx, index, fields);
  regions.push_back(lr);
  return lr;
}

/* Inline mappings need to be disallowed */
PhysicalRegion Runtime::map_region(
  Context ctx, const InlineLauncher &launcher)
{
  return lrt->map_region(ctx, launcher);
}

void Runtime::unmap_region(
  Context ctx, PhysicalRegion region)
{
  return lrt->unmap_region(ctx, region);
}

void Runtime::destroy_index_space(Context ctx, IndexSpace handle)
{
  lrt->destroy_index_space(ctx, handle);
}

void Runtime::destroy_field_space(Context ctx, FieldSpace handle)
{
  lrt->destroy_field_space(ctx, handle);
}

void Runtime::destroy_logical_region(Context ctx, LogicalRegion handle)
{
  lrt->destroy_logical_region(ctx, handle);
}

void Runtime::destroy_index_partition(Context ctx, IndexPartition handle)
{
  if (replay) return;

  for (auto &rip : partitions)
  {
    if (rip.ip == handle)
    {
      // Should not delete an already deleted partition
      assert(rip.is_valid);
      rip.is_valid = false;
      break;
    }
  }
  lrt->destroy_index_partition(ctx, handle);
}

IndexSpace Runtime::restore_index_space(Context ctx)
{
  ResilientIndexSpace ris = index_spaces[index_space_tag++];
  std::vector<Domain> rects;

  int dim = ris.get_dim();
  if (dim == 1)
  {
    for (auto &raw_rect : ris.domain.raw_rects)
    {
      Domain domain(Rect<1, long long>(raw_rect[0].x, raw_rect[1].x));
      rects.push_back(domain);
    }
  }
  else if (dim == 2)
  {
    for (auto &raw_rect : ris.domain.raw_rects)
    {
      Domain domain(Rect<2, long long>(
        {raw_rect[0].x, raw_rect[0].y},
        {raw_rect[1].x, raw_rect[1].y}));
      rects.push_back(domain);
    }
  }
  else if (dim == 3)
  {
    for (auto &raw_rect : ris.domain.raw_rects)
    {
      Domain domain(Rect<3, long long>(
        {raw_rect[0].x, raw_rect[0].y, raw_rect[0].z},
        {raw_rect[1].x, raw_rect[1].y, raw_rect[1].z}));
      rects.push_back(domain);
    }
  }
  else
    assert(false);

  IndexSpace is = lrt->create_index_space(ctx, rects);
  return is;
}

IndexSpace Runtime::create_index_space(Context ctx, const Domain &bounds)
{
  if (replay && index_space_tag < max_index_space_tag)
    restore_index_space(ctx);

  IndexSpace is = lrt->create_index_space(ctx, bounds);
  ResilientIndexSpace ris(lrt->get_index_space_domain(ctx, is));
  index_spaces.push_back(ris);
  index_space_tag++;
  return is;
}

IndexSpace Runtime::create_index_space_union(Context ctx, IndexPartition parent, const DomainPoint &color, const std::vector<IndexSpace> &handles)
{
  if (replay && index_space_tag < max_index_space_tag)
  {
    index_space_tag++;
    return lrt->get_index_subspace(ctx, parent, color);
  }

  IndexSpace is = lrt->create_index_space_union(ctx, parent, color, handles);
  ResilientIndexSpace ris(lrt->get_index_space_domain(ctx, is));
  index_spaces.push_back(ris);
  index_space_tag++;
  return is;
}

IndexSpace Runtime::create_index_space_union(Context ctx, IndexPartition parent, const DomainPoint &color, IndexPartition handle)
{
  if (replay && index_space_tag < max_index_space_tag)
  {
    index_space_tag++;
    return lrt->get_index_subspace(ctx, parent, color);
  }

  IndexSpace is = lrt->create_index_space_union(ctx, parent, color, handle);
  ResilientIndexSpace ris(lrt->get_index_space_domain(ctx, is));
  index_spaces.push_back(ris);
  index_space_tag++;
  return is;
}

IndexSpace Runtime::create_index_space_difference(Context ctx, IndexPartition parent, const DomainPoint &color, IndexSpace initial, const std::vector<IndexSpace> &handles)
{
  if (replay && index_space_tag < max_index_space_tag)
  {
    index_space_tag++;
    return lrt->get_index_subspace(ctx, parent, color);
  }

  IndexSpace is = lrt->create_index_space_difference(ctx, parent, color, initial, handles);
  ResilientIndexSpace ris(lrt->get_index_space_domain(ctx, is));
  index_spaces.push_back(ris);
  index_space_tag++;
  return is;
}

Rect<1> make_rect(std::array<ResilientDomainPoint, 2> raw_rect)
{
  Rect<1> rect(raw_rect[0].x, raw_rect[1].x);
  return rect;
}

Rect<2> make_rect_2d(std::array<ResilientDomainPoint, 2> raw_rect)
{
  Rect<2> rect(Point<2>(raw_rect[0].x, raw_rect[0].y),
    Point<2>(raw_rect[1].x, raw_rect[1].y));
  return rect;
}

Rect<3> make_rect_3d(std::array<ResilientDomainPoint, 2> raw_rect)
{
  Rect<3> rect(Point<3>(raw_rect[0].x, raw_rect[0].y, raw_rect[0].z),
    Point<3>(raw_rect[1].x, raw_rect[1].y, raw_rect[1].z));
  return rect;
}

void ResilientIndexPartition::save(Context ctx, Legion::Runtime *lrt, DomainPoint d)
{
  IndexSpace sub_is = lrt->get_index_subspace(ctx, ip, d);
  if (sub_is == IndexSpace::NO_SPACE)
    return;
  ResilientIndexSpace sub_ris(lrt->get_index_space_domain(ctx, sub_is));
  ResilientDomainPoint pt(d);
  map[pt] = sub_ris;
}

void ResilientIndexPartition::setup_for_checkpoint(Context ctx, Legion::Runtime *lrt)
{
  if (!is_valid) return;

  Domain color_domain = lrt->get_index_partition_color_space(ctx, ip);

  color_space = color_domain; /* Implicit conversion */

  /* For rect in color space
   *   For point in rect
   *     Get the index space under this point
   *     Stuff everything into a ResilientIndexPartition
   */
  int DIM = color_domain.get_dim();
  if (DIM == 1)
  {
    for (RectInDomainIterator<1> i(color_domain); i(); i++)
    {
      for (PointInRectIterator<1> j(*i); j(); j++)
        save(ctx, lrt, static_cast<DomainPoint>(*j));
    }
  }
  else if (DIM == 2)
  {
    for (RectInDomainIterator<2> i(color_domain); i(); i++)
    {
      for (PointInRectIterator<2> j(*i); j(); j++)
        save(ctx, lrt, static_cast<DomainPoint>(*j));
    }
  }
  else if (DIM == 3)
  {
    for (RectInDomainIterator<3> i(color_domain); i(); i++)
    {
      for (PointInRectIterator<3> j(*i); j(); j++)
        save(ctx, lrt, static_cast<DomainPoint>(*j));
    }
  }
  else
    assert(false);
}

IndexPartition Runtime::restore_index_partition(
  Context ctx, IndexSpace index_space, IndexSpace color_space)
{
  ResilientIndexPartition rip = partitions[partition_tag++];
  MultiDomainPointColoring *mdpc = new MultiDomainPointColoring();

  /* For rect in color space
   *   For point in rect
   *     Get the index space under this point
   *     For rect in index space
   *       Insert into mdpc at point
   */
  int DIM = color_space.get_dim();
  if (DIM == 1)
  {
    for (auto &raw_rect : rip.color_space.domain.raw_rects)
    {
      for (PointInRectIterator<1> i(make_rect(raw_rect)); i(); i++)
      {
        ResilientIndexSpace ris = rip.map[(DomainPoint) *i];
        for (auto &raw_rect_ris : ris.domain.raw_rects)
          (*mdpc)[*i].insert(make_rect(raw_rect_ris));
      }
    }
  }
  else if (DIM == 2)
  {
    for (auto &raw_rect : rip.color_space.domain.raw_rects)
    {
      for (PointInRectIterator<2> i(make_rect_2d(raw_rect)); i(); i++)
      {
        ResilientIndexSpace ris = rip.map[(DomainPoint) *i];
        for (auto &raw_rect_ris : ris.domain.raw_rects)
          (*mdpc)[*i].insert(make_rect_2d(raw_rect_ris));
      }
    }
  }
  else if (DIM == 3)
  {
    for (auto &raw_rect : rip.color_space.domain.raw_rects)
    {
      for (PointInRectIterator<3> i(make_rect_3d(raw_rect)); i(); i++)
      {
        ResilientIndexSpace ris = rip.map[(DomainPoint) *i];
        for (auto &raw_rect_ris : ris.domain.raw_rects)
          (*mdpc)[*i].insert(make_rect_3d(raw_rect_ris));
      }
    }
  }
  else
    assert(false);

  /* Assuming the domain cannot change */
  Domain color_domain = lrt->get_index_space_domain(ctx, color_space);
  IndexPartition ip = lrt->create_index_partition(ctx, index_space, color_domain, *mdpc);
  return ip;
}

IndexPartition Runtime::create_equal_partition(
  Context ctx, IndexSpace parent, IndexSpace color_space)
{
  if (replay && !partitions[partition_tag].is_valid)
  {
    partition_tag++;
    return IndexPartition::NO_PART;
  }

  if (replay && partition_tag < max_partition_tag)
    return restore_index_partition(ctx, parent, color_space);

  ResilientIndexPartition rip = lrt->create_equal_partition(ctx, parent, color_space); 
  partitions.push_back(rip);
  partition_tag++;
  return rip.ip;
}

IndexPartition Runtime::create_pending_partition(
  Context ctx, IndexSpace parent, IndexSpace color_space)
{
  if (replay && !partitions[partition_tag].is_valid)
  {
    partition_tag++;
    return IndexPartition::NO_PART;
  }

  if (replay && partition_tag < max_partition_tag)
    return restore_index_partition(ctx, parent, color_space);

  ResilientIndexPartition rip = lrt->create_pending_partition(ctx, parent, color_space); 
  partitions.push_back(rip);
  partition_tag++;
  return rip.ip;
}

IndexPartition Runtime::create_partition_by_field(Context ctx,
  LogicalRegion handle, LogicalRegion parent, FieldID fid, IndexSpace color_space)
{
  if (replay && !partitions[partition_tag].is_valid)
  {
    partition_tag++;
    return IndexPartition::NO_PART;
  }

  if (replay && partition_tag < max_partition_tag)
    return restore_index_partition(ctx, handle.get_index_space(), color_space);

  ResilientIndexPartition rip = lrt->create_partition_by_field(ctx, handle, parent, fid, color_space);
  partitions.push_back(rip);
  partition_tag++;
  return rip.ip;
}

IndexPartition Runtime::create_partition_by_image(
  Context ctx, IndexSpace handle, LogicalPartition projection,
  LogicalRegion parent, FieldID fid, IndexSpace color_space)
{
  if (replay && !partitions[partition_tag].is_valid)
  {
    partition_tag++;
    return IndexPartition::NO_PART;
  }

  if (replay && partition_tag < max_partition_tag)
    return restore_index_partition(ctx, handle, color_space);

  ResilientIndexPartition rip = lrt->create_partition_by_image(ctx, handle, projection, parent, fid, color_space);
  partitions.push_back(rip);
  partition_tag++;
  return rip.ip;
}

IndexPartition Runtime::create_partition_by_preimage(
  Context ctx, IndexPartition projection, LogicalRegion handle,
  LogicalRegion parent, FieldID fid, IndexSpace color_space)
{
  if (replay && !partitions[partition_tag].is_valid)
  {
    partition_tag++;
    return IndexPartition::NO_PART;
  }

  if (replay && partition_tag < max_partition_tag)
    return restore_index_partition(ctx, handle.get_index_space(), color_space);

  ResilientIndexPartition rip = lrt->create_partition_by_preimage(ctx, projection, handle, parent, fid, color_space);
  partitions.push_back(rip);
  partition_tag++;
  return rip.ip;
}

IndexPartition Runtime::create_partition_by_difference(
  Context ctx, IndexSpace parent, IndexPartition handle1,
  IndexPartition handle2, IndexSpace color_space)
{
  if (replay && !partitions[partition_tag].is_valid)
  {
    partition_tag++;
    return IndexPartition::NO_PART;
  }

  if (replay && partition_tag < max_partition_tag)
    return restore_index_partition(ctx, parent, color_space);

  ResilientIndexPartition rip = lrt->create_partition_by_difference(ctx, parent, handle1, handle2, color_space);
  partitions.push_back(rip);
  partition_tag++;
  return rip.ip;
}

Color Runtime::create_cross_product_partitions(Context ctx, IndexPartition handle1, IndexPartition handle2, std::map<IndexSpace, IndexPartition> &handles)
{
  return lrt->create_cross_product_partitions(ctx, handle1, handle2, handles);
}

LogicalPartition Runtime::get_logical_partition(
  Context ctx, LogicalRegion parent, IndexPartition handle)
{
  return lrt->get_logical_partition(ctx, parent, handle);
}

LogicalPartition Runtime::get_logical_partition(
  LogicalRegion parent, IndexPartition handle)
{
  return lrt->get_logical_partition(parent, handle);
}

LogicalPartition Runtime::get_logical_partition_by_tree(IndexPartition handle, FieldSpace fspace, RegionTreeID tid)
{
  return lrt->get_logical_partition_by_tree(handle, fspace, tid);
}

LogicalRegion Runtime::get_logical_subregion_by_color(
  Context ctx, LogicalPartition parent, Color c)
{
  return lrt->get_logical_subregion_by_color(ctx, parent, c);
}

LogicalRegion Runtime::get_logical_subregion_by_color(
  Context ctx, LogicalPartition parent, DomainPoint c)
{
  return lrt->get_logical_subregion_by_color(ctx, parent, c);
}

LogicalRegion Runtime::get_logical_subregion_by_color(
  LogicalPartition parent, const DomainPoint &c)
{
  return lrt->get_logical_subregion_by_color(parent, c);
}

Legion::Mapping::MapperRuntime* Runtime::get_mapper_runtime(void)
{
  return lrt->get_mapper_runtime();
}

void Runtime::replace_default_mapper(Legion::Mapping::Mapper *mapper, Processor proc)
{
  lrt->replace_default_mapper(mapper, proc);
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

ptr_t Runtime::safe_cast(Context ctx, ptr_t pointer, LogicalRegion region)
{
  return lrt->safe_cast(ctx, pointer, region);
}

DomainPoint Runtime::safe_cast(Context ctx, DomainPoint point, LogicalRegion region)
{
  return lrt->safe_cast(ctx, point, region);
}

void Runtime::save_logical_region(
  Context ctx, const Task *task, LogicalRegion &lr, const char *file_name)
{
  bool ok = generate_disk_file(file_name);
  assert(ok);

  LogicalRegion cpy = lrt->create_logical_region(ctx,
                        lr.get_index_space(), lr.get_field_space());

  std::vector<FieldID> fids;
  lrt->get_field_space_fields(lr.get_field_space(), fids);

  AttachLauncher al(LEGION_EXTERNAL_POSIX_FILE, cpy, cpy, false);
  al.attach_file(file_name, fids, LEGION_FILE_READ_WRITE);

  PhysicalRegion pr = lrt->attach_external_resource(ctx, al);

  CopyLauncher cl;
  cl.add_copy_requirements(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr),
                           RegionRequirement(cpy, READ_WRITE, EXCLUSIVE, cpy));

  for (long unsigned i = 0; i < fids.size(); i++)
  {
    if (i % task->get_total_shards() == task->get_shard_id())
    {
      cl.add_src_field(0, fids[i]);
      cl.add_dst_field(0, fids[i]);
    }
  }

  // Index launch this?
  lrt->issue_copy_operation(ctx, cl);

  {
    Legion::Future f = lrt->detach_external_resource(ctx, pr);
    f.get_void_result(true);
  }
}

void resilient_write(const Task *task,
  const std::vector<PhysicalRegion> &regions, Context ctx, Legion::Runtime *runtime)
{
  const char *cstr = static_cast<char *>(task->args);
  std::string str(cstr, task->arglen);
  std::string tag = str.substr(0, str.find(","));
  std::string file_name = "checkpoint." + tag;
  file_name += ".dat";
  std::cout << "File name is " << file_name << std::endl;
  std::ofstream file(file_name);
  file << str.substr(str.find(",") + 1, str.size());
  file.close();
}

void Runtime::checkpoint(Context ctx, const Task *task)
{
  if (replay) return;

  std::cout << "In checkpoint " << checkpoint_tag << std::endl;

  char file_name[60];
  int counter = 0;
  for (auto &lr : regions)
  {
    sprintf(file_name, "checkpoint.%ld.lr.%d.dat", checkpoint_tag, counter++);
    save_logical_region(ctx, task, lr, file_name);
  }

  max_future_tag = future_tag;
  max_future_map_tag = future_map_tag;
  max_index_space_tag = index_space_tag;
  max_partition_tag = partition_tag;
  max_checkpoint_tag = checkpoint_tag;

  for (auto &ft : futures)
    ft.setup_for_checkpoint();

  for (auto &fm : future_maps)
    fm.setup_for_checkpoint();

  // Do not need to setup index spaces

  for (auto &ip : partitions)
    ip.setup_for_checkpoint(ctx, lrt);

  std::stringstream serialized;
  {
    cereal::XMLOutputArchive oarchive(serialized);
    oarchive(*this);
  }
  std::string tmp = std::to_string(checkpoint_tag) + ",";
  tmp += serialized.str();
  const char *cstr = tmp.c_str();

  TaskID tid = lrt->generate_dynamic_task_id();
  {
    TaskVariantRegistrar registrar(tid, "resilient_write");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    lrt->register_task_variant<resilient_write>(registrar);
  }
  TaskLauncher resilient_write_launcher(tid, TaskArgument(cstr, strlen(cstr)));
  lrt->execute_task(ctx, resilient_write_launcher);

  checkpoint_tag++;
}

void Runtime::enable_checkpointing()
{
  is_checkpoint = true;
}
