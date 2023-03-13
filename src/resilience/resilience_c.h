/* Copyright 2023 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef __RESILIENCE_C_H__
#define __RESILIENCE_C_H__

/**
 * \file resilience_c.h
 * Legion C API
 */

// ******************** IMPORTANT **************************
//
// This file is PURE C, **NOT** C++.
//
// ******************** IMPORTANT **************************

#include "legion.h"
#include "legion/legion_c.h"

#include <stdbool.h>
#ifndef LEGION_USE_PYTHON_CFFI
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#endif // LEGION_USE_PYTHON_CFFI

#ifdef __cplusplus
extern "C" {
#endif

  // Types from legion_config.h
  typedef legion_address_space_t resilient_legion_address_space_t;
  typedef legion_allocate_mode_t resilient_legion_allocate_mode_t;
  typedef legion_coherence_property_t resilient_legion_coherence_property_t;
  typedef legion_color_t resilient_legion_color_t;
  typedef legion_custom_serdez_id_t resilient_legion_custom_serdez_id_t;
  typedef legion_dimension_kind_t resilient_legion_dimension_kind_t;
  typedef legion_equality_kind_t resilient_legion_equality_kind_t;
  typedef legion_external_resource_t resilient_legion_external_resource_t;
  typedef legion_field_id_t resilient_legion_field_id_t;
  typedef legion_field_space_id_t resilient_legion_field_space_id_t;
  typedef legion_file_mode_t resilient_legion_file_mode_t;
  typedef legion_garbage_collection_priority_t resilient_legion_garbage_collection_priority_t;
  typedef legion_handle_type_t resilient_legion_handle_type_t;
  typedef legion_index_partition_id_t resilient_legion_index_partition_id_t;
  typedef legion_index_space_id_t resilient_legion_index_space_id_t;
  typedef legion_index_tree_id_t resilient_legion_index_tree_id_t;
  typedef legion_launch_constraint_t resilient_legion_launch_constraint_t;
  typedef legion_layout_constraint_id_t resilient_legion_layout_constraint_id_t;
  typedef enum legion_mappable_type_id_t resilient_legion_mappable_type_id_t;
  typedef legion_mapper_id_t resilient_legion_mapper_id_t;
  typedef legion_mapping_tag_id_t resilient_legion_mapping_tag_id_t;
  typedef legion_memory_kind_t resilient_legion_memory_kind_t;
  typedef legion_partition_kind_t resilient_legion_partition_kind_t;
  typedef legion_privilege_mode_t resilient_legion_privilege_mode_t;
  typedef legion_proc_id_t resilient_legion_proc_id_t;
  typedef legion_processor_kind_t resilient_legion_processor_kind_t;
  typedef legion_projection_id_t resilient_legion_projection_id_t;
  typedef legion_reduction_op_id_t resilient_legion_reduction_op_id_t;
  typedef legion_region_flags_t resilient_legion_region_flags_t;
  typedef legion_region_tree_id_t resilient_legion_region_tree_id_t;
  typedef legion_resource_constraint_t resilient_legion_resource_constraint_t;
  typedef legion_semantic_tag_t resilient_legion_semantic_tag_t;
  typedef legion_shard_id_t resilient_legion_shard_id_t;
  typedef legion_sharding_id_t resilient_legion_sharding_id_t;
  typedef legion_specialized_constraint_t resilient_legion_specialized_constraint_t;
  typedef legion_task_id_t resilient_legion_task_id_t;
  typedef legion_task_priority_t resilient_legion_task_priority_t;
  typedef legion_trace_id_t resilient_legion_trace_id_t;
  typedef legion_tunable_id_t resilient_legion_tunable_id_t;
  typedef legion_type_tag_t resilient_legion_type_tag_t;
  typedef legion_unique_id_t resilient_legion_unique_id_t;
  typedef legion_variant_id_t resilient_legion_variant_id_t;

  // Types from legion_c.h

  // Opaque types
  typedef legion_context_t resilient_legion_context_t;
  typedef legion_domain_point_iterator_t resilient_legion_domain_point_iterator_t;
#define NEW_ITERATOR_TYPE(DIM) \
  typedef legion_rect_in_domain_iterator_##DIM##d_t resilient_legion_rect_in_domain_iterator_##DIM##d_t;
  LEGION_FOREACH_N(NEW_ITERATOR_TYPE);
#undef NEW_ITERATOR_TYPE
  typedef legion_coloring_t resilient_legion_coloring_t;
  typedef legion_domain_coloring_t resilient_legion_domain_coloring_t;
  typedef legion_point_coloring_t resilient_legion_point_coloring_t;
  typedef legion_domain_point_coloring_t resilient_legion_domain_point_coloring_t;
  typedef legion_multi_domain_point_coloring_t resilient_legion_multi_domain_point_coloring_t;
  typedef legion_index_space_allocator_t resilient_legion_index_space_allocator_t;
  typedef legion_field_allocator_t resilient_legion_field_allocator_t;
  typedef legion_argument_map_t resilient_legion_argument_map_t;
  typedef legion_predicate_t resilient_legion_predicate_t;
#define NEW_DEFERRED_BUFFER_TYPE(DIM) \
  typedef legion_deferred_buffer_char_##DIM##d_t resilient_legion_deferred_buffer_char_##DIM##d_t;
  LEGION_FOREACH_N(NEW_DEFERRED_BUFFER_TYPE)
#undef NEW_DEFERRED_BUFFER_TYPE
  typedef legion_inline_launcher_t resilient_legion_inline_launcher_t;
  typedef legion_copy_launcher_t resilient_legion_copy_launcher_t;
  typedef legion_index_copy_launcher_t resilient_legion_index_copy_launcher_t;
  typedef legion_discard_launcher_t resilient_legion_discard_launcher_t;
  typedef legion_acquire_launcher_t resilient_legion_acquire_launcher_t;
  typedef legion_release_launcher_t resilient_legion_release_launcher_t;
  typedef legion_attach_launcher_t resilient_legion_attach_launcher_t;
  typedef legion_index_attach_launcher_t resilient_legion_index_attach_launcher_t;
  typedef legion_must_epoch_launcher_t resilient_legion_must_epoch_launcher_t;
  typedef legion_physical_region_t resilient_legion_physical_region_t;
  typedef legion_external_resources_t resilient_legion_external_resources_t;
#define NEW_ACCESSOR_ARRAY_TYPE(DIM) \
  typedef legion_accessor_array_##DIM##d_t resilient_legion_accessor_array_##DIM##d_t;
  LEGION_FOREACH_N(NEW_ACCESSOR_ARRAY_TYPE)
#undef NEW_ACCESSOR_ARRAY_TYPE
  typedef legion_task_t resilient_legion_task_t;
  typedef legion_task_mut_t resilient_legion_task_mut_t;
  typedef legion_copy_t resilient_legion_copy_t;
  typedef legion_fill_t resilient_legion_fill_t;
  typedef legion_inline_t resilient_legion_inline_t;
  typedef legion_mappable_t resilient_legion_mappable_t;
  typedef legion_region_requirement_t resilient_legion_region_requirement_t;
  typedef legion_output_requirement_t resilient_legion_output_requirement_t;
  typedef legion_machine_t resilient_legion_machine_t;
  typedef legion_mapper_t resilient_legion_mapper_t;
  typedef legion_default_mapper_t resilient_legion_default_mapper_t;
  typedef legion_processor_query_t resilient_legion_processor_query_t;
  typedef legion_memory_query_t resilient_legion_memory_query_t;
  typedef legion_machine_query_interface_t resilient_legion_machine_query_interface_t;
  typedef legion_execution_constraint_set_t resilient_legion_execution_constraint_set_t;
  typedef legion_layout_constraint_set_t resilient_legion_layout_constraint_set_t;
  typedef legion_task_layout_constraint_set_t resilient_legion_task_layout_constraint_set_t;
  typedef legion_slice_task_output_t resilient_legion_slice_task_output_t;
  typedef legion_map_task_input_t resilient_legion_map_task_input_t;
  typedef legion_map_task_output_t resilient_legion_map_task_output_t;
  typedef legion_physical_instance_t resilient_legion_physical_instance_t;
  typedef legion_mapper_runtime_t resilient_legion_mapper_runtime_t;
  typedef legion_mapper_context_t resilient_legion_mapper_context_t;
  typedef legion_field_map_t resilient_legion_field_map_t;
  typedef legion_point_transform_functor_t resilient_legion_point_transform_functor_t;

  // Non-opaque types
  typedef legion_ptr_t resilient_legion_ptr_t;

#define NEW_POINT_TYPE(DIM) typedef legion_point_##DIM##d_t resilient_legion_point_##DIM##d_t;
  LEGION_FOREACH_N(NEW_POINT_TYPE)
#undef NEW_POINT_TYPE

#define NEW_RECT_TYPE(DIM) typedef legion_rect_##DIM##d_t resilient_legion_rect_##DIM##d_t;
  LEGION_FOREACH_N(NEW_RECT_TYPE)
#undef NEW_RECT_TYPE

#define NEW_BLOCKIFY_TYPE(DIM) \
  typedef legion_blockify_##DIM##d_t  resilient_legion_blockify_##DIM##d_t;
  LEGION_FOREACH_N(NEW_BLOCKIFY_TYPE)
#undef NEW_BLOCKIFY_TYPE

#define NEW_TRANSFORM_TYPE(D1,D2) \
  typedef legion_transform_##D1##x##D2##_t resilient_legion_transform_##D1##x##D2##_t;
  LEGION_FOREACH_NN(NEW_TRANSFORM_TYPE)
#undef NEW_TRANSFORM_TYPE

#define NEW_AFFINE_TRANSFORM_TYPE(D1,D2) \
  typedef legion_affine_transform_##D1##x##D2##_t resilient_legion_affine_transform_##D1##x##D2##_t;
  LEGION_FOREACH_NN(NEW_AFFINE_TRANSFORM_TYPE)
#undef NEW_AFFINE_TRANSFORM_TYPE

  typedef legion_domain_t resilient_legion_domain_t;
  typedef legion_domain_point_t resilient_legion_domain_point_t;
  typedef legion_domain_transform_t resilient_legion_domain_transform_t;
  typedef legion_domain_affine_transform_t resilient_legion_domain_affine_transform_t;
  typedef legion_index_space_t resilient_legion_index_space_t;
  typedef legion_index_partition_t resilient_legion_index_partition_t;
  typedef legion_field_space_t resilient_legion_field_space_t;
  typedef legion_logical_region_t resilient_legion_logical_region_t;
  typedef legion_logical_partition_t resilient_legion_logical_partition_t;
  typedef legion_untyped_buffer_t resilient_legion_untyped_buffer_t;
  typedef legion_task_argument_t resilient_legion_task_argument_t;
  typedef legion_byte_offset_t resilient_legion_byte_offset_t;
  typedef legion_input_args_t resilient_legion_input_args_t;
  typedef legion_task_config_options_t resilient_legion_task_config_options_t;
  typedef legion_processor_t resilient_legion_processor_t;
  typedef legion_memory_t resilient_legion_memory_t;
  typedef legion_task_slice_t resilient_legion_task_slice_t;
  typedef legion_phase_barrier_t resilient_legion_phase_barrier_t;
  typedef legion_dynamic_collective_t resilient_legion_dynamic_collective_t;
  typedef legion_task_options_t resilient_legion_task_options_t;
  typedef legion_slice_task_input_t resilient_legion_slice_task_input_t;

  // -----------------------------------------------------------------------
  // Proxy Types
  // -----------------------------------------------------------------------

// #define NEW_OPAQUE_TYPE(T) typedef void * T
#define NEW_OPAQUE_TYPE(T) typedef struct T { void *impl; } T
  NEW_OPAQUE_TYPE(resilient_legion_runtime_t);
  NEW_OPAQUE_TYPE(resilient_legion_future_t);
  NEW_OPAQUE_TYPE(resilient_legion_future_map_t);
  NEW_OPAQUE_TYPE(resilient_legion_task_launcher_t);
  NEW_OPAQUE_TYPE(resilient_legion_index_launcher_t);
  NEW_OPAQUE_TYPE(resilient_legion_fill_launcher_t);
  NEW_OPAQUE_TYPE(resilient_legion_index_fill_launcher_t);
#undef NEW_OPAQUE_TYPE

  /**
   * Interface for a Legion C registration callback.
   */
  typedef
    void (*resilient_legion_registration_callback_pointer_t)(
      resilient_legion_machine_t /* machine */,
      resilient_legion_runtime_t /* runtime */,
      const resilient_legion_processor_t * /* local_procs */,
      unsigned /* num_local_procs */);

  /**
   * Interface for a Legion C task that is wrapped (i.e. this is the Realm
   * task interface)
   */
  typedef realm_task_pointer_t resilient_legion_task_pointer_wrapped_t;

  /**
   * Interface for a Legion C projection functor (Logical Region
   * upper bound).
   */
  typedef
    resilient_legion_logical_region_t (*resilient_legion_projection_functor_logical_region_t)(
      resilient_legion_runtime_t /* runtime */,
      resilient_legion_logical_region_t /* upper_bound */,
      resilient_legion_domain_point_t /* point */,
      resilient_legion_domain_t /* launch domain */);

  /**
   * Interface for a Legion C projection functor (Logical Partition
   * upper bound).
   */
  typedef
    resilient_legion_logical_region_t (*resilient_legion_projection_functor_logical_partition_t)(
      resilient_legion_runtime_t /* runtime */,
      resilient_legion_logical_partition_t /* upper_bound */,
      resilient_legion_domain_point_t /* point */,
      resilient_legion_domain_t /* launch domain */);

  /**
   * Interface for a Legion C projection functor (Logical Region
   * upper bound).
   */
  typedef
    resilient_legion_logical_region_t (*resilient_legion_projection_functor_logical_region_mappable_t)(
      resilient_legion_runtime_t /* runtime */,
      resilient_legion_mappable_t /* mappable */,
      unsigned /* index */,
      resilient_legion_logical_region_t /* upper_bound */,
      resilient_legion_domain_point_t /* point */);

  /**
   * Interface for a Legion C projection functor (Logical Partition
   * upper bound).
   */
  typedef
    resilient_legion_logical_region_t (*resilient_legion_projection_functor_logical_partition_mappable_t)(
      resilient_legion_runtime_t /* runtime */,
      resilient_legion_mappable_t /* mappable */,
      unsigned /* index */,
      resilient_legion_logical_partition_t /* upper_bound */,
      resilient_legion_domain_point_t /* point */);

  // -----------------------------------------------------------------------
  // Pointer Operations
  // -----------------------------------------------------------------------

  /**
   * @see ptr_t::nil()
   */
  resilient_legion_ptr_t
  resilient_legion_ptr_nil(void);

  /**
   * @see ptr_t::is_null()
   */
  bool
  resilient_legion_ptr_is_null(resilient_legion_ptr_t ptr);

  /**
   * @see Legion::Runtime::safe_cast(
   *        Context, ptr_t, LogicalRegion)
   */
  resilient_legion_ptr_t
  resilient_legion_ptr_safe_cast(resilient_legion_runtime_t runtime,
                       resilient_legion_context_t ctx,
                       resilient_legion_ptr_t pointer,
                       resilient_legion_logical_region_t region);

  // -----------------------------------------------------------------------
  // Domain Operations
  // -----------------------------------------------------------------------
  
  /**
   * @see Legion::Domain::Domain()
   */
  resilient_legion_domain_t
  resilient_legion_domain_empty(unsigned dim);

  /**
   * @see Legion::Domain::from_rect()
   */
#define FROM_RECT(DIM) \
  resilient_legion_domain_t \
  resilient_legion_domain_from_rect_##DIM##d(resilient_legion_rect_##DIM##d_t r);
  LEGION_FOREACH_N(FROM_RECT)
#undef FROM_RECT

  /**
   * @see Legion::Domain::Domain(Legion::IndexSpace)
   */
  resilient_legion_domain_t
  resilient_legion_domain_from_index_space(resilient_legion_runtime_t runtime,
                                 resilient_legion_index_space_t is);

  /**
   * @see Legion::Domain::get_rect()
   */
#define GET_RECT(DIM) \
  resilient_legion_rect_##DIM##d_t \
  resilient_legion_domain_get_rect_##DIM##d(resilient_legion_domain_t d);
  LEGION_FOREACH_N(GET_RECT)
#undef GET_RECT

  bool
  resilient_legion_domain_is_dense(resilient_legion_domain_t d);

  // These are the same as above but will ignore 
  // the existence of any sparsity map, whereas the 
  // ones above will fail if a sparsity map exists
#define GET_BOUNDS(DIM) \
  resilient_legion_rect_##DIM##d_t \
  resilient_legion_domain_get_bounds_##DIM##d(resilient_legion_domain_t d);
  LEGION_FOREACH_N(GET_BOUNDS)
#undef GET_BOUNDS

  /**
   * @see Legion::Domain::contains()
   */
  bool
  resilient_legion_domain_contains(resilient_legion_domain_t d, resilient_legion_domain_point_t p);

  /**
   * @see Legion::Domain::get_volume()
   */
  size_t
  resilient_legion_domain_get_volume(resilient_legion_domain_t d);

  // -----------------------------------------------------------------------
  // Domain Transform Operations
  // -----------------------------------------------------------------------
  
  resilient_legion_domain_transform_t
  resilient_legion_domain_transform_identity(unsigned m, unsigned n);

#define FROM_TRANSFORM(D1,D2) \
  resilient_legion_domain_transform_t \
  resilient_legion_domain_transform_from_##D1##x##D2(resilient_legion_transform_##D1##x##D2##_t t);
  LEGION_FOREACH_NN(FROM_TRANSFORM)
#undef FROM_TRANSFORM

  resilient_legion_domain_affine_transform_t
  resilient_legion_domain_affine_transform_identity(unsigned m, unsigned n);

#define FROM_AFFINE(D1,D2) \
  resilient_legion_domain_affine_transform_t \
  resilient_legion_domain_affine_transform_from_##D1##x##D2(resilient_legion_affine_transform_##D1##x##D2##_t t);
  LEGION_FOREACH_NN(FROM_AFFINE)
#undef FROM_AFFINE

  // -----------------------------------------------------------------------
  // Domain Point Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::DomainPoint::from_point()
   */
#define FROM_POINT(DIM) \
  resilient_legion_domain_point_t \
  resilient_legion_domain_point_from_point_##DIM##d(resilient_legion_point_##DIM##d_t p);
  LEGION_FOREACH_N(FROM_POINT)
#undef FROM_POINT

  /**
   * @see Legion::DomainPoint::get_point()
   */
#define GET_POINT(DIM) \
  resilient_legion_point_##DIM##d_t \
  resilient_legion_domain_point_get_point_##DIM##d(resilient_legion_domain_point_t p);
  LEGION_FOREACH_N(GET_POINT)
#undef GET_POINT

  resilient_legion_domain_point_t
  resilient_legion_domain_point_origin(unsigned dim);

  /**
   * @see Legion::DomainPoint::nil()
   */
  resilient_legion_domain_point_t
  resilient_legion_domain_point_nil(void);

  /**
   * @see Legion::DomainPoint::is_null()
   */
  bool
  resilient_legion_domain_point_is_null(resilient_legion_domain_point_t point);

  /**
   * @see Legion::Runtime::safe_cast(
   *        Context, DomainPoint, LogicalRegion)
   */
  resilient_legion_domain_point_t
  resilient_legion_domain_point_safe_cast(resilient_legion_runtime_t runtime,
                                resilient_legion_context_t ctx,
                                resilient_legion_domain_point_t point,
                                resilient_legion_logical_region_t region);

  // -----------------------------------------------------------------------
  // Domain Point Iterator
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Domain::DomainPointIterator::DomainPointIterator()
   */
  resilient_legion_domain_point_iterator_t
  resilient_legion_domain_point_iterator_create(resilient_legion_domain_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Domain::DomainPointIterator::~DomainPointIterator()
   */
  void
  resilient_legion_domain_point_iterator_destroy(resilient_legion_domain_point_iterator_t handle);

  /**
   * @see Legion::Domain::DomainPointIterator::any_left
   */
  bool
  resilient_legion_domain_point_iterator_has_next(resilient_legion_domain_point_iterator_t handle);

  /**
   * @see Legion::Domain::DomainPointIterator::step()
   */
  resilient_legion_domain_point_t
  resilient_legion_domain_point_iterator_next(resilient_legion_domain_point_iterator_t handle);

  // -----------------------------------------------------------------------
  // Rect in Domain Iterator
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Domain::RectInDomainIterator::RectInDomainIterator()
   */
#define ITERATOR_CREATE(DIM) \
  resilient_legion_rect_in_domain_iterator_##DIM##d_t \
  resilient_legion_rect_in_domain_iterator_create_##DIM##d(resilient_legion_domain_t handle);
  LEGION_FOREACH_N(ITERATOR_CREATE)
#undef ITERATOR_CREATE

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Domain::RectInDomainIterator::~RectInDomainIterator()
   */
#define ITERATOR_DESTROY(DIM) \
  void resilient_legion_rect_in_domain_iterator_destroy_##DIM##d( \
        resilient_legion_rect_in_domain_iterator_##DIM##d_t handle);
  LEGION_FOREACH_N(ITERATOR_DESTROY)
#undef ITERATOR_DESTROY

  /**
   * @see Legion::Domain::RectInDomainIterator::valid()
   */
#define ITERATOR_VALID(DIM) \
  bool resilient_legion_rect_in_domain_iterator_valid_##DIM##d( \
        resilient_legion_rect_in_domain_iterator_##DIM##d_t handle);
  LEGION_FOREACH_N(ITERATOR_VALID)
#undef ITERATOR_VALID

  /**
   * @see Legion::Domain::RectInDomainIterator::step()
   */
#define ITERATOR_STEP(DIM) \
  bool resilient_legion_rect_in_domain_iterator_step_##DIM##d( \
        resilient_legion_rect_in_domain_iterator_##DIM##d_t handle);
  LEGION_FOREACH_N(ITERATOR_STEP)
#undef ITERATOR_STEP

  /**
   * @see Legion::Domain::RectInDomainIterator::operator*()
   */
#define ITERATOR_OP(DIM) \
  resilient_legion_rect_##DIM##d_t \
  resilient_legion_rect_in_domain_iterator_get_rect_##DIM##d( \
      resilient_legion_rect_in_domain_iterator_##DIM##d_t handle);
  LEGION_FOREACH_N(ITERATOR_OP)
#undef ITERATOR_OP

  // -----------------------------------------------------------------------
  // Coloring Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Coloring
   */
  resilient_legion_coloring_t
  resilient_legion_coloring_create(void);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Coloring
   */
  void
  resilient_legion_coloring_destroy(resilient_legion_coloring_t handle);

  /**
   * @see Legion::Coloring
   */
  void
  resilient_legion_coloring_ensure_color(resilient_legion_coloring_t handle,
                               resilient_legion_color_t color);

  /**
   * @see Legion::Coloring
   */
  void
  resilient_legion_coloring_add_point(resilient_legion_coloring_t handle,
                            resilient_legion_color_t color,
                            resilient_legion_ptr_t point);

  /**
   * @see Legion::Coloring
   */
  void
  resilient_legion_coloring_delete_point(resilient_legion_coloring_t handle,
                               resilient_legion_color_t color,
                               resilient_legion_ptr_t point);

  /**
   * @see Legion::Coloring
   */
  bool
  resilient_legion_coloring_has_point(resilient_legion_coloring_t handle,
                            resilient_legion_color_t color,
                            resilient_legion_ptr_t point);

  /**
   * @see Legion::Coloring
   */
  void
  resilient_legion_coloring_add_range(resilient_legion_coloring_t handle,
                            resilient_legion_color_t color,
                            resilient_legion_ptr_t start,
                            resilient_legion_ptr_t end /**< inclusive */);

  // -----------------------------------------------------------------------
  // Domain Coloring Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::DomainColoring
   */
  resilient_legion_domain_coloring_t
  resilient_legion_domain_coloring_create(void);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::DomainColoring
   */
  void
  resilient_legion_domain_coloring_destroy(resilient_legion_domain_coloring_t handle);

  /**
   * @see Legion::DomainColoring
   */
  void
  resilient_legion_domain_coloring_color_domain(resilient_legion_domain_coloring_t handle,
                                      resilient_legion_color_t color,
                                      resilient_legion_domain_t domain);

  /**
   * @see Legion::DomainColoring
   */
  resilient_legion_domain_t
  resilient_legion_domain_coloring_get_color_space(resilient_legion_domain_coloring_t handle);

  // -----------------------------------------------------------------------
  // Point Coloring Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::PointColoring
   */
  resilient_legion_point_coloring_t
  resilient_legion_point_coloring_create(void);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::PointColoring
   */
  void
  resilient_legion_point_coloring_destroy(
    resilient_legion_point_coloring_t handle);

  /**
   * @see Legion::PointColoring
   */
  void
  resilient_legion_point_coloring_add_point(resilient_legion_point_coloring_t handle,
                                  resilient_legion_domain_point_t color,
                                  resilient_legion_ptr_t point);

  /**
   * @see Legion::PointColoring
   */
  void
  resilient_legion_point_coloring_add_range(resilient_legion_point_coloring_t handle,
                                  resilient_legion_domain_point_t color,
                                  resilient_legion_ptr_t start,
                                  resilient_legion_ptr_t end /**< inclusive */);

  // -----------------------------------------------------------------------
  // Domain Point Coloring Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::DomainPointColoring
   */
  resilient_legion_domain_point_coloring_t
  resilient_legion_domain_point_coloring_create(void);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::DomainPointColoring
   */
  void
  resilient_legion_domain_point_coloring_destroy(
    resilient_legion_domain_point_coloring_t handle);

  /**
   * @see Legion::DomainPointColoring
   */
  void
  resilient_legion_domain_point_coloring_color_domain(
    resilient_legion_domain_point_coloring_t handle,
    resilient_legion_domain_point_t color,
    resilient_legion_domain_t domain);

  // -----------------------------------------------------------------------
  // Multi-Domain Point Coloring Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::MultiDomainPointColoring
   */
  resilient_legion_multi_domain_point_coloring_t
  resilient_legion_multi_domain_point_coloring_create(void);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::MultiDomainPointColoring
   */
  void
  resilient_legion_multi_domain_point_coloring_destroy(
    resilient_legion_multi_domain_point_coloring_t handle);

  /**
   * @see Legion::MultiDomainPointColoring
   */
  void
  resilient_legion_multi_domain_point_coloring_color_domain(
    resilient_legion_multi_domain_point_coloring_t handle,
    resilient_legion_domain_point_t color,
    resilient_legion_domain_t domain);

  // -----------------------------------------------------------------------
  // Index Space Operations
  // ----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_space(Context, size_t)
   */
  resilient_legion_index_space_t
  resilient_legion_index_space_create(resilient_legion_runtime_t runtime,
                            resilient_legion_context_t ctx,
                            size_t max_num_elmts);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_space(Context, Domain)
   */
  resilient_legion_index_space_t
  resilient_legion_index_space_create_domain(resilient_legion_runtime_t runtime,
                                   resilient_legion_context_t ctx,
                                   resilient_legion_domain_t domain);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::create_index_space(Context, size_t, Future, TypeTag)
   */
  resilient_legion_index_space_t
  resilient_legion_index_space_create_future(resilient_legion_runtime_t runtime,
                                   resilient_legion_context_t ctx,
                                   size_t dimensions,
                                   resilient_legion_future_t future,
                                   resilient_legion_type_tag_t type_tag/*=0*/);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::union_index_spaces
   */
  resilient_legion_index_space_t
  resilient_legion_index_space_union(resilient_legion_runtime_t runtime,
                           resilient_legion_context_t ctx,
                           const resilient_legion_index_space_t *spaces,
                           size_t num_spaces);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::intersect_index_spaces
   */
  resilient_legion_index_space_t
  resilient_legion_index_space_intersection(resilient_legion_runtime_t runtime,
                                  resilient_legion_context_t ctx,
                                  const resilient_legion_index_space_t *spaces,
                                  size_t num_spaces);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::subtract_index_spaces
   */
  resilient_legion_index_space_t
  resilient_legion_index_space_subtraction(resilient_legion_runtime_t runtime,
                                 resilient_legion_context_t ctx,
                                 resilient_legion_index_space_t left,
                                 resilient_legion_index_space_t right);

  /**
   * @see Legion::Runtime::has_multiple_domains().
   */
  bool
  resilient_legion_index_space_has_multiple_domains(resilient_legion_runtime_t runtime,
                                          resilient_legion_index_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::get_index_space_domain()
   */
  resilient_legion_domain_t
  resilient_legion_index_space_get_domain(resilient_legion_runtime_t runtime,
                                resilient_legion_index_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::has_parent_index_partition()
   */
  bool
  resilient_legion_index_space_has_parent_index_partition(resilient_legion_runtime_t runtime,
                                                resilient_legion_index_space_t handle);
  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::get_parent_index_partition()
   */
  resilient_legion_index_partition_t
  resilient_legion_index_space_get_parent_index_partition(resilient_legion_runtime_t runtime,
                                                resilient_legion_index_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::create_shared_ownership
   */
  void
  resilient_legion_index_space_create_shared_ownership(resilient_legion_runtime_t runtime,
                                             resilient_legion_context_t ctx,
                                             resilient_legion_index_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_index_space()
   */
  void
  resilient_legion_index_space_destroy(resilient_legion_runtime_t runtime,
                             resilient_legion_context_t ctx,
                             resilient_legion_index_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_index_space()
   */
  void
  resilient_legion_index_space_destroy_unordered(resilient_legion_runtime_t runtime,
                                       resilient_legion_context_t ctx,
                                       resilient_legion_index_space_t handle,
                                       bool unordered);

  /**
   * @see Legion::Runtime::attach_semantic_information()
   */
  void
  resilient_legion_index_space_attach_semantic_information(resilient_legion_runtime_t runtime,
                                                 resilient_legion_index_space_t handle,
                                                 resilient_legion_semantic_tag_t tag,
                                                 const void *buffer,
                                                 size_t size,
                                                 bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_semantic_information()
   */
  bool
  resilient_legion_index_space_retrieve_semantic_information(
                                           resilient_legion_runtime_t runtime,
                                           resilient_legion_index_space_t handle,
                                           resilient_legion_semantic_tag_t tag,
                                           const void **result,
                                           size_t *size,
                                           bool can_fail /* = false */,
                                           bool wait_until_ready /* = false */);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  resilient_legion_index_space_attach_name(resilient_legion_runtime_t runtime,
                                 resilient_legion_index_space_t handle,
                                 const char *name,
                                 bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  resilient_legion_index_space_retrieve_name(resilient_legion_runtime_t runtime,
                                   resilient_legion_index_space_t handle,
                                   const char **result);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::IndexSpace::get_dim()
   */
  int
  resilient_legion_index_space_get_dim(resilient_legion_index_space_t handle);

  // -----------------------------------------------------------------------
  // Index Partition Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition(
   *        Context, IndexSpace, Coloring, bool, int)
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_coloring(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t parent,
    resilient_legion_coloring_t coloring,
    bool disjoint,
    resilient_legion_color_t part_color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition(
   *        Context, IndexSpace, Domain, DomainColoring, bool, int)
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_domain_coloring(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t parent,
    resilient_legion_domain_t color_space,
    resilient_legion_domain_coloring_t coloring,
    bool disjoint,
    resilient_legion_color_t part_color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition(
   *        Context, IndexSpace, Domain, PointColoring, PartitionKind, int)
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_point_coloring(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t parent,
    resilient_legion_domain_t color_space,
    resilient_legion_point_coloring_t coloring,
    resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    resilient_legion_color_t color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition(
   *        Context, IndexSpace, Domain, DomainPointColoring, PartitionKind, int)
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_domain_point_coloring(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t parent,
    resilient_legion_domain_t color_space,
    resilient_legion_domain_point_coloring_t coloring,
    resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    resilient_legion_color_t color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition(
   *        Context, IndexSpace, Domain, MultiDomainPointColoring, PartitionKind, int)
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_multi_domain_point_coloring(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t parent,
    resilient_legion_domain_t color_space,
    resilient_legion_multi_domain_point_coloring_t coloring,
    resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    resilient_legion_color_t color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition<T>(
   *        Context, IndexSpace, const T&, int)
   */
#define CREATE_BLOCKIFY(DIM) \
  resilient_legion_index_partition_t \
  resilient_legion_index_partition_create_blockify_##DIM##d( \
    resilient_legion_runtime_t runtime, \
    resilient_legion_context_t ctx, \
    resilient_legion_index_space_t parent, \
    resilient_legion_blockify_##DIM##d_t blockify, \
    resilient_legion_color_t part_color /* = AUTO_GENERATE_ID */);
  LEGION_FOREACH_N(CREATE_BLOCKIFY)
#undef CREATE_BLOCKIFY

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_equal_partition()
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_equal(resilient_legion_runtime_t runtime,
                                      resilient_legion_context_t ctx,
                                      resilient_legion_index_space_t parent,
                                      resilient_legion_index_space_t color_space,
                                      size_t granularity,
                                      resilient_legion_color_t color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::create_partition_by_weights
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_weights(
      resilient_legion_runtime_t runtime,
      resilient_legion_context_t ctx,
      resilient_legion_index_space_t parent,
      resilient_legion_domain_point_t *colors,
      int *weights,
      size_t num_colors,
      resilient_legion_index_space_t color_space,
      size_t granularity /* = 1 */,
      resilient_legion_color_t color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::create_partition_by_weights
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_weights_future_map(
      resilient_legion_runtime_t runtime,
      resilient_legion_context_t ctx,
      resilient_legion_index_space_t parent,
      resilient_legion_future_map_t future_map,
      resilient_legion_index_space_t color_space,
      size_t granularity /* = 1 */,
      resilient_legion_color_t color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_union()
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_union(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t parent,
    resilient_legion_index_partition_t handle1,
    resilient_legion_index_partition_t handle2,
    resilient_legion_index_space_t color_space,
    resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    resilient_legion_color_t color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_intersection()
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_intersection(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t parent,
    resilient_legion_index_partition_t handle1,
    resilient_legion_index_partition_t handle2,
    resilient_legion_index_space_t color_space,
    resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    resilient_legion_color_t color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_intersection()
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_intersection_mirror(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t parent,
    resilient_legion_index_partition_t handle,
    resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    resilient_legion_color_t color /* = AUTO_GENERATE_ID */,
    bool dominates /* = false */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_difference()
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_difference(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t parent,
    resilient_legion_index_partition_t handle1,
    resilient_legion_index_partition_t handle2,
    resilient_legion_index_space_t color_space,
    resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    resilient_legion_color_t color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::create_partition_by_domain
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_domain(
      resilient_legion_runtime_t runtime,
      resilient_legion_context_t ctx,
      resilient_legion_index_space_t parent,
      resilient_legion_domain_point_t *colors,
      resilient_legion_domain_t *domains,
      size_t num_color_domains,
      resilient_legion_index_space_t color_space,
      bool perform_intersections /* = true */,
      resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
      resilient_legion_color_t color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::create_partition_by_domain
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_domain_future_map(
      resilient_legion_runtime_t runtime,
      resilient_legion_context_t ctx,
      resilient_legion_index_space_t parent,
      resilient_legion_future_map_t future_map,
      resilient_legion_index_space_t color_space,
      bool perform_intersections /* = true */,
      resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
      resilient_legion_color_t color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_field()
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_field(resilient_legion_runtime_t runtime,
                                         resilient_legion_context_t ctx,
                                         resilient_legion_logical_region_t handle,
                                         resilient_legion_logical_region_t parent,
                                         resilient_legion_field_id_t fid,
                                         resilient_legion_index_space_t color_space,
                                         resilient_legion_color_t color /* = AUTO_GENERATE_ID */,
                                         resilient_legion_mapper_id_t id /* = 0 */,
                                         resilient_legion_mapping_tag_id_t tag /* = 0 */,
                                         resilient_legion_partition_kind_t part_kind /* = DISJOINT_KIND */,
    resilient_legion_untyped_buffer_t map_arg);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_image()
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_image(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t handle,
    resilient_legion_logical_partition_t projection,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    resilient_legion_index_space_t color_space,
    resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    resilient_legion_color_t color /* = AUTO_GENERATE_ID */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    resilient_legion_untyped_buffer_t map_arg);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_preimage()
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_preimage(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_partition_t projection,
    resilient_legion_logical_region_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    resilient_legion_index_space_t color_space,
    resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    resilient_legion_color_t color /* = AUTO_GENERATE_ID */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    resilient_legion_untyped_buffer_t map_arg);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_image_range()
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_image_range(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t handle,
    resilient_legion_logical_partition_t projection,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    resilient_legion_index_space_t color_space,
    resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    resilient_legion_color_t color /* = AUTO_GENERATE_ID */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    resilient_legion_untyped_buffer_t map_arg);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_preimage()
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_preimage_range(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_partition_t projection,
    resilient_legion_logical_region_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    resilient_legion_index_space_t color_space,
    resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    resilient_legion_color_t color /* = AUTO_GENERATE_ID */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    resilient_legion_untyped_buffer_t map_arg);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_restriction()
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_by_restriction(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t parent,
    resilient_legion_index_space_t color_space,
    resilient_legion_domain_transform_t transform,
    resilient_legion_domain_t extent,
    resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    resilient_legion_color_t color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_pending_partition()
   */
  resilient_legion_index_partition_t
  resilient_legion_index_partition_create_pending_partition(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t parent,
    resilient_legion_index_space_t color_space,
    resilient_legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    resilient_legion_color_t color /* = AUTO_GENERATE_ID */);

  /**
   * @see Legion::Runtime::create_index_space_union()
   */
  resilient_legion_index_space_t
  resilient_legion_index_partition_create_index_space_union_spaces(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_partition_t parent,
    resilient_legion_domain_point_t color,
    const resilient_legion_index_space_t *spaces,
    size_t num_spaces);

  /**
   * @see Legion::Runtime::create_index_space_union()
   */
  resilient_legion_index_space_t
  resilient_legion_index_partition_create_index_space_union_partition(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_partition_t parent,
    resilient_legion_domain_point_t color,
    resilient_legion_index_partition_t handle);

  /**
   * @see Legion::Runtime::create_index_space_intersection()
   */
  resilient_legion_index_space_t
  resilient_legion_index_partition_create_index_space_intersection_spaces(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_partition_t parent,
    resilient_legion_domain_point_t color,
    const resilient_legion_index_space_t *spaces,
    size_t num_spaces);

  /**
   * @see Legion::Runtime::create_index_space_intersection()
   */
  resilient_legion_index_space_t
  resilient_legion_index_partition_create_index_space_intersection_partition(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_partition_t parent,
    resilient_legion_domain_point_t color,
    resilient_legion_index_partition_t handle);

  /**
   * @see Legion::Runtime::create_index_space_difference()
   */
  resilient_legion_index_space_t
  resilient_legion_index_partition_create_index_space_difference(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_partition_t parent,
    resilient_legion_domain_point_t color,
    resilient_legion_index_space_t initial,
    const resilient_legion_index_space_t *spaces,
    size_t num_spaces);

  /**
   * @see Legion::Runtime::is_index_partition_disjoint()
   */
  bool
  resilient_legion_index_partition_is_disjoint(resilient_legion_runtime_t runtime,
                                     resilient_legion_index_partition_t handle);

  /**
   * @see Legion::Runtime::is_index_partition_complete()
   */
  bool
  resilient_legion_index_partition_is_complete(resilient_legion_runtime_t runtime,
                                     resilient_legion_index_partition_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_index_subspace()
   */
  resilient_legion_index_space_t
  resilient_legion_index_partition_get_index_subspace(resilient_legion_runtime_t runtime,
                                            resilient_legion_index_partition_t handle,
                                            resilient_legion_color_t color);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_index_subspace()
   */
  resilient_legion_index_space_t
  resilient_legion_index_partition_get_index_subspace_domain_point(
    resilient_legion_runtime_t runtime,
    resilient_legion_index_partition_t handle,
    resilient_legion_domain_point_t color);

  /**
   * @see Legion::Runtime::has_index_subspace()
   */
  bool
  resilient_legion_index_partition_has_index_subspace_domain_point(
    resilient_legion_runtime_t runtime,
    resilient_legion_index_partition_t handle,
    resilient_legion_domain_point_t color);

  /**
   * @see Legion::Runtime::get_index_partition_color_space_name()
   */
  resilient_legion_index_space_t
  resilient_legion_index_partition_get_color_space(resilient_legion_runtime_t runtime,
                                         resilient_legion_index_partition_t handle);

  /**
   * @see Legion::Runtime::get_index_partition_color()
   */
  resilient_legion_color_t
  resilient_legion_index_partition_get_color(resilient_legion_runtime_t runtime,
                                   resilient_legion_index_partition_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_parent_index_space()
   */
  resilient_legion_index_space_t
  resilient_legion_index_partition_get_parent_index_space(resilient_legion_runtime_t runtime,
                                                resilient_legion_index_partition_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::create_shared_ownership
   */
  void
  resilient_legion_index_partition_create_shared_ownership(resilient_legion_runtime_t runtime,
                                                 resilient_legion_context_t ctx,
                                                 resilient_legion_index_partition_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_index_space()
   */
  void
  resilient_legion_index_partition_destroy(resilient_legion_runtime_t runtime,
                                 resilient_legion_context_t ctx,
                                 resilient_legion_index_partition_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_index_space()
   */
  void
  resilient_legion_index_partition_destroy_unordered(resilient_legion_runtime_t runtime,
                                           resilient_legion_context_t ctx,
                                           resilient_legion_index_partition_t handle,
                                           bool unordered /* = false */,
                                           bool recurse /* = true */);

  /**
   * @see Legion::Runtime::attach_semantic_information()
   */
  void
  resilient_legion_index_partition_attach_semantic_information(
                                                resilient_legion_runtime_t runtime,
                                                resilient_legion_index_partition_t handle,
                                                resilient_legion_semantic_tag_t tag,
                                                const void *buffer,
                                                size_t size,
                                                bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_semantic_information()
   */
  bool
  resilient_legion_index_partition_retrieve_semantic_information(
                                           resilient_legion_runtime_t runtime,
                                           resilient_legion_index_partition_t handle,
                                           resilient_legion_semantic_tag_t tag,
                                           const void **result,
                                           size_t *size,
                                           bool can_fail /* = false */,
                                           bool wait_until_ready /* = false */);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  resilient_legion_index_partition_attach_name(resilient_legion_runtime_t runtime,
                                     resilient_legion_index_partition_t handle,
                                     const char *name,
                                     bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  resilient_legion_index_partition_retrieve_name(resilient_legion_runtime_t runtime,
                                       resilient_legion_index_partition_t handle,
                                       const char **result);

  // -----------------------------------------------------------------------
  // Field Space Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_field_space()
   */
  resilient_legion_field_space_t
  resilient_legion_field_space_create(resilient_legion_runtime_t runtime,
                            resilient_legion_context_t ctx);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::create_field_space()
   */
  resilient_legion_field_space_t
  resilient_legion_field_space_create_with_fields(resilient_legion_runtime_t runtime,
                                        resilient_legion_context_t ctx,
                                        size_t *field_sizes,
                                        resilient_legion_field_id_t *field_ids,
                                        size_t num_fields, 
                                        resilient_legion_custom_serdez_id_t serdez);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::create_field_space()
   */
  resilient_legion_field_space_t
  resilient_legion_field_space_create_with_futures(resilient_legion_runtime_t runtime,
                                         resilient_legion_context_t ctx,
                                         resilient_legion_future_t *field_sizes,
                                         resilient_legion_field_id_t *field_ids,
                                         size_t num_fields, 
                                         resilient_legion_custom_serdez_id_t serdez);

  /**
   * @see Legion::FieldSpace::NO_SPACE
   */
  resilient_legion_field_space_t
  resilient_legion_field_space_no_space();

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::create_shared_ownership
   */
  void
  resilient_legion_field_space_create_shared_ownership(resilient_legion_runtime_t runtime,
                                             resilient_legion_context_t ctx,
                                             resilient_legion_field_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_field_space()
   */
  void
  resilient_legion_field_space_destroy(resilient_legion_runtime_t runtime,
                             resilient_legion_context_t ctx,
                             resilient_legion_field_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_field_space()
   */
  void
  resilient_legion_field_space_destroy_unordered(resilient_legion_runtime_t runtime,
                                       resilient_legion_context_t ctx,
                                       resilient_legion_field_space_t handle,
                                       bool unordered);

  /**
   * @see Legion::Runtime::attach_semantic_information()
   */
  void
  resilient_legion_field_space_attach_semantic_information(
                                                resilient_legion_runtime_t runtime,
                                                resilient_legion_field_space_t handle,
                                                resilient_legion_semantic_tag_t tag,
                                                const void *buffer,
                                                size_t size,
                                                bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_semantic_information()
   */
  bool
  resilient_legion_field_space_retrieve_semantic_information(
                                           resilient_legion_runtime_t runtime,
                                           resilient_legion_field_space_t handle,
                                           resilient_legion_semantic_tag_t tag,
                                           const void **result,
                                           size_t *size,
                                           bool can_fail /* = false */,
                                           bool wait_until_ready /* = false */);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::get_field_space_fields()
   */
  resilient_legion_field_id_t *
  resilient_legion_field_space_get_fields(resilient_legion_runtime_t runtime,
                                resilient_legion_context_t ctx,
                                resilient_legion_field_space_t handle,
                                size_t *size);

  /**
   * @param handle Caller must have ownership of parameter `fields`.
   *
   * @see Legion::Runtime::get_field_space_fields()
   */
  bool
  resilient_legion_field_space_has_fields(resilient_legion_runtime_t runtime,
                                resilient_legion_context_t ctx,
                                resilient_legion_field_space_t handle,
                                const resilient_legion_field_id_t *fields,
                                size_t fields_size);

  /**
   * @see Legion::Runtime::attach_semantic_information()
   */
  void
  resilient_legion_field_id_attach_semantic_information(resilient_legion_runtime_t runtime,
                                              resilient_legion_field_space_t handle,
                                              resilient_legion_field_id_t id,
                                              resilient_legion_semantic_tag_t tag,
                                              const void *buffer,
                                              size_t size,
                                              bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_semantic_information()
   */
  bool
  resilient_legion_field_id_retrieve_semantic_information(
                                           resilient_legion_runtime_t runtime,
                                           resilient_legion_field_space_t handle,
                                           resilient_legion_field_id_t id,
                                           resilient_legion_semantic_tag_t tag,
                                           const void **result,
                                           size_t *size,
                                           bool can_fail /* = false */,
                                           bool wait_until_ready /* = false */);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  resilient_legion_field_space_attach_name(resilient_legion_runtime_t runtime,
                                 resilient_legion_field_space_t handle,
                                 const char *name,
                                 bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  resilient_legion_field_space_retrieve_name(resilient_legion_runtime_t runtime,
                                   resilient_legion_field_space_t handle,
                                   const char **result);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  resilient_legion_field_id_attach_name(resilient_legion_runtime_t runtime,
                              resilient_legion_field_space_t handle,
                              resilient_legion_field_id_t id,
                              const char *name,
                              bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  resilient_legion_field_id_retrieve_name(resilient_legion_runtime_t runtime,
                                resilient_legion_field_space_t handle,
                                resilient_legion_field_id_t id,
                                const char **result);

  /**
   * @see Legion::Runtime::get_field_size()
   */
  size_t
  resilient_legion_field_id_get_size(resilient_legion_runtime_t runtime,
                           resilient_legion_context_t ctx,
                           resilient_legion_field_space_t handle,
                           resilient_legion_field_id_t id);

  // -----------------------------------------------------------------------
  // Logical Region Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_logical_region()
   */
  resilient_legion_logical_region_t
  resilient_legion_logical_region_create(resilient_legion_runtime_t runtime,
                               resilient_legion_context_t ctx,
                               resilient_legion_index_space_t index,
                               resilient_legion_field_space_t fields,
                               bool task_local);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::create_shared_ownership
   */
  void
  resilient_legion_logical_region_create_shared_ownership(resilient_legion_runtime_t runtime,
                                                resilient_legion_context_t ctx,
                                                resilient_legion_logical_region_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_logical_region()
   */
  void
  resilient_legion_logical_region_destroy(resilient_legion_runtime_t runtime,
                                resilient_legion_context_t ctx,
                                resilient_legion_logical_region_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_logical_region()
   */
  void
  resilient_legion_logical_region_destroy_unordered(resilient_legion_runtime_t runtime,
                                          resilient_legion_context_t ctx,
                                          resilient_legion_logical_region_t handle,
                                          bool unordered);

  /**
   * @see Legion::Runtime::get_logical_region_color()
   */
  resilient_legion_color_t
  resilient_legion_logical_region_get_color(resilient_legion_runtime_t runtime,
                                  resilient_legion_logical_region_t handle);

  /**
   * @see Legion::Runtime::get_logical_region_color_point()
   */
  resilient_legion_domain_point_t
  resilient_legion_logical_region_get_color_domain_point(resilient_legion_runtime_t runtime_,
                                               resilient_legion_logical_region_t handle_);

  /**
   * @see Legion::Runtime::has_parent_logical_partition()
   */
  bool
  resilient_legion_logical_region_has_parent_logical_partition(
    resilient_legion_runtime_t runtime,
    resilient_legion_logical_region_t handle);

  /**
   * @see Legion::Runtime::get_parent_logical_partition()
   */
  resilient_legion_logical_partition_t
  resilient_legion_logical_region_get_parent_logical_partition(
    resilient_legion_runtime_t runtime,
    resilient_legion_logical_region_t handle);

  /**
   * @see Legion::Runtime::attach_semantic_information()
   */
  void
  resilient_legion_logical_region_attach_semantic_information(
                                                resilient_legion_runtime_t runtime,
                                                resilient_legion_logical_region_t handle,
                                                resilient_legion_semantic_tag_t tag,
                                                const void *buffer,
                                                size_t size,
                                                bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_semantic_information()
   */
  bool
  resilient_legion_logical_region_retrieve_semantic_information(
                                           resilient_legion_runtime_t runtime,
                                           resilient_legion_logical_region_t handle,
                                           resilient_legion_semantic_tag_t tag,
                                           const void **result,
                                           size_t *size,
                                           bool can_fail /* = false */,
                                           bool wait_until_ready /* = false */);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  resilient_legion_logical_region_attach_name(resilient_legion_runtime_t runtime,
                                    resilient_legion_logical_region_t handle,
                                    const char *name,
                                    bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  resilient_legion_logical_region_retrieve_name(resilient_legion_runtime_t runtime,
                                      resilient_legion_logical_region_t handle,
                                      const char **result);

  /**
   * @see Legion::LogicalRegion::get_index_space
   */
  resilient_legion_index_space_t
  resilient_legion_logical_region_get_index_space(resilient_legion_logical_region_t handle); 
  
  // -----------------------------------------------------------------------
  // Logical Region Tree Traversal Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::get_logical_partition()
   */
  resilient_legion_logical_partition_t
  resilient_legion_logical_partition_create(resilient_legion_runtime_t runtime,
                                  resilient_legion_logical_region_t parent,
                                  resilient_legion_index_partition_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::get_logical_partition_by_tree()
   */
  resilient_legion_logical_partition_t
  resilient_legion_logical_partition_create_by_tree(resilient_legion_runtime_t runtime,
                                          resilient_legion_context_t ctx,
                                          resilient_legion_index_partition_t handle,
                                          resilient_legion_field_space_t fspace,
                                          resilient_legion_region_tree_id_t tid);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_logical_partition()
   */
  void
  resilient_legion_logical_partition_destroy(resilient_legion_runtime_t runtime,
                                   resilient_legion_context_t ctx,
                                   resilient_legion_logical_partition_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_logical_partition()
   */
  void
  resilient_legion_logical_partition_destroy_unordered(resilient_legion_runtime_t runtime,
                                             resilient_legion_context_t ctx,
                                             resilient_legion_logical_partition_t handle,
                                             bool unordered);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_logical_subregion()
   */
  resilient_legion_logical_region_t
  resilient_legion_logical_partition_get_logical_subregion(
    resilient_legion_runtime_t runtime,
    resilient_legion_logical_partition_t parent,
    resilient_legion_index_space_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_logical_subregion_by_color()
   */
  resilient_legion_logical_region_t
  resilient_legion_logical_partition_get_logical_subregion_by_color(
    resilient_legion_runtime_t runtime,
    resilient_legion_logical_partition_t parent,
    resilient_legion_color_t c);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_logical_subregion_by_color()
   */
  resilient_legion_logical_region_t
  resilient_legion_logical_partition_get_logical_subregion_by_color_domain_point(
    resilient_legion_runtime_t runtime,
    resilient_legion_logical_partition_t parent,
    resilient_legion_domain_point_t c);

  /**
   * @see Legion::Runtime::has_logical_subregion_by_color()
   */
  bool
  resilient_legion_logical_partition_has_logical_subregion_by_color_domain_point(
    resilient_legion_runtime_t runtime,
    resilient_legion_logical_partition_t parent,
    resilient_legion_domain_point_t c);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_logical_subregion_by_tree()
   */
  resilient_legion_logical_region_t
  resilient_legion_logical_partition_get_logical_subregion_by_tree(
    resilient_legion_runtime_t runtime,
    resilient_legion_index_space_t handle,
    resilient_legion_field_space_t fspace,
    resilient_legion_region_tree_id_t tid);

  /**
   * @see Legion::Runtime::get_parent_logical_region()
   */
  resilient_legion_logical_region_t
  resilient_legion_logical_partition_get_parent_logical_region(
    resilient_legion_runtime_t runtime,
    resilient_legion_logical_partition_t handle);

  /**
   * @see Legion::Runtime::attach_semantic_information()
   */
  void
  resilient_legion_logical_partition_attach_semantic_information(
                                                resilient_legion_runtime_t runtime,
                                                resilient_legion_logical_partition_t handle,
                                                resilient_legion_semantic_tag_t tag,
                                                const void *buffer,
                                                size_t size,
                                                bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_semantic_information()
   */
  bool
  resilient_legion_logical_partition_retrieve_semantic_information(
                                           resilient_legion_runtime_t runtime,
                                           resilient_legion_logical_partition_t handle,
                                           resilient_legion_semantic_tag_t tag,
                                           const void **result,
                                           size_t *size,
                                           bool can_fail /* = false */,
                                           bool wait_until_ready /* = false */);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  resilient_legion_logical_partition_attach_name(resilient_legion_runtime_t runtime,
                                       resilient_legion_logical_partition_t handle,
                                       const char *name,
                                       bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  resilient_legion_logical_partition_retrieve_name(resilient_legion_runtime_t runtime,
                                         resilient_legion_logical_partition_t handle,
                                         const char **result);

  /**
   * The caller must have ownership of all regions, partitions and fields
   * passed into this function.
   *
   * @see Legion::Runtime::advise_analysis_subtree()
   */
  void resilient_legion_advise_analysis_subtree(resilient_legion_runtime_t runtime,
                                      resilient_legion_context_t ctx,
                                      resilient_legion_logical_region_t parent,
                                      int num_regions,
                                      resilient_legion_logical_region_t* regions,
                                      int num_parts,
                                      resilient_legion_logical_partition_t* partitions,
                                      int num_fields,
                                      resilient_legion_field_id_t* fields);

  // -----------------------------------------------------------------------
  // Region Requirement Operations
  // -----------------------------------------------------------------------
  
  /**
   * @see Legion::RegionRequirement::RegionRequirement()
   */
  resilient_legion_region_requirement_t
  resilient_legion_region_requirement_create_logical_region(
    resilient_legion_logical_region_t handle,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::RegionRequirement::RegionRequirement()
   */
  resilient_legion_region_requirement_t
  resilient_legion_region_requirement_create_logical_region_projection(
    resilient_legion_logical_region_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::RegionRequirement::RegionRequirement()
   */
  resilient_legion_region_requirement_t
  resilient_legion_region_requirement_create_logical_partition(
    resilient_legion_logical_partition_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::Requirement::~Requirement()
   */
  void
  resilient_legion_region_requirement_destroy(resilient_legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::add_field()
   */
  void
  resilient_legion_region_requirement_add_field(resilient_legion_region_requirement_t handle,
                                      resilient_legion_field_id_t field,
                                      bool instance_field);

  /**
   * @see Legion::RegionRequirement::add_flags
   */
  void
  resilient_legion_region_requirement_add_flags(resilient_legion_region_requirement_t handle,
                                      resilient_legion_region_flags_t flags);

  /**
   * @see Legion::RegionRequirement::region
   */
  resilient_legion_logical_region_t
  resilient_legion_region_requirement_get_region(resilient_legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::parent
   */
  resilient_legion_logical_region_t
  resilient_legion_region_requirement_get_parent(resilient_legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::partition
   */
  resilient_legion_logical_partition_t
  resilient_legion_region_requirement_get_partition(resilient_legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::privilege_fields
   */
  unsigned
  resilient_legion_region_requirement_get_privilege_fields_size(
      resilient_legion_region_requirement_t handle);

  /**
   * @param fields Caller should give a buffer of the size fields_size
   *
   * @param fields_size the size of the buffer fields
   *
   * @return returns privilege fields in the region requirement.
   *         The return might be truncated if the buffer size is
   *         smaller than the number of privilege fields.
   *
   * @see Legion::RegionRequirement::privilege_fields
   */
  void
  resilient_legion_region_requirement_get_privilege_fields(
      resilient_legion_region_requirement_t handle,
      resilient_legion_field_id_t* fields,
      unsigned fields_size);

  /**
   * @return returns the i-th privilege field in the region requirement.
   *         note that this function takes O(n) time due to the underlying
   *         data structure does not provide an indexing operation.
   *
   * @see Legion::RegionRequirement::privilege_fields
   */
  resilient_legion_field_id_t
  resilient_legion_region_requirement_get_privilege_field(
      resilient_legion_region_requirement_t handle,
      unsigned idx);

  /**
   * @see Legion::RegionRequirement::instance_fields
   */
  unsigned
  resilient_legion_region_requirement_get_instance_fields_size(
      resilient_legion_region_requirement_t handle);

  /**
   * @param fields Caller should give a buffer of the size fields_size
   *
   * @param fields_size the size of the buffer fields
   *
   * @return returns instance fields in the region requirement.
   *         The return might be truncated if the buffer size is
   *         smaller than the number of instance fields.
   *
   * @see Legion::RegionRequirement::instance_fields
   */
  void
  resilient_legion_region_requirement_get_instance_fields(
      resilient_legion_region_requirement_t handle,
      resilient_legion_field_id_t* fields,
      unsigned fields_size);

  /**
   * @return returns the i-th instance field in the region requirement.
   *
   * @see Legion::RegionRequirement::instance_fields
   */
  resilient_legion_field_id_t
  resilient_legion_region_requirement_get_instance_field(
      resilient_legion_region_requirement_t handle,
      unsigned idx);

  /**
   * @see Legion::RegionRequirement::privilege
   */
  resilient_legion_privilege_mode_t
  resilient_legion_region_requirement_get_privilege(resilient_legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::prop
   */
  resilient_legion_coherence_property_t
  resilient_legion_region_requirement_get_prop(resilient_legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::redop
   */
  resilient_legion_reduction_op_id_t
  resilient_legion_region_requirement_get_redop(resilient_legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::tag
   */
  resilient_legion_mapping_tag_id_t
  resilient_legion_region_requirement_get_tag(resilient_legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::handle_type
   */
  resilient_legion_handle_type_t
  resilient_legion_region_requirement_get_handle_type(resilient_legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::projection
   */
  resilient_legion_projection_id_t
  resilient_legion_region_requirement_get_projection(resilient_legion_region_requirement_t handle);

  // -----------------------------------------------------------------------
  // Output Requirement Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::OutputRequirement::OutputRequirement()
   */
  resilient_legion_output_requirement_t
  resilient_legion_output_requirement_create(resilient_legion_field_space_t field_space,
                                   resilient_legion_field_id_t *fields,
                                   size_t fields_size,
                                   int dim,
                                   bool global_indexing);

  /**
   * @see Legion::OutputRequirement::OutputRequirement()
   */
  resilient_legion_output_requirement_t
  resilient_legion_output_requirement_create_region_requirement(
      resilient_legion_region_requirement_t handle);

  /**
   * @see Legion::OutputRequirement::~OutputRequirement()
   */
  void
  resilient_legion_output_requirement_destroy(resilient_legion_output_requirement_t handle);

  /**
   * @see Legion::OutputRequirement::add_field()
   */
  void
  resilient_legion_output_requirement_add_field(resilient_legion_output_requirement_t handle,
                                      resilient_legion_field_id_t field,
                                      bool instance);

  /**
   * @see Legion::OutputRequirement::region
   */
  resilient_legion_logical_region_t
  resilient_legion_output_requirement_get_region(resilient_legion_output_requirement_t handle);

  /**
   * @see Legion::OutputRequirement::parent
   */
  resilient_legion_logical_region_t
  resilient_legion_output_requirement_get_parent(resilient_legion_output_requirement_t handle);

  /**
   * @see Legion::OutputRequirement::partition
   */
  resilient_legion_logical_partition_t
  resilient_legion_output_requirement_get_partition(resilient_legion_output_requirement_t handle);

  // -----------------------------------------------------------------------
  // Allocator and Argument Map Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_field_allocator()
   */
  resilient_legion_field_allocator_t
  resilient_legion_field_allocator_create(resilient_legion_runtime_t runtime,
                                resilient_legion_context_t ctx,
                                resilient_legion_field_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::FieldAllocator::~FieldAllocator()
   */
  void
  resilient_legion_field_allocator_destroy(resilient_legion_field_allocator_t handle);

  /**
   * This will give the value of the macro AUTO_GENERATE_ID
   */
  resilient_legion_field_id_t
  resilient_legion_auto_generate_id(void);

  /**
   * @see Legion::FieldAllocator::allocate_field()
   */
  resilient_legion_field_id_t
  resilient_legion_field_allocator_allocate_field(
    resilient_legion_field_allocator_t allocator,
    size_t field_size,
    resilient_legion_field_id_t desired_fieldid /* = AUTO_GENERATE_ID */);

  /**
   * @see Legion::FieldAllocator::allocate_field()
   */
  resilient_legion_field_id_t
  resilient_legion_field_allocator_allocate_field_future(
    resilient_legion_field_allocator_t allocator,
    resilient_legion_future_t field_size,
    resilient_legion_field_id_t desired_fieldid /* = AUTO_GENERATE_ID */);

  /**
   * @see Legion::FieldAllocator::free_field()
   */
  void
  resilient_legion_field_allocator_free_field(resilient_legion_field_allocator_t allocator,
                                    resilient_legion_field_id_t fid);

  /**
   * @see Legion::FieldAllocator::free_field()
   */
  void
  resilient_legion_field_allocator_free_field_unordered(resilient_legion_field_allocator_t allocator,
                                              resilient_legion_field_id_t fid,
                                              bool unordered);

  /**
   * @see Legion::FieldAllocator::allocate_local_field()
   */
  resilient_legion_field_id_t
  resilient_legion_field_allocator_allocate_local_field(
    resilient_legion_field_allocator_t allocator,
    size_t field_size,
    resilient_legion_field_id_t desired_fieldid /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::ArgumentMap::ArgumentMap()
   */
  resilient_legion_argument_map_t
  resilient_legion_argument_map_create(void);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::ArgumentMap::ArgumentMap()
   */
  resilient_legion_argument_map_t
  resilient_legion_argument_map_from_future_map(resilient_legion_future_map_t map);

  /**
   * @see Legion::ArgumentMap::set_point()
   */
  void
  resilient_legion_argument_map_set_point(resilient_legion_argument_map_t map,
                                resilient_legion_domain_point_t dp,
                                resilient_legion_untyped_buffer_t arg,
                                bool replace /* = true */);

  /**
   * @see Legion::ArgumentMap::set_point()
   */
  void
  resilient_legion_argument_map_set_future(resilient_legion_argument_map_t map,
                                 resilient_legion_domain_point_t dp,
                                 resilient_legion_future_t future,
                                 bool replace /* = true */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::ArgumentMap::~ArgumentMap()
   */
  void
  resilient_legion_argument_map_destroy(resilient_legion_argument_map_t handle);

  // -----------------------------------------------------------------------
  // Predicate Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_predicate()
   */
  resilient_legion_predicate_t
  resilient_legion_predicate_create(resilient_legion_runtime_t runtime,
                          resilient_legion_context_t ctx,
                          resilient_legion_future_t f);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Predicate::~Predicate()
   */
  void
  resilient_legion_predicate_destroy(resilient_legion_predicate_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Predicate::TRUE_PRED
   */
  const resilient_legion_predicate_t
  resilient_legion_predicate_true(void);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Predicate::FALSE_PRED
   */
  const resilient_legion_predicate_t
  resilient_legion_predicate_false(void);

  // -----------------------------------------------------------------------
  // Phase Barrier Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_phase_barrier()
   */
  resilient_legion_phase_barrier_t
  resilient_legion_phase_barrier_create(resilient_legion_runtime_t runtime,
                              resilient_legion_context_t ctx,
                              unsigned arrivals);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_phase_barrier()
   */
  void
  resilient_legion_phase_barrier_destroy(resilient_legion_runtime_t runtime,
                               resilient_legion_context_t ctx,
                               resilient_legion_phase_barrier_t handle);

  /**
   * @see Legion::PhaseBarrier::alter_arrival_count()
   */
  resilient_legion_phase_barrier_t
  resilient_legion_phase_barrier_alter_arrival_count(resilient_legion_runtime_t runtime,
                                           resilient_legion_context_t ctx,
                                           resilient_legion_phase_barrier_t handle,
                                           int delta);

  /**
   * @see Legion::PhaseBarrier::arrive()
   */
  void
  resilient_legion_phase_barrier_arrive(resilient_legion_runtime_t runtime,
                              resilient_legion_context_t ctx,
                              resilient_legion_phase_barrier_t handle,
                              unsigned count /* = 1 */);

  /**
   * @see Legion::PhaseBarrier::wait()
   */
  void
  resilient_legion_phase_barrier_wait(resilient_legion_runtime_t runtime,
                            resilient_legion_context_t ctx,
                            resilient_legion_phase_barrier_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::advance_phase_barrier()
   */
  resilient_legion_phase_barrier_t
  resilient_legion_phase_barrier_advance(resilient_legion_runtime_t runtime,
                               resilient_legion_context_t ctx,
                               resilient_legion_phase_barrier_t handle);

  // -----------------------------------------------------------------------
  // Dynamic Collective Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_dynamic_collective()
   */
  resilient_legion_dynamic_collective_t
  resilient_legion_dynamic_collective_create(resilient_legion_runtime_t runtime,
                                   resilient_legion_context_t ctx,
                                   unsigned arrivals,
                                   resilient_legion_reduction_op_id_t redop,
                                   const void *init_value,
                                   size_t init_size);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_dynamic_collective()
   */
  void
  resilient_legion_dynamic_collective_destroy(resilient_legion_runtime_t runtime,
                                    resilient_legion_context_t ctx,
                                    resilient_legion_dynamic_collective_t handle);

  /**
   * @see Legion::DynamicCollective::alter_arrival_count()
   */
  resilient_legion_dynamic_collective_t
  resilient_legion_dynamic_collective_alter_arrival_count(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_dynamic_collective_t handle,
    int delta);

  /**
   * @see Legion::Runtime::arrive_dynamic_collective()
   */
  void
  resilient_legion_dynamic_collective_arrive(resilient_legion_runtime_t runtime,
                                   resilient_legion_context_t ctx,
                                   resilient_legion_dynamic_collective_t handle,
                                   const void *buffer,
                                   size_t size,
                                   unsigned count /* = 1 */);

  /**
   * @see Legion::Runtime::defer_dynamic_collective_arrival()
   */
  void
  resilient_legion_dynamic_collective_defer_arrival(resilient_legion_runtime_t runtime,
                                          resilient_legion_context_t ctx,
                                          resilient_legion_dynamic_collective_t handle,
                                          resilient_legion_future_t f,
                                          unsigned count /* = 1 */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::get_dynamic_collective_result()
   */
  resilient_legion_future_t
  resilient_legion_dynamic_collective_get_result(resilient_legion_runtime_t runtime,
                                       resilient_legion_context_t ctx,
                                       resilient_legion_dynamic_collective_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::advance_dynamic_collective()
   */
  resilient_legion_dynamic_collective_t
  resilient_legion_dynamic_collective_advance(resilient_legion_runtime_t runtime,
                                    resilient_legion_context_t ctx,
                                    resilient_legion_dynamic_collective_t handle);

  // -----------------------------------------------------------------------
  // Future Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Future::from_untyped_pointer()
   */
  resilient_legion_future_t
  resilient_legion_future_from_untyped_pointer(resilient_legion_runtime_t runtime,
                                     const void *buffer,
                                     size_t size);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Future::Future()
   */
  resilient_legion_future_t
  resilient_legion_future_copy(resilient_legion_future_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Future::~Future()
   */
  void
  resilient_legion_future_destroy(resilient_legion_future_t handle);

  /**
   * @see Legion::Future::get_void_result()
   */
  void
  resilient_legion_future_get_void_result(resilient_legion_future_t handle);

  /**
   * @see Legion::Future::wait
   */
  void
  resilient_legion_future_wait(resilient_legion_future_t handle, 
                     bool silence_warnings /* = false */,
                     const char *warning_string /* = NULL */);

  /**
   * @see Legion::Future::is_empty()
   */
  bool
  resilient_legion_future_is_empty(resilient_legion_future_t handle,
                         bool block /* = false */);

  /**
   * @see Legion::Future::is_ready()
   */
  bool
  resilient_legion_future_is_ready(resilient_legion_future_t handle);

  /**
   * @see Legion::Future::is_ready()
   */
  bool
  resilient_legion_future_is_ready_subscribe(resilient_legion_future_t handle, bool subscribe);

  /**
   * @see Legion::Future::get_untyped_pointer()
   */
  const void *
  resilient_legion_future_get_untyped_pointer(resilient_legion_future_t handle);

  /**
   * @see Legion::Future::get_untyped_size()
   */
  size_t
  resilient_legion_future_get_untyped_size(resilient_legion_future_t handle);

  /**
   * @see Legion::Future::get_metadata(size_t *size)
   */
  const void *
  resilient_legion_future_get_metadata(resilient_legion_future_t handle, size_t *size/*=NULL*/);

  // -----------------------------------------------------------------------
  // Future Map Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::FutureMap::FutureMap()
   */
  resilient_legion_future_map_t
  resilient_legion_future_map_copy(resilient_legion_future_map_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::FutureMap::~FutureMap()
   */
  void
  resilient_legion_future_map_destroy(resilient_legion_future_map_t handle);

  /**
   * @see Legion::FutureMap::wait_all_results()
   */
  void
  resilient_legion_future_map_wait_all_results(resilient_legion_future_map_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Future::get_future()
   */
  resilient_legion_future_t
  resilient_legion_future_map_get_future(resilient_legion_future_map_t handle,
                               resilient_legion_domain_point_t point);

  /**
   * @see Legion::FutureMap::get_future_map_domain
   */
  resilient_legion_domain_t
  resilient_legion_future_map_get_domain(resilient_legion_future_map_t handle);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::reduce_future_map
   */
  resilient_legion_future_t
  resilient_legion_future_map_reduce(resilient_legion_runtime_t runtime,
                           resilient_legion_context_t ctx,
                           resilient_legion_future_map_t handle,
                           resilient_legion_reduction_op_id_t redop,
                           bool deterministic,
                           resilient_legion_mapper_id_t map_id,
                           resilient_legion_mapping_tag_id_t tag);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::construct_future_map
   */
  resilient_legion_future_map_t
  resilient_legion_future_map_construct_from_buffers(resilient_legion_runtime_t runtime,
                                           resilient_legion_context_t ctx,
                                           resilient_legion_domain_t domain,
                                           resilient_legion_domain_point_t *points,
                                           resilient_legion_untyped_buffer_t *buffers,
                                           size_t num_points,
                                           bool collective,
                                           resilient_legion_sharding_id_t sid,
                                           bool implicit_sharding);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::construct_future_map
   */
  resilient_legion_future_map_t
  resilient_legion_future_map_construct_from_futures(resilient_legion_runtime_t runtime,
                                           resilient_legion_context_t ctx,
                                           resilient_legion_domain_t domain,
                                           resilient_legion_domain_point_t *points,
                                           resilient_legion_future_t *futures,
                                           size_t num_futures,
                                           bool collective,
                                           resilient_legion_sharding_id_t sid,
                                           bool implicit_sharding);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::transform_future_map
   */
  resilient_legion_future_map_t
  resilient_legion_future_map_transform(resilient_legion_runtime_t runtime,
                              resilient_legion_context_t ctx,
                              resilient_legion_future_map_t fm,
                              resilient_legion_index_space_t new_domain,
                              resilient_legion_point_transform_functor_t functor,
                              bool take_ownership);

  // -----------------------------------------------------------------------
  // Deferred Buffer Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::DeferredBuffer::DeferredBuffer()
   */
#define BUFFER_CREATE(DIM) \
  resilient_legion_deferred_buffer_char_##DIM##d_t \
  resilient_legion_deferred_buffer_char_##DIM##d_create( \
      resilient_legion_rect_##DIM##d_t bounds, \
      resilient_legion_memory_kind_t kind, \
      char *initial_value);
  LEGION_FOREACH_N(BUFFER_CREATE)
#undef BUFFER_CREATE

  /*
   * @see Legion::DeferredBuffer::ptr()
   */
#define BUFFER_PTR(DIM) \
  char* \
  resilient_legion_deferred_buffer_char_##DIM##d_ptr( \
      resilient_legion_deferred_buffer_char_##DIM##d_t buffer, \
      resilient_legion_point_##DIM##d_t p);
  LEGION_FOREACH_N(BUFFER_PTR)
#undef BUFFER_PTR

  /*
   * @see Legion::DeferredBuffer::~DeferredBuffer()
   */
#define BUFFER_DESTROY(DIM) \
  void \
  resilient_legion_deferred_buffer_char_##DIM##d_destroy( \
      resilient_legion_deferred_buffer_char_##DIM##d_t buffer);
  LEGION_FOREACH_N(BUFFER_DESTROY)
#undef BUFFER_DESTROY

  // -----------------------------------------------------------------------
  // Task Launch Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::TaskLauncher::TaskLauncher()
   */
  resilient_legion_task_launcher_t
  resilient_legion_task_launcher_create(
    resilient_legion_task_id_t tid,
    resilient_legion_untyped_buffer_t arg,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t tag /* = 0 */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::TaskLauncher::TaskLauncher()
   */
  resilient_legion_task_launcher_t
  resilient_legion_task_launcher_create_from_buffer(
    resilient_legion_task_id_t tid,
    const void *buffer,
    size_t buffer_size,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::TaskLauncher::~TaskLauncher()
   */
  void
  resilient_legion_task_launcher_destroy(resilient_legion_task_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::execute_task()
   */
  resilient_legion_future_t
  resilient_legion_task_launcher_execute(resilient_legion_runtime_t runtime,
                               resilient_legion_context_t ctx,
                               resilient_legion_task_launcher_t launcher);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::execute_task()
   */
  resilient_legion_future_t
  resilient_legion_task_launcher_execute_outputs(resilient_legion_runtime_t runtime,
                                       resilient_legion_context_t ctx,
                                       resilient_legion_task_launcher_t launcher,
                                       resilient_legion_output_requirement_t *reqs,
                                       size_t reqs_size);

  /**
   * @see Legion::TaskLauncher::add_region_requirement()
   */
  unsigned
  resilient_legion_task_launcher_add_region_requirement_logical_region(
    resilient_legion_task_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::TaskLauncher::add_region_requirement()
   */
  unsigned
  resilient_legion_task_launcher_add_region_requirement_logical_region_reduction(
    resilient_legion_task_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_reduction_op_id_t redop,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::TaskLauncher::region_requirements
   */
  void
  resilient_legion_task_launcher_set_region_requirement_logical_region(
    resilient_legion_task_launcher_t launcher,
    unsigned idx,
    resilient_legion_logical_region_t handle,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::TaskLauncher::region_requirements
   */
  void
  resilient_legion_task_launcher_set_region_requirement_logical_region_reduction(
    resilient_legion_task_launcher_t launcher,
    unsigned idx,
    resilient_legion_logical_region_t handle,
    resilient_legion_reduction_op_id_t redop,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::TaskLauncher::add_field()
   */
  void
  resilient_legion_task_launcher_add_field(resilient_legion_task_launcher_t launcher,
                                 unsigned idx,
                                 resilient_legion_field_id_t fid,
                                 bool inst /* = true */);

  /**
   * @see Legion::RegionRequirement::get_projection_args()
   */
  const void*
  resilient_legion_index_launcher_get_projection_args(resilient_legion_region_requirement_t requirement,
					    size_t *size);

  /**
   * @see Legion::RegionRequirement::set_projection_args()
   */
  void
  resilient_legion_index_launcher_set_projection_args(resilient_legion_index_launcher_t launcher_,
					    unsigned idx,
					    const void *args,
					    size_t size,
					    bool own);

  /**
   * @see Legion::RegionRequirement::add_flags()
   */
  void
  resilient_legion_task_launcher_add_flags(resilient_legion_task_launcher_t launcher,
                                 unsigned idx,
                                 resilient_legion_region_flags_t flags);

  /**
   * @see Legion::RegionRequirement::flags
   */
  void
  resilient_legion_task_launcher_intersect_flags(resilient_legion_task_launcher_t launcher,
                                       unsigned idx,
                                       resilient_legion_region_flags_t flags);

  /**
   * @see Legion::TaskLauncher::add_index_requirement()
   */
  unsigned
  resilient_legion_task_launcher_add_index_requirement(
    resilient_legion_task_launcher_t launcher,
    resilient_legion_index_space_t handle,
    resilient_legion_allocate_mode_t priv,
    resilient_legion_index_space_t parent,
    bool verified /* = false*/);

  /**
   * @see Legion::TaskLauncher::add_future()
   */
  void
  resilient_legion_task_launcher_add_future(resilient_legion_task_launcher_t launcher,
                                  resilient_legion_future_t future);

  /**
   * @see Legion::TaskLauncher::add_wait_barrier()
   */
  void
  resilient_legion_task_launcher_add_wait_barrier(resilient_legion_task_launcher_t launcher,
                                        resilient_legion_phase_barrier_t bar);

  /**
   * @see Legion::TaskLauncher::add_arrival_barrier()
   */
  void
  resilient_legion_task_launcher_add_arrival_barrier(resilient_legion_task_launcher_t launcher,
                                           resilient_legion_phase_barrier_t bar);

  /**
   * @see Legion::TaskLauncher::argument
   */
  void
  resilient_legion_task_launcher_set_argument(resilient_legion_task_launcher_t launcher,
                                    resilient_legion_untyped_buffer_t arg);

  /**
   * @see Legion::TaskLauncher::point
   */
  void
  resilient_legion_task_launcher_set_point(resilient_legion_task_launcher_t launcher,
                                 resilient_legion_domain_point_t point);

  /**
   * @see Legion::TaskLauncher::sharding_space
   */
  void
  resilient_legion_task_launcher_set_sharding_space(resilient_legion_task_launcher_t launcher,
                                          resilient_legion_index_space_t is);

  /**
   * @see Legion::TaskLauncher::predicate_false_future
   */
  void
  resilient_legion_task_launcher_set_predicate_false_future(resilient_legion_task_launcher_t launcher,
                                                  resilient_legion_future_t f);

  /**
   * @see Legion::TaskLauncher::predicate_false_result
   */
  void
  resilient_legion_task_launcher_set_predicate_false_result(resilient_legion_task_launcher_t launcher,
                                                  resilient_legion_untyped_buffer_t arg);

  /**
   * @see Legion::TaskLauncher::map_id
   */
  void
  resilient_legion_task_launcher_set_mapper(resilient_legion_task_launcher_t launcher,
                                  resilient_legion_mapper_id_t mapper_id); 

  /**
   * @see Legion::TaskLauncher::tag
   */
  void
  resilient_legion_task_launcher_set_mapping_tag(resilient_legion_task_launcher_t launcher,
                                       resilient_legion_mapping_tag_id_t tag);

  /**
   * @see Legion::TaskLauncher::map_arg
   */
  void
  resilient_legion_task_launcher_set_mapper_arg(resilient_legion_task_launcher_t launcher,
                                      resilient_legion_untyped_buffer_t arg);

  /**
   * @see Legion::TaskLauncher::enable_inlining
   */
  void
  resilient_legion_task_launcher_set_enable_inlining(resilient_legion_task_launcher_t launcher,
                                           bool enable_inlining);

  /**
   * @see Legion::TaskLauncher::local_task_function
   */
  void
  resilient_legion_task_launcher_set_local_function_task(resilient_legion_task_launcher_t launcher,
                                               bool local_function_task);

  /**
   * @see Legion::TaskLauncher::elide_future_return
   */
  void
  resilient_legion_task_launcher_set_elide_future_return(resilient_legion_task_launcher_t launcher,
                                               bool elide_future_return);

  /**
   * @see Legion::TaskLauncher::provenance
   */
  void
  resilient_legion_task_launcher_set_provenance(resilient_legion_task_launcher_t launcher,
                                      const char *provenance);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::IndexTaskLauncher::IndexTaskLauncher()
   */
  resilient_legion_index_launcher_t
  resilient_legion_index_launcher_create(
    resilient_legion_task_id_t tid,
    resilient_legion_domain_t domain,
    resilient_legion_untyped_buffer_t global_arg,
    resilient_legion_argument_map_t map,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    bool must /* = false */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t tag /* = 0 */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::IndexTaskLauncher::IndexTaskLauncher()
   */
  resilient_legion_index_launcher_t
  resilient_legion_index_launcher_create_from_buffer(
    resilient_legion_task_id_t tid,
    resilient_legion_domain_t domain,
    const void *buffer,
    size_t buffer_size,
    resilient_legion_argument_map_t map,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    bool must /* = false */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::IndexTaskLauncher::~IndexTaskLauncher()
   */
  void
  resilient_legion_index_launcher_destroy(resilient_legion_index_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &)
   */
  resilient_legion_future_map_t
  resilient_legion_index_launcher_execute(resilient_legion_runtime_t runtime,
                               resilient_legion_context_t ctx,
                               resilient_legion_index_launcher_t launcher);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &, ReductionOpID)
   */
  resilient_legion_future_t
  resilient_legion_index_launcher_execute_reduction(resilient_legion_runtime_t runtime,
                                          resilient_legion_context_t ctx,
                                          resilient_legion_index_launcher_t launcher,
                                          resilient_legion_reduction_op_id_t redop);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &, std::vector<OutputRequirement>*)
   */
  resilient_legion_future_map_t
  resilient_legion_index_launcher_execute_outputs(resilient_legion_runtime_t runtime,
                                        resilient_legion_context_t ctx,
                                        resilient_legion_index_launcher_t launcher,
                                        resilient_legion_output_requirement_t *reqs,
                                        size_t reqs_size);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &, ReductionOpID)
   */
  resilient_legion_future_t
  resilient_legion_index_launcher_execute_deterministic_reduction(resilient_legion_runtime_t runtime,
                                                        resilient_legion_context_t ctx,
                                                        resilient_legion_index_launcher_t launcher,
                                                        resilient_legion_reduction_op_id_t redop,
                                                        bool deterministic);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &, ReductionOpID, std::vector<OutputRequirement>*)
   */
  resilient_legion_future_t
  resilient_legion_index_launcher_execute_reduction_and_outputs(resilient_legion_runtime_t runtime,
                                                      resilient_legion_context_t ctx,
                                                      resilient_legion_index_launcher_t launcher,
                                                      resilient_legion_reduction_op_id_t redop,
                                                      bool deterministic,
                                                      resilient_legion_output_requirement_t *reqs,
                                                      size_t reqs_size);

  /**
   * @see Legion::IndexTaskLauncher::add_region_requirement()
   */
  unsigned
  resilient_legion_index_launcher_add_region_requirement_logical_region(
    resilient_legion_index_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexTaskLauncher::add_region_requirement()
   */
  unsigned
  resilient_legion_index_launcher_add_region_requirement_logical_partition(
    resilient_legion_index_launcher_t launcher,
    resilient_legion_logical_partition_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexTaskLauncher::add_region_requirement()
   */
  unsigned
  resilient_legion_index_launcher_add_region_requirement_logical_region_reduction(
    resilient_legion_index_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_reduction_op_id_t redop,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexTaskLauncher::add_region_requirement()
   */
  unsigned
  resilient_legion_index_launcher_add_region_requirement_logical_partition_reduction(
    resilient_legion_index_launcher_t launcher,
    resilient_legion_logical_partition_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_reduction_op_id_t redop,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexTaskLauncher::region_requirements
   */
  void
  resilient_legion_index_launcher_set_region_requirement_logical_region(
    resilient_legion_index_launcher_t launcher,
    unsigned idx,
    resilient_legion_logical_region_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexTaskLauncher::region_requirements
   */
  void
  resilient_legion_index_launcher_set_region_requirement_logical_partition(
    resilient_legion_index_launcher_t launcher,
    unsigned idx,
    resilient_legion_logical_partition_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexTaskLauncher::region_requirements
   */
  void
  resilient_legion_index_launcher_set_region_requirement_logical_region_reduction(
    resilient_legion_index_launcher_t launcher,
    unsigned idx,
    resilient_legion_logical_region_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_reduction_op_id_t redop,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexTaskLauncher::region_requirements
   */
  void
  resilient_legion_index_launcher_set_region_requirement_logical_partition_reduction(
    resilient_legion_index_launcher_t launcher,
    unsigned idx,
    resilient_legion_logical_partition_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_reduction_op_id_t redop,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexLaunchxer::add_field()
   */
  void
  resilient_legion_index_launcher_add_field(resilient_legion_index_launcher_t launcher,
                                 unsigned idx,
                                 resilient_legion_field_id_t fid,
                                 bool inst /* = true */);

  /**
   * @see Legion::RegionRequirement::add_flags()
   */
  void
  resilient_legion_index_launcher_add_flags(resilient_legion_index_launcher_t launcher,
                                  unsigned idx,
                                  resilient_legion_region_flags_t flags);

  /**
   * @see Legion::RegionRequirement::flags
   */
  void
  resilient_legion_index_launcher_intersect_flags(resilient_legion_index_launcher_t launcher,
                                        unsigned idx,
                                        resilient_legion_region_flags_t flags);

  /**
   * @see Legion::IndexTaskLauncher::add_index_requirement()
   */
  unsigned
  resilient_legion_index_launcher_add_index_requirement(
    resilient_legion_index_launcher_t launcher,
    resilient_legion_index_space_t handle,
    resilient_legion_allocate_mode_t priv,
    resilient_legion_index_space_t parent,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexTaskLauncher::add_future()
   */
  void
  resilient_legion_index_launcher_add_future(resilient_legion_index_launcher_t launcher,
                                   resilient_legion_future_t future);

  /**
   * @see Legion::IndexTaskLauncher::add_wait_barrier()
   */
  void
  resilient_legion_index_launcher_add_wait_barrier(resilient_legion_index_launcher_t launcher,
                                         resilient_legion_phase_barrier_t bar);

  /**
   * @see Legion::IndexTaskLauncher::add_arrival_barrier()
   */
  void
  resilient_legion_index_launcher_add_arrival_barrier(resilient_legion_index_launcher_t launcher,
                                            resilient_legion_phase_barrier_t bar);

  /**
   * @see Legion::IndexTaskLauncher::point_futures
   */
  void
  resilient_legion_index_launcher_add_point_future(resilient_legion_index_launcher_t launcher,
                                         resilient_legion_argument_map_t map);

  /**
   * @see Legion::IndexTaskLauncher::global_arg
   */
  void
  resilient_legion_index_launcher_set_global_arg(resilient_legion_index_launcher_t launcher,
                                       resilient_legion_untyped_buffer_t global_arg);

  /**
   * @see Legion::IndexTaskLauncher::sharding_space
   */
  void
  resilient_legion_index_launcher_set_sharding_space(resilient_legion_index_launcher_t launcher,
                                           resilient_legion_index_space_t is);

  /**
   * @see Legion::IndexTaskLauncher::map_id
   */
  void
  resilient_legion_index_launcher_set_mapper(resilient_legion_index_launcher_t launcher,
                                   resilient_legion_mapper_id_t mapper_id); 

  /**
   * @see Legion::IndexTaskLauncher::tag
   */
  void
  resilient_legion_index_launcher_set_mapping_tag(resilient_legion_index_launcher_t launcher,
                                        resilient_legion_mapping_tag_id_t tag);

  /**
   * @see Legion::IndexTaskLauncher::map_arg
   */
  void
  resilient_legion_index_launcher_set_mapper_arg(resilient_legion_index_launcher_t launcher,
                                       resilient_legion_untyped_buffer_t map_arg);

  /**
   * @see Legion::IndexTaskLauncher::elide_future_return
   */
  void
  resilient_legion_index_launcher_set_elide_future_return(resilient_legion_index_launcher_t launcher,
                                                bool elide_future_return);

  /**
   * @see Legion::IndexTaskLauncher::provenance
   */
  void
  resilient_legion_index_launcher_set_provenance(resilient_legion_index_launcher_t launcher,
                                       const char *provenance);

  /**
   * @see Legion::IndexTaskLauncher::concurrent
   */
  void
  resilient_legion_index_launcher_set_concurrent(resilient_legion_index_launcher_t launcher,
                                       bool concurrent);

  // -----------------------------------------------------------------------
  // Inline Mapping Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::InlineLauncher::InlineLauncher()
   */
  resilient_legion_inline_launcher_t
  resilient_legion_inline_launcher_create_logical_region(
    resilient_legion_logical_region_t handle,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t region_tag /* = 0 */,
    bool verified /* = false*/,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::InlineLauncher::~InlineLauncher()
   */
  void
  resilient_legion_inline_launcher_destroy(resilient_legion_inline_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::map_region()
   */
  resilient_legion_physical_region_t
  resilient_legion_inline_launcher_execute(resilient_legion_runtime_t runtime,
                                 resilient_legion_context_t ctx,
                                 resilient_legion_inline_launcher_t launcher);

  /**
   * @see Legion::InlineLauncher::add_field()
   */
  void
  resilient_legion_inline_launcher_add_field(resilient_legion_inline_launcher_t launcher,
                                   resilient_legion_field_id_t fid,
                                   bool inst /* = true */);

  /**
   * @see Legion::InlineLauncher::map_arg
   */
  void
  resilient_legion_inline_launcher_set_mapper_arg(resilient_legion_inline_launcher_t launcher,
                                        resilient_legion_untyped_buffer_t arg);

  /**
   * @see Legion::InlineLauncher::provenance
   */
  void
  resilient_legion_inline_launcher_set_provenance(resilient_legion_inline_launcher_t launcher,
                                        const char *provenance);

  /**
   * @see Legion::Runtime::remap_region()
   */
  void
  resilient_legion_runtime_remap_region(resilient_legion_runtime_t runtime,
                              resilient_legion_context_t ctx,
                              resilient_legion_physical_region_t region);

  /**
   * @see Legion::Runtime::unmap_region()
   */
  void
  resilient_legion_runtime_unmap_region(resilient_legion_runtime_t runtime,
                              resilient_legion_context_t ctx,
                              resilient_legion_physical_region_t region);

  /**
   * @see Legion::Runtime::unmap_all_regions()
   */
  void
  resilient_legion_runtime_unmap_all_regions(resilient_legion_runtime_t runtime,
                                   resilient_legion_context_t ctx); 

  // -----------------------------------------------------------------------
  // Fill Field Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Runtime::fill_field()
   */
  void
  resilient_legion_runtime_fill_field(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_logical_region_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    const void *value,
    size_t value_size,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */);

  /**
   * @see Legion::Runtime::fill_field()
   */
  void
  resilient_legion_runtime_fill_field_future(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_logical_region_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    resilient_legion_future_t f,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::FillLauncher::FillLauncher()
   */
  resilient_legion_fill_launcher_t
  resilient_legion_fill_launcher_create(
    resilient_legion_logical_region_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    const void *value,
    size_t value_size,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t tag /* = 0 */);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::FillLauncher::FillLauncher()
   */
  resilient_legion_fill_launcher_t
  resilient_legion_fill_launcher_create_from_future(
    resilient_legion_logical_region_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    resilient_legion_future_t f,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::FillLauncher::~FillLauncher()
   */
  void
  resilient_legion_fill_launcher_destroy(resilient_legion_fill_launcher_t handle);

  /**
   * @see Legion::FillLauncher::add_field()
   */
  void
  resilient_legion_fill_launcher_add_field(resilient_legion_fill_launcher_t handle,
                                 resilient_legion_field_id_t fid);

  /**
   * @see Legion::Runtime::fill_fields()
   */
  void
  resilient_legion_fill_launcher_execute(resilient_legion_runtime_t runtime,
                               resilient_legion_context_t ctx,
                               resilient_legion_fill_launcher_t launcher);

  /**
   * @see Legion::FillLauncher::point
   */
  void
  resilient_legion_fill_launcher_set_point(resilient_legion_fill_launcher_t launcher,
                                 resilient_legion_domain_point_t point);

  /**
   * @see Legion::FillLauncher::sharding_space
   */
  void resilient_legion_fill_launcher_set_sharding_space(resilient_legion_fill_launcher_t launcher,
                                               resilient_legion_index_space_t space);

  /**
   * @see Legion::FillLauncher::map_arg
   */
  void
  resilient_legion_fill_launcher_set_mapper_arg(resilient_legion_fill_launcher_t launcher,
                                      resilient_legion_untyped_buffer_t arg);

  /**
   * @see Legion::FillLauncher::provenance
   */
  void
  resilient_legion_fill_launcher_set_provenance(resilient_legion_fill_launcher_t launcher,
                                      const char *provenance);

  // -----------------------------------------------------------------------
  // Index Fill Field Operations
  // -----------------------------------------------------------------------
  
  /**
   * @see Legion::Runtime::fill_field()
   * Same as above except using index fills
   */
  void
  resilient_legion_runtime_index_fill_field(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_logical_partition_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    const void *value,
    size_t value_size,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @see Legion::Runtime::fill_field()
   * Same as above except using index fills
   */
  void
  resilient_legion_runtime_index_fill_field_with_space(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t space,
    resilient_legion_logical_partition_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    const void *value,
    size_t value_size,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @see Legion::Runtime::fill_field()
   * Same as above except using index fills
   */
  void
  resilient_legion_runtime_index_fill_field_with_domain(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_domain_t domain,
    resilient_legion_logical_partition_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    const void *value,
    size_t value_size,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @see Legion::Runtime::fill_field()
   * Same as above except using index fills
   */
  void
  resilient_legion_runtime_index_fill_field_future(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_logical_partition_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    resilient_legion_future_t f,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @see Legion::Runtime::fill_field()
   * Same as above except using index fills
   */
  void
  resilient_legion_runtime_index_fill_field_future_with_space(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_index_space_t space,
    resilient_legion_logical_partition_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    resilient_legion_future_t f,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @see Legion::Runtime::fill_field()
   * Same as above except using index fills
   */
  void
  resilient_legion_runtime_index_fill_field_future_with_domain(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_domain_t domain,
    resilient_legion_logical_partition_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    resilient_legion_future_t f,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::IndexFillLauncher()
   */
  resilient_legion_index_fill_launcher_t
  resilient_legion_index_fill_launcher_create_with_space(
    resilient_legion_index_space_t space,
    resilient_legion_logical_partition_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    const void *value,
    size_t value_size,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::IndexFillLauncher()
   */
  resilient_legion_index_fill_launcher_t
  resilient_legion_index_fill_launcher_create_with_domain(
    resilient_legion_domain_t domain,
    resilient_legion_logical_partition_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    const void *value,
    size_t value_size,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::IndexFillLauncher()
   */
  resilient_legion_index_fill_launcher_t
  resilient_legion_index_fill_launcher_create_from_future_with_space(
    resilient_legion_index_space_t space,
    resilient_legion_logical_partition_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    resilient_legion_future_t future,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::IndexFillLauncher()
   */
  resilient_legion_index_fill_launcher_t
  resilient_legion_index_fill_launcher_create_from_future_with_domain(
    resilient_legion_domain_t domain,
    resilient_legion_logical_partition_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_id_t fid,
    resilient_legion_future_t future,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);
  
  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::IndexFillLauncher::~IndexFillLauncher()
   */
  void
  resilient_legion_index_fill_launcher_destroy(resilient_legion_index_fill_launcher_t handle);

  /**
   * @see Legion::IndexFillLauncher::add_field()
   */
  void
  resilient_legion_index_fill_launcher_add_field(resilient_legion_fill_launcher_t handle,
                                       resilient_legion_field_id_t fid);

  /**
   * @see Legion::Runtime::fill_fields()
   */
  void
  resilient_legion_index_fill_launcher_execute(resilient_legion_runtime_t runtime,
                                     resilient_legion_context_t ctx,
                                     resilient_legion_index_fill_launcher_t launcher);

  /**
   * @see Legion::IndexFillLauncher::sharding_space
   */
  void resilient_legion_index_fill_launcher_set_sharding_space(resilient_legion_index_fill_launcher_t launcher,
                                                     resilient_legion_index_space_t space);

  /**
   * @see Legion::IndexFillLauncher::map_arg
   */
  void
  resilient_legion_index_fill_launcher_set_mapper_arg(resilient_legion_index_fill_launcher_t launcher,
                                            resilient_legion_untyped_buffer_t arg);

  /**
   * @see Legion::IndexFillLauncher::provenance
   */
  void
  resilient_legion_index_fill_launcher_set_provenance(resilient_legion_index_fill_launcher_t launcher,
                                            const char *provenance);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Fill::requirement
   */
  resilient_legion_region_requirement_t
  resilient_legion_fill_get_requirement(resilient_legion_fill_t fill);

  // -----------------------------------------------------------------------
  // Discard Operation
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::DiscardLauncher::DiscardLauncher()
   */
  resilient_legion_discard_launcher_t
  resilient_legion_discard_launcher_create(
    resilient_legion_logical_region_t handle,
    resilient_legion_logical_region_t parent);

  /**
   * @param hanlde Caller must have ownership of parameter 'handle'
   *
   * @see Legion::DiscardLauncher::~DiscardLauncher()
   */
  void
  resilient_legion_discard_launcher_destroy(resilient_legion_discard_launcher_t handle);

  /**
   * @see Legion::DiscardLauncher::add_field()
   */
  void
  resilient_legion_discard_launcher_add_field(resilient_legion_discard_launcher_t handle,
                                    resilient_legion_field_id_t fid);

  /**
   * @see Legion::Runtime::discard_fields()
   */
  void
  resilient_legion_discard_launcher_execute(resilient_legion_runtime_t runtime,
                                  resilient_legion_context_t ctx,
                                  resilient_legion_discard_launcher_t launcher);

  /**
   * @see Legion::DiscardLauncher::provenance
   */
  void
  resilient_legion_discard_launcher_set_provenance(resilient_legion_discard_launcher_t launcher,
                                         const char *provenance);

  // -----------------------------------------------------------------------
  // File Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   */
  resilient_legion_field_map_t
  resilient_legion_field_map_create(void);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   */
  void
  resilient_legion_field_map_destroy(resilient_legion_field_map_t handle);

  void
  resilient_legion_field_map_insert(resilient_legion_field_map_t handle,
                          resilient_legion_field_id_t key,
                          const char *value);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::attach_hdf5()
   */
  resilient_legion_physical_region_t
  resilient_legion_runtime_attach_hdf5(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    const char *filename,
    resilient_legion_logical_region_t handle,
    resilient_legion_logical_region_t parent,
    resilient_legion_field_map_t field_map,
    resilient_legion_file_mode_t mode);

  /**
   * @see Legion::Runtime::detach_hdf5()
   */
  void
  resilient_legion_runtime_detach_hdf5(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    resilient_legion_physical_region_t region);

  // -----------------------------------------------------------------------
  // Copy Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::CopyLauncher::CopyLauncher()
   */
  resilient_legion_copy_launcher_t
  resilient_legion_copy_launcher_create(
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::CopyLauncher::~CopyLauncher()
   */
  void
  resilient_legion_copy_launcher_destroy(resilient_legion_copy_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::issue_copy_operation()
   */
  void
  resilient_legion_copy_launcher_execute(resilient_legion_runtime_t runtime,
                               resilient_legion_context_t ctx,
                               resilient_legion_copy_launcher_t launcher);

  /**
   * @see Legion::CopyLauncher::add_copy_requirements()
   */
  unsigned
  resilient_legion_copy_launcher_add_src_region_requirement_logical_region(
    resilient_legion_copy_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::CopyLauncher::add_copy_requirements()
   */
  unsigned
  resilient_legion_copy_launcher_add_dst_region_requirement_logical_region(
    resilient_legion_copy_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::CopyLauncher::add_region_requirement()
   */
  unsigned
  resilient_legion_copy_launcher_add_dst_region_requirement_logical_region_reduction(
    resilient_legion_copy_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_reduction_op_id_t redop,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::CopyLauncher::add_src_indirect_field()
   */
  unsigned
  resilient_legion_copy_launcher_add_src_indirect_region_requirement_logical_region(
    resilient_legion_copy_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_field_id_t fid,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool is_range_indirection /* = false */,
    bool verified /* = false*/);

  /**
   * @see Legion::CopyLauncher::add_dst_indirect_field()
   */
  unsigned
  resilient_legion_copy_launcher_add_dst_indirect_region_requirement_logical_region(
    resilient_legion_copy_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_field_id_t fid,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool is_range_indirection /* = false */,
    bool verified /* = false*/);

  /**
   * @see Legion::CopyLauncher::add_src_field()
   */
  void
  resilient_legion_copy_launcher_add_src_field(resilient_legion_copy_launcher_t launcher,
                                     unsigned idx,
                                     resilient_legion_field_id_t fid,
                                     bool inst /* = true */);

  /**
   * @see Legion::CopyLauncher::add_dst_field()
   */
  void
  resilient_legion_copy_launcher_add_dst_field(resilient_legion_copy_launcher_t launcher,
                                     unsigned idx,
                                     resilient_legion_field_id_t fid,
                                     bool inst /* = true */);

  /**
   * @see Legion::CopyLauncher::add_wait_barrier()
   */
  void
  resilient_legion_copy_launcher_add_wait_barrier(resilient_legion_copy_launcher_t launcher,
                                        resilient_legion_phase_barrier_t bar);

  /**
   * @see Legion::CopyLauncher::add_arrival_barrier()
   */
  void
  resilient_legion_copy_launcher_add_arrival_barrier(resilient_legion_copy_launcher_t launcher,
                                           resilient_legion_phase_barrier_t bar);

  /**
   * @see Legion::CopyLauncher::possible_src_indirect_out_of_range
   */
  void
  resilient_legion_copy_launcher_set_possible_src_indirect_out_of_range(
      resilient_legion_copy_launcher_t launcher, bool flag);

  /**
   * @see Legion::CopyLauncher::possible_dst_indirect_out_of_range
   */
  void
  resilient_legion_copy_launcher_set_possible_dst_indirect_out_of_range(
      resilient_legion_copy_launcher_t launcher, bool flag);

  /**
   * @see Legion::CopyLauncher::point
   */
  void
  resilient_legion_copy_launcher_set_point(resilient_legion_copy_launcher_t launcher,
                                 resilient_legion_domain_point_t point);

  /**
   * @see Legion::CopyLauncher::sharding_space
   */
  void resilient_legion_copy_launcher_set_sharding_space(resilient_legion_copy_launcher_t launcher,
                                               resilient_legion_index_space_t space);

  /**
   * @see Legion::CopyLauncher::map_arg
   */
  void
  resilient_legion_copy_launcher_set_mapper_arg(resilient_legion_copy_launcher_t launcher,
                                      resilient_legion_untyped_buffer_t arg);

  /**
   * @see Legion::CopyLauncher::provenance
   */
  void
  resilient_legion_copy_launcher_set_provenance(resilient_legion_copy_launcher_t launcher,
                                      const char *provenance);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Copy::src_requirements
   * @see Legion::Copy::dst_requirements
   * @see Legion::Copy::src_indirect_requirements
   * @see Legion::Copy::dst_indirect_requirements
   */
  resilient_legion_region_requirement_t
  resilient_legion_copy_get_requirement(resilient_legion_copy_t copy, unsigned idx);

  // -----------------------------------------------------------------------
  // Index Copy Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::IndexCopyLauncher::IndexCopyLauncher()
   */
  resilient_legion_index_copy_launcher_t
  resilient_legion_index_copy_launcher_create(
    resilient_legion_domain_t domain,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::IndexCopyLauncher::~IndexCopyLauncher()
   */
  void
  resilient_legion_index_copy_launcher_destroy(resilient_legion_index_copy_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::issue_index_copy_operation()
   */
  void
  resilient_legion_index_copy_launcher_execute(resilient_legion_runtime_t runtime,
                                     resilient_legion_context_t ctx,
                                     resilient_legion_index_copy_launcher_t launcher);

  /**
   * @see Legion::IndexCopyLauncher::add_copy_requirements()
   */
  unsigned
  resilient_legion_index_copy_launcher_add_src_region_requirement_logical_region(
    resilient_legion_index_copy_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_copy_requirements()
   */
  unsigned
  resilient_legion_index_copy_launcher_add_dst_region_requirement_logical_region(
    resilient_legion_index_copy_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_copy_requirements()
   */
  unsigned
  resilient_legion_index_copy_launcher_add_src_region_requirement_logical_partition(
    resilient_legion_index_copy_launcher_t launcher,
    resilient_legion_logical_partition_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_copy_requirements()
   */
  unsigned
  resilient_legion_index_copy_launcher_add_dst_region_requirement_logical_partition(
    resilient_legion_index_copy_launcher_t launcher,
    resilient_legion_logical_partition_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_privilege_mode_t priv,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_copy_requirements()
   */
  unsigned
  resilient_legion_index_copy_launcher_add_dst_region_requirement_logical_region_reduction(
    resilient_legion_index_copy_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_reduction_op_id_t redop,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_copy_requirements()
   */
  unsigned
  resilient_legion_index_copy_launcher_add_dst_region_requirement_logical_partition_reduction(
    resilient_legion_index_copy_launcher_t launcher,
    resilient_legion_logical_partition_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_reduction_op_id_t redop,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_src_indirect_field()
   */
  unsigned
  resilient_legion_index_copy_launcher_add_src_indirect_region_requirement_logical_region(
    resilient_legion_index_copy_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_field_id_t fid,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool is_range_indirection /* = false */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_dst_indirect_field()
   */
  unsigned
  resilient_legion_index_copy_launcher_add_dst_indirect_region_requirement_logical_region(
    resilient_legion_index_copy_launcher_t launcher,
    resilient_legion_logical_region_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_field_id_t fid,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool is_range_indirection /* = false */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_src_indirect_field()
   */
  unsigned
  resilient_legion_index_copy_launcher_add_src_indirect_region_requirement_logical_partition(
    resilient_legion_index_copy_launcher_t launcher,
    resilient_legion_logical_partition_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_field_id_t fid,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool is_range_indirection /* = false */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_dst_indirect_field()
   */
  unsigned
  resilient_legion_index_copy_launcher_add_dst_indirect_region_requirement_logical_partition(
    resilient_legion_index_copy_launcher_t launcher,
    resilient_legion_logical_partition_t handle,
    resilient_legion_projection_id_t proj /* = 0 */,
    resilient_legion_field_id_t fid,
    resilient_legion_coherence_property_t prop,
    resilient_legion_logical_region_t parent,
    resilient_legion_mapping_tag_id_t tag /* = 0 */,
    bool is_range_indirection /* = false */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_src_field()
   */
  void
  resilient_legion_index_copy_launcher_add_src_field(resilient_legion_index_copy_launcher_t launcher,
                                           unsigned idx,
                                           resilient_legion_field_id_t fid,
                                           bool inst /* = true */);

  /**
   * @see Legion::IndexCopyLauncher::add_dst_field()
   */
  void
  resilient_legion_index_copy_launcher_add_dst_field(resilient_legion_index_copy_launcher_t launcher,
                                           unsigned idx,
                                           resilient_legion_field_id_t fid,
                                           bool inst /* = true */);

  /**
   * @see Legion::IndexCopyLauncher::add_wait_barrier()
   */
  void
  resilient_legion_index_copy_launcher_add_wait_barrier(resilient_legion_index_copy_launcher_t launcher,
                                              resilient_legion_phase_barrier_t bar);

  /**
   * @see Legion::IndexCopyLauncher::add_arrival_barrier()
   */
  void
  resilient_legion_index_copy_launcher_add_arrival_barrier(resilient_legion_index_copy_launcher_t launcher,
                                                 resilient_legion_phase_barrier_t bar);

  /**
   * @see Legion::IndexCopyLauncher::possible_src_indirect_out_of_range
   */
  void
  resilient_legion_index_copy_launcher_set_possible_src_indirect_out_of_range(
      resilient_legion_index_copy_launcher_t launcher, bool flag);

  /**
   * @see Legion::IndexCopyLauncher::possible_dst_indirect_out_of_range
   */
  void
  resilient_legion_index_copy_launcher_set_possible_dst_indirect_out_of_range(
      resilient_legion_index_copy_launcher_t launcher, bool flag);

  /**
   * @see Legion::IndexCopyLauncher::sharding_space
   */
  void
  resilient_legion_index_copy_launcher_set_sharding_space(resilient_legion_index_copy_launcher_t launcher,
                                                resilient_legion_index_space_t is);

  /**
   * @see Legion::IndexCopyLauncher::map_arg
   */
  void
  resilient_legion_index_copy_launcher_set_mapper_arg(resilient_legion_index_copy_launcher_t launcher,
                                            resilient_legion_untyped_buffer_t arg);

  /**
   * @see Legion::IndexCopyLauncher::provenance
   */
  void
  resilient_legion_index_copy_launcher_set_provenance(resilient_legion_index_copy_launcher_t launcher,
                                            const char *provenance);

  // -----------------------------------------------------------------------
  // Acquire Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::AcquireLauncher::AcquireLauncher()
   */
  resilient_legion_acquire_launcher_t
  resilient_legion_acquire_launcher_create(
    resilient_legion_logical_region_t logical_region,
    resilient_legion_logical_region_t parent_region,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::AcquireLauncher::~AcquireLauncher()
   */
  void
  resilient_legion_acquire_launcher_destroy(resilient_legion_acquire_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::issue_acquire()
   */
  void
  resilient_legion_acquire_launcher_execute(resilient_legion_runtime_t runtime,
                                  resilient_legion_context_t ctx,
                                  resilient_legion_acquire_launcher_t launcher);

  /**
   * @see Legion::AcquireLauncher::add_field()
   */
  void
  resilient_legion_acquire_launcher_add_field(resilient_legion_acquire_launcher_t launcher,
                                    resilient_legion_field_id_t fid);

  /**
   * @see Legion::AcquireLauncher::add_wait_barrier()
   */
  void
  resilient_legion_acquire_launcher_add_wait_barrier(resilient_legion_acquire_launcher_t launcher,
                                           resilient_legion_phase_barrier_t bar);

  /**
   * @see Legion::AcquireLauncher::add_arrival_barrier()
   */
  void
  resilient_legion_acquire_launcher_add_arrival_barrier(
    resilient_legion_acquire_launcher_t launcher,
    resilient_legion_phase_barrier_t bar);

  /**
   * @see Legion::AcquireLauncher::sharding_space
   */
  void 
  resilient_legion_acquire_launcher_set_sharding_space(resilient_legion_acquire_launcher_t launcher,
                                             resilient_legion_index_space_t space);

  /**
   * @see Legion::AcquireLauncher::map_arg
   */
  void
  resilient_legion_acquire_launcher_set_mapper_arg(resilient_legion_acquire_launcher_t launcher,
                                         resilient_legion_untyped_buffer_t arg);

  /**
   * @see Legion::AcquireLauncher::provenance
   */
  void
  resilient_legion_acquire_launcher_set_provenance(resilient_legion_acquire_launcher_t launcher,
                                         const char *provenance);

  // -----------------------------------------------------------------------
  // Release Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::ReleaseLauncher::ReleaseLauncher()
   */
  resilient_legion_release_launcher_t
  resilient_legion_release_launcher_create(
    resilient_legion_logical_region_t logical_region,
    resilient_legion_logical_region_t parent_region,
    resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */,
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::ReleaseLauncher::~ReleaseLauncher()
   */
  void
  resilient_legion_release_launcher_destroy(resilient_legion_release_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::issue_release()
   */
  void
  resilient_legion_release_launcher_execute(resilient_legion_runtime_t runtime,
                                  resilient_legion_context_t ctx,
                                  resilient_legion_release_launcher_t launcher);

  /**
   * @see Legion::ReleaseLauncher::add_field()
   */
  void
  resilient_legion_release_launcher_add_field(resilient_legion_release_launcher_t launcher,
                                    resilient_legion_field_id_t fid);

  /**
   * @see Legion::ReleaseLauncher::add_wait_barrier()
   */
  void
  resilient_legion_release_launcher_add_wait_barrier(resilient_legion_release_launcher_t launcher,
                                           resilient_legion_phase_barrier_t bar);

  /**
   * @see Legion::ReleaseLauncher::add_arrival_barrier()
   */
  void
  resilient_legion_release_launcher_add_arrival_barrier(
    resilient_legion_release_launcher_t launcher,
    resilient_legion_phase_barrier_t bar);

  /**
   * @see Legion::ReleaseLauncher::sharding_space
   */
  void
  resilient_legion_release_launcher_set_sharding_space(resilient_legion_release_launcher_t launcher,
                                             resilient_legion_index_space_t space);

  /**
   * @see Legion::ReleaseLauncher::map_arg
   */
  void
  resilient_legion_release_launcher_set_mapper_arg(resilient_legion_release_launcher_t launcher,
                                         resilient_legion_untyped_buffer_t arg);

  /**
   * @see Legion::ReleaseLauncher::provenance
   */
  void
  resilient_legion_release_launcher_set_provenance(resilient_legion_release_launcher_t launcher,
                                         const char *provenance);

  // -----------------------------------------------------------------------
  // Attach/Detach Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::AttachLauncher::AttachLauncher()
   */
  resilient_legion_attach_launcher_t
  resilient_legion_attach_launcher_create(
    resilient_legion_logical_region_t logical_region,
    resilient_legion_logical_region_t parent_region,
    resilient_legion_external_resource_t resource);

  /**
   * @see Legion::AttachLauncher::attach_hdf5()
   */
  void
  resilient_legion_attach_launcher_attach_hdf5(resilient_legion_attach_launcher_t handle,
                                     const char *filename,
                                     resilient_legion_field_map_t field_map,
                                     resilient_legion_file_mode_t mode);

  /**
   * @see Legion::AttachLauncher::restricted
   */
  void
  resilient_legion_attach_launcher_set_restricted(resilient_legion_attach_launcher_t handle,
                                        bool restricted);

  /**
   * @see Legion::AttachLauncher::mapped
   */
  void
  resilient_legion_attach_launcher_set_mapped(resilient_legion_attach_launcher_t handle,
                                    bool mapped);

  /**
   * @see Legion::AttachLauncher::provenance
   */
  void
  resilient_legion_attach_launcher_set_provenance(resilient_legion_attach_launcher_t handle,
                                        const char *provenance);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::AttachLauncher::~AttachLauncher()
   */
  void
  resilient_legion_attach_launcher_destroy(resilient_legion_attach_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::attach_external_resource()
   */
  resilient_legion_physical_region_t
  resilient_legion_attach_launcher_execute(resilient_legion_runtime_t runtime,
                                 resilient_legion_context_t ctx,
                                 resilient_legion_attach_launcher_t launcher);

  /**
   * @see Legion::AttachLauncher::attach_array_soa()
   */
  void
  resilient_legion_attach_launcher_add_cpu_soa_field(resilient_legion_attach_launcher_t launcher,
                                           resilient_legion_field_id_t fid,
                                           void *base_ptr,
                                           bool column_major);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::detach_external_resource()
   */
  resilient_legion_future_t
  resilient_legion_detach_external_resource(resilient_legion_runtime_t runtime,
                                  resilient_legion_context_t ctx,
                                  resilient_legion_physical_region_t handle);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::detach_external_resource()
   */
  resilient_legion_future_t
  resilient_legion_flush_detach_external_resource(resilient_legion_runtime_t runtime,
                                        resilient_legion_context_t ctx,
                                        resilient_legion_physical_region_t handle,
                                        bool flush);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::detach_external_resource()
   */
  resilient_legion_future_t
  resilient_legion_unordered_detach_external_resource(resilient_legion_runtime_t runtime,
                                            resilient_legion_context_t ctx,
                                            resilient_legion_physical_region_t handle,
                                            bool flush,
                                            bool unordered);

  /**
   * @see Legion::Runtime;:progress_unordered_operations()
   */
  void
  resilient_legion_context_progress_unordered_operations(resilient_legion_runtime_t runtime,
                                               resilient_legion_context_t ctx);

  // -----------------------------------------------------------------------
  // Index Attach/Detach Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::IndexAttachLauncher::IndexAttachLauncher()
   */
  resilient_legion_index_attach_launcher_t
  resilient_legion_index_attach_launcher_create(
      resilient_legion_logical_region_t parent_region,
      resilient_legion_external_resource_t resource,
      bool restricted/*=true*/);

  /**
   * @see Legion::IndexAttachLauncher::restricted
   */
  void
  resilient_legion_index_attach_launcher_set_restricted(
      resilient_legion_index_attach_launcher_t handle, bool restricted);

  /**
   * @see Legion::IndexAttachLauncher::provenance
   */
  void
  resilient_legion_index_attach_launcher_set_provenance(
      resilient_legion_index_attach_launcher_t handle, const char *provenance);

  /**
   * @see Legion::IndexAttachLauncher::deduplicate_across_shards
   */
  void
  resilient_legion_index_attach_launcher_set_deduplicate_across_shards(
      resilient_legion_index_attach_launcher_t handle, bool deduplicate);

  /**
   * @see Legion::IndexAttachLauncher::attach_file
   */
  void
  resilient_legion_index_attach_launcher_attach_file(resilient_legion_index_attach_launcher_t handle,
                                           resilient_legion_logical_region_t region,
                                           const char *filename,
                                           const resilient_legion_field_id_t *fields,
                                           size_t num_fields,
                                           resilient_legion_file_mode_t mode);

  /**
   * @see Legion::IndexAttachLauncher::attach_hdf5()
   */
  void
  resilient_legion_index_attach_launcher_attach_hdf5(resilient_legion_index_attach_launcher_t handle,
                                           resilient_legion_logical_region_t region,
                                           const char *filename,
                                           resilient_legion_field_map_t field_map,
                                           resilient_legion_file_mode_t mode);

  /**
   * @see Legion::IndexAttachLauncher::attach_array_soa()
   */
  void
  resilient_legion_index_attach_launcher_attach_array_soa(resilient_legion_index_attach_launcher_t handle,
                                           resilient_legion_logical_region_t region,
                                           void *base_ptr, bool column_major,
                                           const resilient_legion_field_id_t *fields,
                                           size_t num_fields,
                                           resilient_legion_memory_t memory);

  /**
   * @see Legion::IndexAttachLauncher::attach_array_aos()
   */
  void
  resilient_legion_index_attach_launcher_attach_array_aos(resilient_legion_index_attach_launcher_t handle,
                                           resilient_legion_logical_region_t region,
                                           void *base_ptr, bool column_major,
                                           const resilient_legion_field_id_t *fields,
                                           size_t num_fields,
                                           resilient_legion_memory_t memory);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::IndexAttachLauncher::~IndexAttachLauncher()
   */
  void
  resilient_legion_index_attach_launcher_destroy(resilient_legion_index_attach_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::attach_external_resources()
   */
  resilient_legion_external_resources_t
  resilient_legion_attach_external_resources(resilient_legion_runtime_t runtime,
                                   resilient_legion_context_t ctx,
                                   resilient_legion_index_attach_launcher_t launcher);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::detach_external_resource()
   */
  resilient_legion_future_t
  resilient_legion_detach_external_resources(resilient_legion_runtime_t runtime,
                                   resilient_legion_context_t ctx,
                                   resilient_legion_external_resources_t,
                                   bool flush, bool unordered);

  // -----------------------------------------------------------------------
  // Must Epoch Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::MustEpochLauncher::MustEpochLauncher()
   */
  resilient_legion_must_epoch_launcher_t
  resilient_legion_must_epoch_launcher_create(
    resilient_legion_mapper_id_t id /* = 0 */,
    resilient_legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::MustEpochLauncher::~MustEpochLauncher()
   */
  void
  resilient_legion_must_epoch_launcher_destroy(resilient_legion_must_epoch_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::execute_must_epoch()
   */
  resilient_legion_future_map_t
  resilient_legion_must_epoch_launcher_execute(resilient_legion_runtime_t runtime,
                                     resilient_legion_context_t ctx,
                                     resilient_legion_must_epoch_launcher_t launcher);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Must_EpochLauncher::add_single_task()
   */
  void
  resilient_legion_must_epoch_launcher_add_single_task(
    resilient_legion_must_epoch_launcher_t launcher,
    resilient_legion_domain_point_t point,
    resilient_legion_task_launcher_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Must_EpochLauncher::add_index_task()
   */
  void
  resilient_legion_must_epoch_launcher_add_index_task(
    resilient_legion_must_epoch_launcher_t launcher,
    resilient_legion_index_launcher_t handle);

  /**
   * @see Legion::Must_EpochLauncher::launch_domain
   */
  void
  resilient_legion_must_epoch_launcher_set_launch_domain(
    resilient_legion_must_epoch_launcher_t launcher,
    resilient_legion_domain_t domain);

  /**
   * @see Legion::Must_EpochLauncher::launch_space
   */
  void
  resilient_legion_must_epoch_launcher_set_launch_space(
    resilient_legion_must_epoch_launcher_t launcher,
    resilient_legion_index_space_t is);

  /**
   * @see Legion::Must_EpochLauncher::provenance
   */
  void
  resilient_legion_must_epoch_launcher_set_provenance(
    resilient_legion_must_epoch_launcher_t launcher, const char *provenance);

  // -----------------------------------------------------------------------
  // Fence Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Runtime::issue_mapping_fence()
   */
  resilient_legion_future_t
  resilient_legion_runtime_issue_mapping_fence(resilient_legion_runtime_t runtime,
                                     resilient_legion_context_t ctx);

  /**
   * @see Legion::Runtime::issue_execution_fence()
   */
  resilient_legion_future_t
  resilient_legion_runtime_issue_execution_fence(resilient_legion_runtime_t runtime,
                                       resilient_legion_context_t ctx);

  // -----------------------------------------------------------------------
  // Tracing Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Runtime::begin_trace()
   */
  void
  resilient_legion_runtime_begin_trace(resilient_legion_runtime_t runtime,
                             resilient_legion_context_t ctx,
                             resilient_legion_trace_id_t tid,
                             bool logical_only);

  /**
   * @see Legion::Runtime::end_trace()
   */
  void
  resilient_legion_runtime_end_trace(resilient_legion_runtime_t runtime,
                           resilient_legion_context_t ctx,
                           resilient_legion_trace_id_t tid);

  // -----------------------------------------------------------------------
  // Frame Operations
  // -----------------------------------------------------------------------

  void
  resilient_legion_runtime_complete_frame(resilient_legion_runtime_t runtime,
                                resilient_legion_context_t ctx);

  // -----------------------------------------------------------------------
  // Tunable Variables
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::select_tunable_value()
   */
  resilient_legion_future_t
  resilient_legion_runtime_select_tunable_value(resilient_legion_runtime_t runtime,
				      resilient_legion_context_t ctx,
				      resilient_legion_tunable_id_t tid,
				      resilient_legion_mapper_id_t mapper /* = 0 */,
				      resilient_legion_mapping_tag_id_t tag /* = 0 */);

  // -----------------------------------------------------------------------
  // Miscellaneous Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Runtime::has_runtime()
   */
  bool
  resilient_legion_runtime_has_runtime(void);

  /**
   * @see Legion::Runtime::get_runtime()
   */
  resilient_legion_runtime_t
  resilient_legion_runtime_get_runtime(void);

  /**
   * @see Legion::Runtime::has_context()
   */
  bool
  resilient_legion_runtime_has_context(void);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::get_context()
   */
  resilient_legion_context_t
  resilient_legion_runtime_get_context(void);

  /**
   * IMPORTANT: This method is ONLY for use with contexts obtained via resilient_legion_runtime_get_context().
   *
   * @param handle Caller must have ownership of parameter `handle`.
   */
  void
  resilient_legion_context_destroy(resilient_legion_context_t);

  /**
   * @see Legion::Runtime::get_executing_processor()
   */
  resilient_legion_processor_t
  resilient_legion_runtime_get_executing_processor(resilient_legion_runtime_t runtime,
                                         resilient_legion_context_t ctx);

  /**
   * @see Legion::Runtime::yield()
   */
  void
  resilient_legion_runtime_yield(resilient_legion_runtime_t runtime, resilient_legion_context_t ctx);

  /**
   * @see Legion::Runtime::local_shard()
   */
  resilient_legion_shard_id_t
  resilient_legion_runtime_local_shard(resilient_legion_runtime_t runtime, resilient_legion_context_t ctx);

  resilient_legion_shard_id_t
  resilient_legion_runtime_local_shard_without_context(void);

  /**
   * @see Legion::Runtime::total_shards()
   */
  size_t
  resilient_legion_runtime_total_shards(resilient_legion_runtime_t runtime, resilient_legion_context_t ctx);

  /**
   * @param sid Must correspond to a previously registered sharding functor.
   *
   * @see Legion::ShardingFunctor::shard()
   */
  resilient_legion_shard_id_t
  resilient_legion_sharding_functor_shard(resilient_legion_sharding_id_t sid,
                                resilient_legion_domain_point_t point,
                                resilient_legion_domain_t full_space,
                                size_t total_shards);

  /**
   * @param sid Must correspond to a previously registered sharding functor.
   *            This functor must be invertible.
   * @param points Pre-allocated array to fill in with the points returned by
   *               the `invert` call. This array must be large enough to fit the
   *               output of any call to this functor's `invert`. A safe limit
   *               that will work for any functor is
   *               `resilient_legion_domain_get_volume(full_domain)`.
   * @param points_size At entry this must be the capacity of the `points`
   *                    array. At exit this value has been updated to the actual
   *                    number of returned points.
   *
   * @see Legion::ShardingFunctor::invert()
   */
  void
  resilient_legion_sharding_functor_invert(resilient_legion_sharding_id_t sid,
                                 resilient_legion_shard_id_t shard,
                                 resilient_legion_domain_t shard_domain,
                                 resilient_legion_domain_t full_domain,
                                 size_t total_shards,
                                 resilient_legion_domain_point_t *points,
                                 size_t *points_size);

  void
  resilient_legion_runtime_enable_scheduler_lock(void);

  void
  resilient_legion_runtime_disable_scheduler_lock(void);

  /**
   * @see Legion::Runtime::print_once()
   */
  void
  resilient_legion_runtime_print_once(resilient_legion_runtime_t runtime,
                            resilient_legion_context_t ctx,
                            FILE *f,
                            const char *message);
  /**
   * @see Legion::Runtime::print_once()
   */
  void
  resilient_legion_runtime_print_once_fd(resilient_legion_runtime_t runtime,
                            resilient_legion_context_t ctx,
                            int fd, const char *mode,
                            const char *message);

  // -----------------------------------------------------------------------
  // Physical Data Operations
  // -----------------------------------------------------------------------

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::PhysicalRegion::~PhysicalRegion()
   */
  void
  resilient_legion_physical_region_destroy(resilient_legion_physical_region_t handle);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::PhysicalRegion::PhysicalRegion
   */
  resilient_legion_physical_region_t
  resilient_legion_physical_region_copy(resilient_legion_physical_region_t handle);

  /**
   * @see Legion::PhysicalRegion::is_mapped()
   */
  bool
  resilient_legion_physical_region_is_mapped(resilient_legion_physical_region_t handle);

  /**
   * @see Legion::PhysicalRegion::wait_until_valid()
   */
  void
  resilient_legion_physical_region_wait_until_valid(resilient_legion_physical_region_t handle);

  /**
   * @see Legion::PhysicalRegion::is_valid()
   */
  bool
  resilient_legion_physical_region_is_valid(resilient_legion_physical_region_t handle);

  /**
   * @see Legion::PhysicalRegion::get_logical_region()
   */
  resilient_legion_logical_region_t
  resilient_legion_physical_region_get_logical_region(resilient_legion_physical_region_t handle);

  /**
   * @see Legion::PhysicalRegion::get_fields()
   */
  size_t
  resilient_legion_physical_region_get_field_count(resilient_legion_physical_region_t handle);
  resilient_legion_field_id_t
  resilient_legion_physical_region_get_field_id(resilient_legion_physical_region_t handle, size_t index);

  /**
   * @see Legion::PhysicalRegion::get_memories()
   */
  size_t
  resilient_legion_physical_region_get_memory_count(resilient_legion_physical_region_t handle);
  resilient_legion_memory_t
  resilient_legion_physical_region_get_memory(resilient_legion_physical_region_t handle, size_t index);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::PhysicalRegion::get_field_accessor()
   */
#define ACCESSOR_ARRAY(DIM) \
  resilient_legion_accessor_array_##DIM##d_t \
  resilient_legion_physical_region_get_field_accessor_array_##DIM##d( \
    resilient_legion_physical_region_t handle, \
    resilient_legion_field_id_t fid);
  LEGION_FOREACH_N(ACCESSOR_ARRAY)
#undef ACCESSOR_ARRAY

#define ACCESSOR_ARRAY(DIM) \
  resilient_legion_accessor_array_##DIM##d_t \
  resilient_legion_physical_region_get_field_accessor_array_##DIM##d_with_transform( \
      resilient_legion_physical_region_t handle, \
      resilient_legion_field_id_t fid, \
      resilient_legion_domain_affine_transform_t transform);
  LEGION_FOREACH_N(ACCESSOR_ARRAY)
#undef ACCESSOR_ARRAY
  
#define RAW_PTR(DIM) \
  void * \
  resilient_legion_accessor_array_##DIM##d_raw_rect_ptr(resilient_legion_accessor_array_##DIM##d_t handle, \
                                        resilient_legion_rect_##DIM##d_t rect, \
                                        resilient_legion_rect_##DIM##d_t *subrect, \
                                        resilient_legion_byte_offset_t *offsets);
  LEGION_FOREACH_N(RAW_PTR)
#undef RAW_PTR

  // Read
  void
  resilient_legion_accessor_array_1d_read(resilient_legion_accessor_array_1d_t handle,
                                resilient_legion_ptr_t ptr,
                                void *dst, size_t bytes);

#define READ_ARRAY(DIM) \
  void \
  resilient_legion_accessor_array_##DIM##d_read_point(resilient_legion_accessor_array_##DIM##d_t handle, \
                                      resilient_legion_point_##DIM##d_t point, \
                                      void *dst, size_t bytes);
  LEGION_FOREACH_N(READ_ARRAY)
#undef READ_ARRAY

  // Write
  void
  resilient_legion_accessor_array_1d_write(resilient_legion_accessor_array_1d_t handle,
                                 resilient_legion_ptr_t ptr,
                                 const void *src, size_t bytes);

#define WRITE_ARRAY(DIM) \
  void \
  resilient_legion_accessor_array_##DIM##d_write_point(resilient_legion_accessor_array_##DIM##d_t handle, \
                                       resilient_legion_point_##DIM##d_t point, \
                                       const void *src, size_t bytes);
  LEGION_FOREACH_N(WRITE_ARRAY)
#undef WRITE_ARRAY

  // Ref
  void *
  resilient_legion_accessor_array_1d_ref(resilient_legion_accessor_array_1d_t handle,
                               resilient_legion_ptr_t ptr);

#define REF_ARRAY(DIM) \
  void * \
  resilient_legion_accessor_array_##DIM##d_ref_point(resilient_legion_accessor_array_##DIM##d_t handle, \
                                     resilient_legion_point_##DIM##d_t point);
  LEGION_FOREACH_N(REF_ARRAY)
#undef REF_ARRAY

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   */
#define DESTROY_ARRAY(DIM) \
  void \
  resilient_legion_accessor_array_##DIM##d_destroy(resilient_legion_accessor_array_##DIM##d_t handle);
  LEGION_FOREACH_N(DESTROY_ARRAY)
#undef DESTROY_ARRAY

  // -----------------------------------------------------------------------
  // External Resource Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::ExternalResources::~ExternalResources()
   */
  void
  resilient_legion_external_resources_destroy(resilient_legion_external_resources_t handle);

  /**
   * @see Legion::ExternalResources::size()
   */
  size_t
  resilient_legion_external_resources_size(resilient_legion_external_resources_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::ExternalResources::operator[]()
   */
  resilient_legion_physical_region_t
  resilient_legion_external_resources_get_region(resilient_legion_external_resources_t handle,
                                       unsigned index);

  // -----------------------------------------------------------------------
  // Mappable Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Mappable::get_mappable_type
   */
  resilient_legion_mappable_type_id_t
  resilient_legion_mappable_get_type(resilient_legion_mappable_t mappable);

  /**
   * @see Legion::Mappable::as_task()
   */
  resilient_legion_task_t
  resilient_legion_mappable_as_task(resilient_legion_mappable_t mappable);

  /**
   * @see Legion::Mappable::as_copy()
   */
  resilient_legion_copy_t
  resilient_legion_mappable_as_copy(resilient_legion_mappable_t mappable);

  /**
   * @see Legion::Mappable::as_fill()
   */
  resilient_legion_fill_t
  resilient_legion_mappable_as_fill(resilient_legion_mappable_t mappable);

  /**
   * @see Legion::Mappable::as_inline_mapping()
   */
  resilient_legion_inline_t
  resilient_legion_mappable_as_inline_mapping(resilient_legion_mappable_t mappable);


  // -----------------------------------------------------------------------
  // Task Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Mappable::get_unique_id()
   */
  resilient_legion_unique_id_t
  resilient_legion_context_get_unique_id(resilient_legion_context_t ctx); 

  /**
   * Important: This creates an *empty* task. In the vast majority of
   * cases you want a pre-filled task passed by the runtime. This
   * returns a separate type, resilient_legion_task_mut_t, to help avoid
   * potential pitfalls.
   *
   * @return Caller takes ownership of return value
   *
   * @see Legion::Task::Task()
   */
  resilient_legion_task_mut_t
  resilient_legion_task_create_empty();

  /**
   * @param handle Caller must have ownership of parameter 'handle'
   *
   * @see Legion::Task::~Task()
   */
  void
  resilient_legion_task_destroy(resilient_legion_task_mut_t handle);

  /**
   * This function turns a resilient_legion_task_mut_t into a resilient_legion_task_t for
   * use with the rest of the API calls. Note that the derived pointer
   * depends on the original and should not outlive it.
   */
  resilient_legion_task_t
  resilient_legion_task_mut_as_task(resilient_legion_task_mut_t task);

  /**
   * @see Legion::Mappable::get_unique_id()
   */
  resilient_legion_unique_id_t
  resilient_legion_task_get_unique_id(resilient_legion_task_t task);

  /**
   * @see Legion::Mappable::get_depth()
   */
  int
  resilient_legion_task_get_depth(resilient_legion_task_t task);

  /**
   * @see Legion::Mappable::map_id
   */
  resilient_legion_mapper_id_t
  resilient_legion_task_get_mapper(resilient_legion_task_t task);

  /**
   * @see Legion::Mappable::tag
   */
  resilient_legion_mapping_tag_id_t
  resilient_legion_task_get_tag(resilient_legion_task_t task);

  /**
   * @see Legion::Runtime::attach_semantic_information()
   */
  void
  resilient_legion_task_id_attach_semantic_information(resilient_legion_runtime_t runtime,
                                             resilient_legion_task_id_t task_id,
                                             resilient_legion_semantic_tag_t tag,
                                             const void *buffer,
                                             size_t size,
                                             bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_semantic_information()
   */
  bool
  resilient_legion_task_id_retrieve_semantic_information(
                                           resilient_legion_runtime_t runtime,
                                           resilient_legion_task_id_t task_id,
                                           resilient_legion_semantic_tag_t tag,
                                           const void **result,
                                           size_t *size,
                                           bool can_fail /* = false */,
                                           bool wait_until_ready /* = false */);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  resilient_legion_task_id_attach_name(resilient_legion_runtime_t runtime,
                             resilient_legion_task_id_t task_id,
                             const char *name,
                             bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  resilient_legion_task_id_retrieve_name(resilient_legion_runtime_t runtime,
                               resilient_legion_task_id_t task_id,
                               const char **result);

  /**
   * @see Legion::Task::args
   */
  void *
  resilient_legion_task_get_args(resilient_legion_task_t task);

  /**
   * @see Legion::Task::args
   */
  void
  resilient_legion_task_set_args(resilient_legion_task_mut_t task, void *args);

  /**
   * @see Legion::Task::arglen
   */
  size_t
  resilient_legion_task_get_arglen(resilient_legion_task_t task);

  /**
   * @see Legion::Task::arglen
   */
  void
  resilient_legion_task_set_arglen(resilient_legion_task_mut_t task, size_t arglen);

  /**
   * @see Legion::Task::index_domain
   */
  resilient_legion_domain_t
  resilient_legion_task_get_index_domain(resilient_legion_task_t task);

  /**
   * @see Legion::Task::index_point
   */
  resilient_legion_domain_point_t
  resilient_legion_task_get_index_point(resilient_legion_task_t task);

  /**
   * @see Legion::Task::is_index_space
   */
  bool
  resilient_legion_task_get_is_index_space(resilient_legion_task_t task);

  /**
   * @see Legion::Task::local_args
   */
  void *
  resilient_legion_task_get_local_args(resilient_legion_task_t task);

  /**
   * @see Legion::Task::local_arglen
   */
  size_t
  resilient_legion_task_get_local_arglen(resilient_legion_task_t task);

  /**
   * @see Legion::Task::regions
   */
  unsigned
  resilient_legion_task_get_regions_size(resilient_legion_task_t task);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Task::regions
   */
  resilient_legion_region_requirement_t
  resilient_legion_task_get_requirement(resilient_legion_task_t task, unsigned idx);

  /**
   * @see Legion::Task::futures
   */
  unsigned
  resilient_legion_task_get_futures_size(resilient_legion_task_t task);

  /**
   * @see Legion::Task::futures
   */
  resilient_legion_future_t
  resilient_legion_task_get_future(resilient_legion_task_t task, unsigned idx);

  /**
   * @see Legion::Task::futures
   */
  void
  resilient_legion_task_add_future(resilient_legion_task_mut_t task, resilient_legion_future_t future);

  /**
   * @see Legion::Task::task_id
   */
  resilient_legion_task_id_t
  resilient_legion_task_get_task_id(resilient_legion_task_t task);

  /**
   * @see Legion::Task::target_proc
   */
  resilient_legion_processor_t
  resilient_legion_task_get_target_proc(resilient_legion_task_t task);

  /**
   * @see Legion::Task::variants::name
   */
  const char *
  resilient_legion_task_get_name(resilient_legion_task_t task);

  // -----------------------------------------------------------------------
  // Inline Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Inline::requirement
   */
  resilient_legion_region_requirement_t
  resilient_legion_inline_get_requirement(resilient_legion_inline_t inline_operation);

  // -----------------------------------------------------------------------
  // Execution Constraints
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value
   * 
   * @see Legion::ExecutionConstraintSet::ExecutionConstraintSet()
   */
  resilient_legion_execution_constraint_set_t
  resilient_legion_execution_constraint_set_create(void);

  /**
   * @param handle Caller must have ownership of parameter 'handle'
   *
   * @see Legion::ExecutionConstraintSet::~ExecutionConstraintSet()
   */
  void
  resilient_legion_execution_constraint_set_destroy(
    resilient_legion_execution_constraint_set_t handle);

  /**
   * @see Legion::ExecutionConstraintSet::add_constraint(Legion::ISAConstraint)
   */
  void
  resilient_legion_execution_constraint_set_add_isa_constraint(
    resilient_legion_execution_constraint_set_t handle,
    uint64_t prop);

  /**
   * @see Legion::ExecutionConstraintSet::add_constraint(
   *        Legion::ProcessorConstraint)
   */
  void
  resilient_legion_execution_constraint_set_add_processor_constraint(
    resilient_legion_execution_constraint_set_t handle,
    resilient_legion_processor_kind_t proc_kind);

  /**
   * @see Legion::ExecutionConstraintSet::add_constraint(
   *        Legion::ResourceConstraint)
   */
  void
  resilient_legion_execution_constraint_set_add_resource_constraint(
    resilient_legion_execution_constraint_set_t handle,
    resilient_legion_resource_constraint_t resource,
    resilient_legion_equality_kind_t eq,
    size_t value);

  /**
   * @see Legion::ExecutionConstraintSet::add_constraint(
   *        Legion::LaunchConstraint)
   */
  void
  resilient_legion_execution_constraint_set_add_launch_constraint(
    resilient_legion_execution_constraint_set_t handle,
    resilient_legion_launch_constraint_t kind,
    size_t value);

  /**
   * @see Legion::ExecutionConstraintSet::add_constraint(
   *        Legion::LaunchConstraint)
   */
  void
  resilient_legion_execution_constraint_set_add_launch_constraint_multi_dim(
    resilient_legion_execution_constraint_set_t handle,
    resilient_legion_launch_constraint_t kind,
    const size_t *values,
    int dims);

  /**
   * @see Legion::ExecutionConstraintSet::add_constraint(
   *        Legion::ColocationConstraint)
   */
  void
  resilient_legion_execution_constraint_set_add_colocation_constraint(
    resilient_legion_execution_constraint_set_t handle,
    const unsigned *indexes,
    size_t num_indexes,
    const resilient_legion_field_id_t *fields,
    size_t num_fields);

  // -----------------------------------------------------------------------
  // Layout Constraints
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::LayoutConstraintSet::LayoutConstraintSet()
   */
  resilient_legion_layout_constraint_set_t
  resilient_legion_layout_constraint_set_create(void);

  /**
   * @param handle Caller must have ownership of parameter 'handle'
   *
   * @see Legion::LayoutConstraintSet::~LayoutConstraintSet()
   */
  void
  resilient_legion_layout_constraint_set_destroy(resilient_legion_layout_constraint_set_t handle);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::register_layout()
   */
  resilient_legion_layout_constraint_id_t
  resilient_legion_layout_constraint_set_register(
    resilient_legion_runtime_t runtime,
    resilient_legion_field_space_t fspace,
    resilient_legion_layout_constraint_set_t handle,
    const char *layout_name /* = NULL */);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::preregister_layout()
   */
  resilient_legion_layout_constraint_id_t
  resilient_legion_layout_constraint_set_preregister(
    resilient_legion_layout_constraint_set_t handle,
    const char *layout_name /* = NULL */);

  /**
   * @param handle Caller must have ownership of parameter 'handle'
   *
   * @see Legion::Runtime::release_layout()
   */
  void
  resilient_legion_layout_constraint_set_release(resilient_legion_runtime_t runtime,
                                       resilient_legion_layout_constraint_id_t handle);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::SpecializedConstraint)
   */
  void
  resilient_legion_layout_constraint_set_add_specialized_constraint(
    resilient_legion_layout_constraint_set_t handle,
    resilient_legion_specialized_constraint_t specialized,
    resilient_legion_reduction_op_id_t redop);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::MemoryConstraint)
   */
  void
  resilient_legion_layout_constraint_set_add_memory_constraint(
    resilient_legion_layout_constraint_set_t handle,
    resilient_legion_memory_kind_t kind);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::FieldConstraint)
   */
  void
  resilient_legion_layout_constraint_set_add_field_constraint(
    resilient_legion_layout_constraint_set_t handle,
    const resilient_legion_field_id_t *fields,
    size_t num_fields,
    bool contiguous,
    bool inorder);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::OrderingConstraint)
   */
  void
  resilient_legion_layout_constraint_set_add_ordering_constraint(
    resilient_legion_layout_constraint_set_t handle,
    const resilient_legion_dimension_kind_t *dims,
    size_t num_dims,
    bool contiguous);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::TilingConstraint)
   */
  void
  resilient_legion_layout_constraint_set_add_tiling_constraint(
    resilient_legion_layout_constraint_set_t handle,
    resilient_legion_dimension_kind_t dim,
    size_t value, bool tiles);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::DimensionConstraint)
   */
  void
  resilient_legion_layout_constraint_set_add_dimension_constraint(
    resilient_legion_layout_constraint_set_t handle,
    resilient_legion_dimension_kind_t dim,
    resilient_legion_equality_kind_t eq,
    size_t value);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::AlignmentConstraint)
   */
  void
  resilient_legion_layout_constraint_set_add_alignment_constraint(
    resilient_legion_layout_constraint_set_t handle,
    resilient_legion_field_id_t field,
    resilient_legion_equality_kind_t eq,
    size_t byte_boundary);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::OffsetConstraint)
   */
  void
  resilient_legion_layout_constraint_set_add_offset_constraint(
    resilient_legion_layout_constraint_set_t handle,
    resilient_legion_field_id_t field,
    size_t offset);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::PointerConstraint)
   */
  void
  resilient_legion_layout_constraint_set_add_pointer_constraint(
    resilient_legion_layout_constraint_set_t handle,
    resilient_legion_memory_t memory,
    uintptr_t ptr);

  // -----------------------------------------------------------------------
  // Task Layout Constraints
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::TaskLayoutConstraintSet::TaskLayoutConstraintSet()
   */
  resilient_legion_task_layout_constraint_set_t
  resilient_legion_task_layout_constraint_set_create(void);

  /**
   * @param handle Caller must have ownership of parameter 'handle'
   *
   * @see Legion::TaskLayoutConstraintSet::~TaskLayoutConstraintSet()
   */
  void
  resilient_legion_task_layout_constraint_set_destroy(
    resilient_legion_task_layout_constraint_set_t handle);

  /**
   * @see Legion::TaskLayoutConstraintSet::add_layout_constraint()
   */
  void
  resilient_legion_task_layout_constraint_set_add_layout_constraint(
    resilient_legion_task_layout_constraint_set_t handle,
    unsigned idx,
    resilient_legion_layout_constraint_id_t layout);

  // -----------------------------------------------------------------------
  // Start-up Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Runtime::initialize()
   */
  void
  resilient_legion_runtime_initialize(int *argc,
                            char ***argv,
                            bool filter /* = false */);

  /**
   * @see Legion::Runtime::start()
   */
  int
  resilient_legion_runtime_start(int argc,
                       char **argv,
                       bool background /* = false */);

  /**
   * @see Legion::Runtime::wait_for_shutdown()
   */
  int 
  resilient_legion_runtime_wait_for_shutdown(void);

  /**
   * @see Legion::Runtime::set_return_code()
   */
  void
  resilient_legion_runtime_set_return_code(int return_code);

  /**
   * @see Legion::Runtime::set_top_level_task_id()
   */
  void
  resilient_legion_runtime_set_top_level_task_id(resilient_legion_task_id_t top_id);

  /**
   * @see Legion::Runtime::get_maximum_dimension()
   */
  size_t
  resilient_legion_runtime_get_maximum_dimension(void);

  /**
   * @see Legion::Runtime::get_input_args()
   */
  const resilient_legion_input_args_t
  resilient_legion_runtime_get_input_args(void);

  /**
   * @see Legion::Runtime::add_registration_callback()
   */
  void
  resilient_legion_runtime_add_registration_callback(
    resilient_legion_registration_callback_pointer_t callback);

  /**
   * @see Legion::Runtime::generate_library_mapper_ids()
   */
  resilient_legion_mapper_id_t
  resilient_legion_runtime_generate_library_mapper_ids(
      resilient_legion_runtime_t runtime,
      const char *library_name,
      size_t count);

  /**
   * @see Legion::Runtime::replace_default_mapper()
   */
  void
  resilient_legion_runtime_replace_default_mapper(
    resilient_legion_runtime_t runtime,
    resilient_legion_mapper_t mapper,
    resilient_legion_processor_t proc);

  /**
   * @see Legion::Runtime::generate_static_projection_id()
   */
  resilient_legion_projection_id_t
  resilient_legion_runtime_generate_static_projection_id();

  /**
   * @see Legion::Runtime::generate_library_projection_ids()
   */
  resilient_legion_projection_id_t
  resilient_legion_runtime_generate_library_projection_ids(
      resilient_legion_runtime_t runtime,
      const char *library_name,
      size_t count);

  /**
   * @see Legion::Runtime::generate_library_sharding_ids()
   */
  resilient_legion_sharding_id_t
  resilient_legion_runtime_generate_library_sharding_ids(
      resilient_legion_runtime_t runtime,
      const char *library_name,
      size_t count);

  /**
   * @see Legion::Runtime::generate_library_reduction_ids()
   */
  resilient_legion_reduction_op_id_t
  resilient_legion_runtime_generate_library_reduction_ids(
      resilient_legion_runtime_t runtime,
      const char *library_name,
      size_t count);

  /**
   * @see Legion::Runtime::preregister_projection_functor()
   */
  void
  resilient_legion_runtime_preregister_projection_functor(
    resilient_legion_projection_id_t id,
    bool exclusive,
    unsigned depth,
    resilient_legion_projection_functor_logical_region_t region_functor,
    resilient_legion_projection_functor_logical_partition_t partition_functor);

  /**
   * @see Legion::Runtime::preregister_projection_functor()
   */
  void
  resilient_legion_runtime_preregister_projection_functor_mappable(
    resilient_legion_projection_id_t id,
    bool exclusive,
    unsigned depth,
    resilient_legion_projection_functor_logical_region_mappable_t region_functor,
    resilient_legion_projection_functor_logical_partition_mappable_t partition_functor);

  /**
   * @see Legion::Runtime::register_projection_functor()
   */
  void
  resilient_legion_runtime_register_projection_functor(
    resilient_legion_runtime_t runtime,
    resilient_legion_projection_id_t id,
    bool exclusive,
    unsigned depth,
    resilient_legion_projection_functor_logical_region_t region_functor,
    resilient_legion_projection_functor_logical_partition_t partition_functor);

  /**
   * @see Legion::Runtime::register_projection_functor()
   */
  void
  resilient_legion_runtime_register_projection_functor_mappable(
    resilient_legion_runtime_t runtime,
    resilient_legion_projection_id_t id,
    bool exclusive,
    unsigned depth,
    resilient_legion_projection_functor_logical_region_mappable_t region_functor,
    resilient_legion_projection_functor_logical_partition_mappable_t partition_functor);

  /**
   * @see Legion::Runtime::generate_library_task_ids()
   */
  resilient_legion_task_id_t
  resilient_legion_runtime_generate_library_task_ids(
      resilient_legion_runtime_t runtime,
      const char *library_name,
      size_t count);

  /**
   * @see Legion::Runtime::register_task_variant()
   */
  resilient_legion_task_id_t
  resilient_legion_runtime_register_task_variant_fnptr(
    resilient_legion_runtime_t runtime,
    resilient_legion_task_id_t id /* = AUTO_GENERATE_ID */,
    const char *task_name /* = NULL*/,
    const char *variant_name /* = NULL*/,
    bool global,
    resilient_legion_execution_constraint_set_t execution_constraints,
    resilient_legion_task_layout_constraint_set_t layout_constraints,
    resilient_legion_task_config_options_t options,
    resilient_legion_task_pointer_wrapped_t wrapped_task_pointer,
    const void *userdata,
    size_t userlen);

  /**
   * @see Legion::Runtime::preregister_task_variant()
   */
  resilient_legion_task_id_t
  resilient_legion_runtime_preregister_task_variant_fnptr(
    resilient_legion_task_id_t id /* = AUTO_GENERATE_ID */,
    resilient_legion_variant_id_t variant_id /* = AUTO_GENERATE_ID */,
    const char *task_name /* = NULL*/,
    const char *variant_name /* = NULL*/,
    resilient_legion_execution_constraint_set_t execution_constraints,
    resilient_legion_task_layout_constraint_set_t layout_constraints,
    resilient_legion_task_config_options_t options,
    resilient_legion_task_pointer_wrapped_t wrapped_task_pointer,
    const void *userdata,
    size_t userlen);

#ifdef REALM_USE_LLVM
  /**
   * @see Legion::Runtime::register_task_variant()
   */
  resilient_legion_task_id_t
  resilient_legion_runtime_register_task_variant_llvmir(
    resilient_legion_runtime_t runtime,
    resilient_legion_task_id_t id /* = AUTO_GENERATE_ID */,
    const char *task_name /* = NULL*/,
    bool global,
    resilient_legion_execution_constraint_set_t execution_constraints,
    resilient_legion_task_layout_constraint_set_t layout_constraints,
    resilient_legion_task_config_options_t options,
    const char *llvmir,
    const char *entry_symbol,
    const void *userdata,
    size_t userlen);

  /**
   * @see Legion::Runtime::preregister_task_variant()
   */
  resilient_legion_task_id_t
  resilient_legion_runtime_preregister_task_variant_llvmir(
    resilient_legion_task_id_t id /* = AUTO_GENERATE_ID */,
    resilient_legion_variant_id_t variant_id /* = AUTO_GENERATE_ID */,
    const char *task_name /* = NULL*/,
    resilient_legion_execution_constraint_set_t execution_constraints,
    resilient_legion_task_layout_constraint_set_t layout_constraints,
    resilient_legion_task_config_options_t options,
    const char *llvmir,
    const char *entry_symbol,
    const void *userdata,
    size_t userlen);
#endif

#ifdef REALM_USE_PYTHON
  /**
   * @see Legion::Runtime::register_task_variant()
   */
  resilient_legion_task_id_t
  resilient_legion_runtime_register_task_variant_python_source(
    resilient_legion_runtime_t runtime,
    resilient_legion_task_id_t id /* = AUTO_GENERATE_ID */,
    const char *task_name /* = NULL*/,
    bool global,
    resilient_legion_execution_constraint_set_t execution_constraints,
    resilient_legion_task_layout_constraint_set_t layout_constraints,
    resilient_legion_task_config_options_t options,
    const char *module_name,
    const char *function_name,
    const void *userdata,
    size_t userlen);

  /**
   * @see Legion::Runtime::register_task_variant()
   */
  resilient_legion_task_id_t
  resilient_legion_runtime_register_task_variant_python_source_qualname(
    resilient_legion_runtime_t runtime,
    resilient_legion_task_id_t id /* = AUTO_GENERATE_ID */,
    const char *task_name /* = NULL*/,
    bool global,
    resilient_legion_execution_constraint_set_t execution_constraints,
    resilient_legion_task_layout_constraint_set_t layout_constraints,
    resilient_legion_task_config_options_t options,
    const char *module_name,
    const char **function_qualname,
    size_t function_qualname_len,
    const void *userdata,
    size_t userlen);
#endif

  /**
   * @see Legion::LegionTaskWrapper::resilient_legion_task_preamble()
   */
  void
  resilient_legion_task_preamble(
    const void *data,
    size_t datalen,
    realm_id_t proc_id,
    resilient_legion_task_t *taskptr,
    const resilient_legion_physical_region_t **regionptr,
    unsigned * num_regions_ptr,
    resilient_legion_context_t * ctxptr,
    resilient_legion_runtime_t * runtimeptr);

  /**
   * @see Legion::LegionTaskWrapper::resilient_legion_task_postamble()
   */
  void
  resilient_legion_task_postamble(
    resilient_legion_runtime_t runtime,
    resilient_legion_context_t ctx,
    const void *retval,
    size_t retsize);

  // -----------------------------------------------------------------------
  // Timing Operations
  // -----------------------------------------------------------------------

  /**
   * @see Realm::Clock::get_current_time_in_micros()
   */
  unsigned long long
  resilient_legion_get_current_time_in_micros(void);

  /**
   * @see Realm::Clock::get_current_time_in_nanos()
   */
  unsigned long long
  resilient_legion_get_current_time_in_nanos(void);

  /**
   * @see Legion::Runtime::get_current_time()
   */
  resilient_legion_future_t
  resilient_legion_issue_timing_op_seconds(resilient_legion_runtime_t runtime,
                                 resilient_legion_context_t ctx);

  /**
   * @see Legion::Runtime::get_current_time_in_microseconds()
   */
  resilient_legion_future_t
  resilient_legion_issue_timing_op_microseconds(resilient_legion_runtime_t runtime,
                                      resilient_legion_context_t ctx);

  /**
   * @see Legion::Runtime::get_current_time_in_nanoseconds()
   */
  resilient_legion_future_t
  resilient_legion_issue_timing_op_nanoseconds(resilient_legion_runtime_t runtime,
                                     resilient_legion_context_t ctx);

  // -----------------------------------------------------------------------
  // Machine Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Realm::Machine::get_machine()
   */
  resilient_legion_machine_t
  resilient_legion_machine_create(void);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Realm::Machine::~Machine()
   */
  void
  resilient_legion_machine_destroy(resilient_legion_machine_t handle);

  /**
   * @see Realm::Machine::get_all_processors()
   */
  void
  resilient_legion_machine_get_all_processors(
    resilient_legion_machine_t machine,
    resilient_legion_processor_t *processors,
    size_t processors_size);

  /**
   * @see Realm::Machine::get_all_processors()
   */
  size_t
  resilient_legion_machine_get_all_processors_size(resilient_legion_machine_t machine);

  /**
   * @see Realm::Machine::get_all_memories()
   */
  void
  resilient_legion_machine_get_all_memories(
    resilient_legion_machine_t machine,
    resilient_legion_memory_t *memories,
    size_t memories_size);

  /**
   * @see Realm::Machine::get_all_memories()
   */
  size_t
  resilient_legion_machine_get_all_memories_size(resilient_legion_machine_t machine);

  // -----------------------------------------------------------------------
  // Processor Operations
  // -----------------------------------------------------------------------

  /**
   * @see Realm::Processor::kind()
   */
  resilient_legion_processor_kind_t
  resilient_legion_processor_kind(resilient_legion_processor_t proc);

  /**
   * @see Realm::Processor::address_space()
   */
  resilient_legion_address_space_t
  resilient_legion_processor_address_space(resilient_legion_processor_t proc);

  // -----------------------------------------------------------------------
  // Memory Operations
  // -----------------------------------------------------------------------

  /**
   * @see Realm::Memory::kind()
   */
  resilient_legion_memory_kind_t
  resilient_legion_memory_kind(resilient_legion_memory_t mem);

  /**
   * @see Realm::Memory::address_space()
   */
  resilient_legion_address_space_t
  resilient_legion_memory_address_space(resilient_legion_memory_t mem);

  // -----------------------------------------------------------------------
  // Processor Query Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Realm::Machine::ProcessorQuery::ProcessorQuery()
   */
  resilient_legion_processor_query_t
  resilient_legion_processor_query_create(resilient_legion_machine_t machine);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Realm::Machine::ProcessorQuery::ProcessorQuery()
   */
  resilient_legion_processor_query_t
  resilient_legion_processor_query_create_copy(resilient_legion_processor_query_t query);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Realm::Machine::ProcessorQuery::~ProcessorQuery()
   */
  void
  resilient_legion_processor_query_destroy(resilient_legion_processor_query_t handle);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::ProcessorQuery::only_kind()
   */
  void
  resilient_legion_processor_query_only_kind(resilient_legion_processor_query_t query,
                                   resilient_legion_processor_kind_t kind);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::ProcessorQuery::local_address_space()
   */
  void
  resilient_legion_processor_query_local_address_space(resilient_legion_processor_query_t query);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::ProcessorQuery::same_address_space_as()
   */
  void
  resilient_legion_processor_query_same_address_space_as_processor(resilient_legion_processor_query_t query,
                                                         resilient_legion_processor_t proc);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::ProcessorQuery::same_address_space_as()
   */
  void
  resilient_legion_processor_query_same_address_space_as_memory(resilient_legion_processor_query_t query,
                                                      resilient_legion_memory_t mem);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::ProcessorQuery::has_affinity_to()
   */
  void
  resilient_legion_processor_query_has_affinity_to_memory(resilient_legion_processor_query_t query,
                                                resilient_legion_memory_t mem,
                                                unsigned min_bandwidth /* = 0 */,
                                                unsigned max_latency /* = 0 */);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::ProcessorQuery::best_affinity_to()
   */
  void
  resilient_legion_processor_query_best_affinity_to_memory(resilient_legion_processor_query_t query,
                                                 resilient_legion_memory_t mem,
                                                 int bandwidth_weight /* = 0 */,
                                                 int latency_weight /* = 0 */);

  /**
   * @see Realm::Machine::ProcessorQuery::count()
   */
  size_t
  resilient_legion_processor_query_count(resilient_legion_processor_query_t query);

  /**
   * @see Realm::Machine::ProcessorQuery::first()
   */
  resilient_legion_processor_t
  resilient_legion_processor_query_first(resilient_legion_processor_query_t query);

  /**
   * @see Realm::Machine::ProcessorQuery::next()
   */
  resilient_legion_processor_t
  resilient_legion_processor_query_next(resilient_legion_processor_query_t query,
                              resilient_legion_processor_t after);

  /**
   * @see Realm::Machine::ProcessorQuery::random()
   */
  resilient_legion_processor_t
  resilient_legion_processor_query_random(resilient_legion_processor_query_t query);

  // -----------------------------------------------------------------------
  // Memory Query Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Realm::Machine::MemoryQuery::MemoryQuery()
   */
  resilient_legion_memory_query_t
  resilient_legion_memory_query_create(resilient_legion_machine_t machine);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Realm::Machine::MemoryQuery::MemoryQuery()
   */
  resilient_legion_memory_query_t
  resilient_legion_memory_query_create_copy(resilient_legion_memory_query_t query);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Realm::Machine::MemoryQuery::~MemoryQuery()
   */
  void
  resilient_legion_memory_query_destroy(resilient_legion_memory_query_t handle);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::only_kind()
   */
  void
  resilient_legion_memory_query_only_kind(resilient_legion_memory_query_t query,
                                resilient_legion_memory_kind_t kind);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::local_address_space()
   */
  void
  resilient_legion_memory_query_local_address_space(resilient_legion_memory_query_t query);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::same_address_space_as()
   */
  void
  resilient_legion_memory_query_same_address_space_as_processor(resilient_legion_memory_query_t query,
                                                      resilient_legion_processor_t proc);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::same_address_space_as()
   */
  void
  resilient_legion_memory_query_same_address_space_as_memory(resilient_legion_memory_query_t query,
                                                   resilient_legion_memory_t mem);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::has_affinity_to()
   */
  void
  resilient_legion_memory_query_has_affinity_to_processor(resilient_legion_memory_query_t query,
                                                resilient_legion_processor_t proc,
                                                unsigned min_bandwidth /* = 0 */,
                                                unsigned max_latency /* = 0 */);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::has_affinity_to()
   */
  void
  resilient_legion_memory_query_has_affinity_to_memory(resilient_legion_memory_query_t query,
                                             resilient_legion_memory_t mem,
                                             unsigned min_bandwidth /* = 0 */,
                                             unsigned max_latency /* = 0 */);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::best_affinity_to()
   */
  void
  resilient_legion_memory_query_best_affinity_to_processor(resilient_legion_memory_query_t query,
                                                 resilient_legion_processor_t proc,
                                                 int bandwidth_weight /* = 0 */,
                                                 int latency_weight /* = 0 */);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::best_affinity_to()
   */
  void
  resilient_legion_memory_query_best_affinity_to_memory(resilient_legion_memory_query_t query,
                                              resilient_legion_memory_t mem,
                                              int bandwidth_weight /* = 0 */,
                                              int latency_weight /* = 0 */);

  /**
   * @see Realm::Machine::MemoryQuery::count()
   */
  size_t
  resilient_legion_memory_query_count(resilient_legion_memory_query_t query);

  /**
   * @see Realm::Machine::MemoryQuery::first()
   */
  resilient_legion_memory_t
  resilient_legion_memory_query_first(resilient_legion_memory_query_t query);

  /**
   * @see Realm::Machine::MemoryQuery::next()
   */
  resilient_legion_memory_t
  resilient_legion_memory_query_next(resilient_legion_memory_query_t query,
                           resilient_legion_memory_t after);

  /**
   * @see Realm::Machine::MemoryQuery::random()
   */
  resilient_legion_memory_t
  resilient_legion_memory_query_random(resilient_legion_memory_query_t query);

  // -----------------------------------------------------------------------
  // Physical Instance Operations
  // -----------------------------------------------------------------------

  /*
   * @param instance Caller must have ownership of parameter `instance`.
   *
   * @see Legion::Mapping::PhysicalInstance
   */
  void
  resilient_legion_physical_instance_destroy(resilient_legion_physical_instance_t instance);

  // -----------------------------------------------------------------------
  // Slice Task Output
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Mapping::Mapper::SliceTaskOutput:slices
   */
  void
  resilient_legion_slice_task_output_slices_add(
      resilient_legion_slice_task_output_t output,
      resilient_legion_task_slice_t slice);

  /**
   * @see Legion::Mapping::Mapper::SliceTaskOutput:verify_correctness
   */
  void
  resilient_legion_slice_task_output_verify_correctness_set(
      resilient_legion_slice_task_output_t output,
      bool verify_correctness);

  // -----------------------------------------------------------------------
  // Map Task Input/Output
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:chosen_instances
   */
  void
  resilient_legion_map_task_output_chosen_instances_clear_all(
      resilient_legion_map_task_output_t output);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:chosen_instances
   */
  void
  resilient_legion_map_task_output_chosen_instances_clear_each(
      resilient_legion_map_task_output_t output,
      size_t idx);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:chosen_instances
   */
  void
  resilient_legion_map_task_output_chosen_instances_add(
      resilient_legion_map_task_output_t output,
      resilient_legion_physical_instance_t *instances,
      size_t instances_size);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:chosen_instances
   */
  void
  resilient_legion_map_task_output_chosen_instances_set(
      resilient_legion_map_task_output_t output,
      size_t idx,
      resilient_legion_physical_instance_t *instances,
      size_t instances_size);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:target_procs
   */
  void
  resilient_legion_map_task_output_target_procs_clear(
      resilient_legion_map_task_output_t output);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:target_procs
   */
  void
  resilient_legion_map_task_output_target_procs_add(
      resilient_legion_map_task_output_t output,
      resilient_legion_processor_t proc);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:target_procs
   */
  resilient_legion_processor_t
  resilient_legion_map_task_output_target_procs_get(
      resilient_legion_map_task_output_t output,
      size_t idx);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:task_priority
   */
  void
  resilient_legion_map_task_output_task_priority_set(
      resilient_legion_map_task_output_t output,
      resilient_legion_task_priority_t priority);

  // -----------------------------------------------------------------------
  // MapperRuntime Operations
  // -----------------------------------------------------------------------

  /**
   * @param result Caller takes ownership of handle pointed by `result`.
   *
   * @see Legion::Mapping::MapperRuntime::create_physical_instance()
   */
  bool
  resilient_legion_mapper_runtime_create_physical_instance_layout_constraint(
      resilient_legion_mapper_runtime_t runtime,
      resilient_legion_mapper_context_t ctx,
      resilient_legion_memory_t target_memory,
      resilient_legion_layout_constraint_set_t constraints,
      const resilient_legion_logical_region_t *regions,
      size_t regions_size,
      resilient_legion_physical_instance_t *result,
      bool acquire,
      resilient_legion_garbage_collection_priority_t priority);

  /**
   * @param result Caller takes ownership of handle pointed by `result`.
   *
   * @see Legion::Mapping::MapperRuntime::create_physical_instance()
   */
  bool
  resilient_legion_mapper_runtime_create_physical_instance_layout_constraint_id(
      resilient_legion_mapper_runtime_t runtime,
      resilient_legion_mapper_context_t ctx,
      resilient_legion_memory_t target_memory,
      resilient_legion_layout_constraint_id_t layout_id,
      const resilient_legion_logical_region_t *regions,
      size_t regions_size,
      resilient_legion_physical_instance_t *result,
      bool acquire,
      resilient_legion_garbage_collection_priority_t priority);

  /**
   * @param result Caller takes ownership of handle pointed by `result`.
   *
   * @see Legion::Mapping::MapperRuntime::find_or_create_physical_instance()
   */
  bool
  resilient_legion_mapper_runtime_find_or_create_physical_instance_layout_constraint(
      resilient_legion_mapper_runtime_t runtime,
      resilient_legion_mapper_context_t ctx,
      resilient_legion_memory_t target_memory,
      resilient_legion_layout_constraint_set_t constraints,
      const resilient_legion_logical_region_t *regions,
      size_t regions_size,
      resilient_legion_physical_instance_t *result,
      bool *created,
      bool acquire,
      resilient_legion_garbage_collection_priority_t priority,
      bool tight_region_bounds);

  /**
   * @param result Caller takes ownership of handle pointed by `result`.
   *
   * @see Legion::Mapping::MapperRuntime::find_or_create_physical_instance()
   */
  bool
  resilient_legion_mapper_runtime_find_or_create_physical_instance_layout_constraint_id(
      resilient_legion_mapper_runtime_t runtime,
      resilient_legion_mapper_context_t ctx,
      resilient_legion_memory_t target_memory,
      resilient_legion_layout_constraint_id_t layout_id,
      const resilient_legion_logical_region_t *regions,
      size_t regions_size,
      resilient_legion_physical_instance_t *result,
      bool *created,
      bool acquire,
      resilient_legion_garbage_collection_priority_t priority,
      bool tight_region_bounds);

  /**
   * @param result Caller takes ownership of handle pointed by `result`.
   *
   * @see Legion::Mapping::MapperRuntime::find_physical_instance()
   */
  bool
  resilient_legion_mapper_runtime_find_physical_instance_layout_constraint(
      resilient_legion_mapper_runtime_t runtime,
      resilient_legion_mapper_context_t ctx,
      resilient_legion_memory_t target_memory,
      resilient_legion_layout_constraint_set_t constraints,
      const resilient_legion_logical_region_t *regions,
      size_t regions_size,
      resilient_legion_physical_instance_t *result,
      bool acquire,
      bool tight_region_bounds);

  /**
   * @param result Caller takes ownership of handle pointed by `result`.
   *
   * @see Legion::Mapping::MapperRuntime::find_physical_instance()
   */
  bool
  resilient_legion_mapper_runtime_find_physical_instance_layout_constraint_id(
      resilient_legion_mapper_runtime_t runtime,
      resilient_legion_mapper_context_t ctx,
      resilient_legion_memory_t target_memory,
      resilient_legion_layout_constraint_id_t layout_id,
      const resilient_legion_logical_region_t *regions,
      size_t regions_size,
      resilient_legion_physical_instance_t *result,
      bool acquire,
      bool tight_region_bounds);


  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Mapping::MapperRuntime::acquire_instance()
   */
  bool
  resilient_legion_mapper_runtime_acquire_instance(
      resilient_legion_mapper_runtime_t runtime,
      resilient_legion_mapper_context_t ctx,
      resilient_legion_physical_instance_t instance);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Mapping::MapperRuntime::acquire_instances()
   */
  bool
  resilient_legion_mapper_runtime_acquire_instances(
      resilient_legion_mapper_runtime_t runtime,
      resilient_legion_mapper_context_t ctx,
      resilient_legion_physical_instance_t *instances,
      size_t instances_size);

  // A hidden method here that hopefully nobody sees or ever needs
  // to use but its here anyway just in case
  resilient_legion_shard_id_t
  resilient_legion_context_get_shard_id(resilient_legion_runtime_t /*runtime*/,
                              resilient_legion_context_t /*context*/,
                              bool /*I know what I am doing*/);
  // Another hidden method for getting the number of shards
  size_t
  resilient_legion_context_get_num_shards(resilient_legion_runtime_t /*runtime*/,
                                resilient_legion_context_t /*context*/,
                                bool /*I know what I am doing*/);
  // Another hidden method for control replication that most
  // people should not be using but for which there are legitamite
  // user, especially in garbage collected languages
  // Note the caller takes ownership of the future
  resilient_legion_future_t
  resilient_legion_context_consensus_match(resilient_legion_runtime_t /*runtime*/,
                                 resilient_legion_context_t /*context*/,
                                 const void* /*input*/,
                                 void* /*output*/,
                                 size_t /*num elements*/,
                                 size_t /*element size*/);

  /**
   * used by fortran API
   */
  resilient_legion_physical_region_t
  resilient_legion_get_physical_region_by_id(
      resilient_legion_physical_region_t *regionptr, 
      int id, 
      int num_regions); 


  // -----------------------------------------------------------------------
  // Checkpointing Operations
  // -----------------------------------------------------------------------

  void
  resilient_legion_runtime_enable_checkpointing(resilient_legion_runtime_t runtime,
                                                resilient_legion_context_t ctx);

  void
  resilient_legion_runtime_checkpoint(resilient_legion_runtime_t runtime,
                                      resilient_legion_context_t ctx,
                                      resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */);

  void
  resilient_legion_runtime_auto_checkpoint(resilient_legion_runtime_t runtime,
                                           resilient_legion_context_t ctx,
                                           resilient_legion_predicate_t pred /* = resilient_legion_predicate_true() */);

#ifdef __cplusplus
}
#endif

#endif // __RESILIENT_LEGION_C_H__
