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

#ifndef RESILIENCE_TYPES_H
#define RESILIENCE_TYPES_H

#include "legion.h"

namespace ResilientLegion {

class Runtime;

using Legion::Acquire;
using Legion::AcquireLauncher;
using Legion::AddressSpace;
using Legion::AffineTransform;
using Legion::AlignmentConstraint;
using Legion::AndReduction;
using Legion::ArgumentMap;
using Legion::AttachLauncher;
using Legion::Close;
using Legion::CObjectWrapper;
using Legion::CoherenceProperty;
using Legion::ColocationConstraint;
using Legion::Color;
using Legion::ColoredPoints;
using Legion::Coloring;
using Legion::ColoringSerializer;
using Legion::Context;
using Legion::coord_t;
using Legion::Copy;
using Legion::CopyLauncher;
using Legion::CustomSerdezID;
using Legion::DeferredBuffer;
using Legion::DeferredReduction;
using Legion::DeferredValue;
using Legion::DiffReduction;
using Legion::DimensionConstraint;
using Legion::DimensionKind;
using Legion::DiscardLauncher;
using Legion::DivReduction;
using Legion::Domain;
using Legion::DomainAffineTransform;
using Legion::DomainColoring;
using Legion::DomainColoringSerializer;
using Legion::DomainPoint;
using Legion::DomainPointColoring;
using Legion::DomainScaleTransform;
using Legion::DomainT;
using Legion::DomainTransform;
using Legion::DynamicCollective;
using Legion::ExecutionConstraintSet;
using Legion::ExternalResources;
using Legion::FieldAccessor;
using Legion::FieldAllocator;
using Legion::FieldConstraint;
using Legion::FieldID;
using Legion::FieldSpace;
using Legion::FieldSpaceRequirement;
using Legion::Fill;
using Legion::FutureFunctor;
using Legion::Grant;
using Legion::IndexAllocator;
using Legion::IndexAttachLauncher;
using Legion::IndexCopyLauncher;
using Legion::IndexIterator;
using Legion::IndexPartition;
using Legion::IndexPartitionT;
using Legion::IndexSpace;
using Legion::IndexSpaceAllocator;
using Legion::IndexSpaceRequirement;
using Legion::IndexSpaceT;
using Legion::InlineLauncher;
using Legion::InlineMapping;
using Legion::InputArgs;
using Legion::ISAConstraint;
using Legion::LaunchConstraint;
using Legion::LayoutConstraintID;
using Legion::LayoutConstraintRegistrar;
using Legion::LayoutConstraintSet;
using Legion::LegionHandshake;
using Legion::Lock;
using Legion::LockRequest;
using Legion::Logger;
using Legion::LogicalPartition;
using Legion::LogicalPartitionT;
using Legion::LogicalRegion;
using Legion::LogicalRegionT;
using Legion::Machine;
using Legion::Mappable;
using Legion::MapperID;
using Legion::MappingTagID;
using Legion::MaxReduction;
using Legion::Memory;
using Legion::MemoryConstraint;
using Legion::MinReduction;
using Legion::MPILegionHandshake;
using Legion::MultiDomainColoring;
using Legion::MultiDomainPointColoring;
using Legion::MustEpoch;
using Legion::MustEpochLauncher;
using Legion::OffsetConstraint;
using Legion::OrderingConstraint;
using Legion::OrReduction;
using Legion::OutputRequirement;
using Legion::Partition;
using Legion::PartitionKind;
using Legion::PhaseBarrier;
using Legion::PhysicalRegion;
using Legion::PieceIterator;
using Legion::PieceIteratorT;
using Legion::Point;
using Legion::PointColoring;
using Legion::PointerConstraint;
using Legion::PointInDomainIterator;
using Legion::PointInRectIterator;
using Legion::PointTransformFunctor;
using Legion::Predicate;
using Legion::PredicateLauncher;
using Legion::PrivilegeMode;
using Legion::Processor;
using Legion::ProcessorConstraint;
using Legion::ProdReduction;
using Legion::ProjectionFunctor;
using Legion::ProjectionID;
using Legion::Rect;
using Legion::RectInDomainIterator;
using Legion::ReductionAccessor;
using Legion::ReductionOp;
using Legion::ReductionOpID;
using Legion::RegionFlags;
using Legion::RegionRequirement;
using Legion::RegionTreeID;
using Legion::RegistrationCallbackArgs;
using Legion::Release;
using Legion::ReleaseLauncher;
using Legion::ResourceConstraint;
using Legion::ScaleTransform;
using Legion::SemanticTag;
using Legion::SerdezFoldFnptr;
using Legion::SerdezInitFnptr;
using Legion::ShardID;
using Legion::ShardingFunctor;
using Legion::ShardingID;
using Legion::Span;
using Legion::SpanIterator;
using Legion::SpecializedConstraint;
using Legion::SplittingConstraint;
using Legion::StaticDependence;
using Legion::SumReduction;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskConfigOptions;
using Legion::TaskID;
using Legion::TaskLayoutConstraintSet;
using Legion::TaskPriority;
using Legion::TaskVariantRegistrar;
using Legion::TimingMeasurement;
using Legion::TraceID;
using Legion::Transform;
using Legion::TunableID;
using Legion::TypeTag;
using Legion::UnsafeFieldAccessor;
using Legion::Unserializable;
using Legion::UntypedBuffer;
using Legion::UntypedDeferredBuffer;
using Legion::UntypedDeferredValue;
using Legion::VariantID;
using Legion::XorReduction;

typedef void (*RegistrationCallbackFnptr)(Machine machine, Runtime *rt,
                                          const std::set<Processor> &local_procs);
typedef void (*RegistrationWithArgsCallbackFnptr)(const RegistrationCallbackArgs &args);

namespace Mapping {
using namespace Legion::Mapping;
}

}  // namespace ResilientLegion

#endif  // RESILIENCE_TYPES_H
