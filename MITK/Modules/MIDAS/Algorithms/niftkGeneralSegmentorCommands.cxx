/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorCommands.h"

#include <mitkContourModelSet.h>
#include <mitkDataNode.h>
#include <mitkOperation.h>
#include <mitkPointSet.h>
#include <mitkTool.h>

#include <itkMIDASImageUpdateClearRegionProcessor.h>
#include <itkMIDASImageUpdatePasteRegionProcessor.h>
#include <itkMIDASRetainMarksNoThresholdingProcessor.h>

namespace niftk
{

//-----------------------------------------------------------------------------
OpGeneralSegmentorBaseCommand::OpGeneralSegmentorBaseCommand(mitk::OperationType type, bool redo)
: mitk::Operation(type),
  m_Redo(redo)
{
}


//-----------------------------------------------------------------------------
OpGeneralSegmentorBaseCommand::~OpGeneralSegmentorBaseCommand()
{
}


//-----------------------------------------------------------------------------
bool OpGeneralSegmentorBaseCommand::IsRedo() const
{
  return m_Redo;
}


//-----------------------------------------------------------------------------
OpChangeSliceCommand::OpChangeSliceCommand(
    mitk::OperationType type,
    bool redo,
    mitk::Point3D beforePoint,
    mitk::Point3D afterPoint
    )
: OpGeneralSegmentorBaseCommand(type, redo),
  m_BeforePoint(beforePoint),
  m_AfterPoint(afterPoint)
{
}


//-----------------------------------------------------------------------------
mitk::Point3D OpChangeSliceCommand::GetBeforePoint() const
{
  return m_BeforePoint;
}


//-----------------------------------------------------------------------------
mitk::Point3D OpChangeSliceCommand::GetAfterPoint() const
{
  return m_AfterPoint;
}


//-----------------------------------------------------------------------------
OpPropagateSeeds::OpPropagateSeeds(
    mitk::OperationType type,
    bool redo,
    int sliceAxis,
    int sliceIndex,
    mitk::PointSet::Pointer seeds
    )
: OpGeneralSegmentorBaseCommand(type, redo),
  m_SliceAxis(sliceAxis),
  m_SliceIndex(sliceIndex),
  m_Seeds(seeds)
{
}


//-----------------------------------------------------------------------------
OpPropagateSeeds::~OpPropagateSeeds()
{
}


//-----------------------------------------------------------------------------
int OpPropagateSeeds::GetSliceAxis() const
{
  return m_SliceAxis;
}


//-----------------------------------------------------------------------------
int OpPropagateSeeds::GetSliceIndex() const
{
  return m_SliceIndex;
}


//-----------------------------------------------------------------------------
mitk::PointSet::Pointer OpPropagateSeeds::GetSeeds() const
{
  return m_Seeds;
}


//-----------------------------------------------------------------------------
OpRetainMarks::OpRetainMarks(
    mitk::OperationType type,
    bool redo,
    int sliceAxis,
    int fromSliceIndex,
    int toSliceIndex,
    itk::Orientation orientation,
    const std::vector<int>& region,
    ProcessorPointer processor
    )
: OpGeneralSegmentorBaseCommand(type, redo),
  m_SliceAxis(sliceAxis),
  m_FromSliceIndex(fromSliceIndex),
  m_ToSliceIndex(toSliceIndex),
  m_Orientation(orientation),
  m_Region(region),
  m_Processor(processor)
{
}


//-----------------------------------------------------------------------------
OpRetainMarks::~OpRetainMarks()
{
}


//-----------------------------------------------------------------------------
int OpRetainMarks::GetSliceAxis() const
{
  return m_SliceAxis;
}


//-----------------------------------------------------------------------------
int OpRetainMarks::GetFromSliceIndex() const
{
  return m_FromSliceIndex;
}


//-----------------------------------------------------------------------------
int OpRetainMarks::GetToSliceIndex() const
{
  return m_ToSliceIndex;
}


//-----------------------------------------------------------------------------
itk::Orientation OpRetainMarks::GetOrientation() const
{
  return m_Orientation;
}


//-----------------------------------------------------------------------------
std::vector<int> OpRetainMarks::GetRegion() const
{
  return m_Region;
}


//-----------------------------------------------------------------------------
OpRetainMarks::ProcessorPointer OpRetainMarks::GetProcessor() const
{
  return m_Processor;
}


//-----------------------------------------------------------------------------
OpPropagate::OpPropagate(
    mitk::OperationType type,
    bool redo,
    const std::vector<int>& region,
    ProcessorPointer processor
    )
: OpGeneralSegmentorBaseCommand(type, redo),
  m_Region(region),
  m_Processor(processor)
{
}


//-----------------------------------------------------------------------------
OpPropagate::~OpPropagate()
{
}


//-----------------------------------------------------------------------------
std::vector<int> OpPropagate::GetRegion() const
{
  return m_Region;
}


//-----------------------------------------------------------------------------
OpPropagate::ProcessorPointer OpPropagate::GetProcessor() const
{
  return m_Processor;
}


//-----------------------------------------------------------------------------
OpThresholdApply::OpThresholdApply(
    mitk::OperationType type,
    bool redo,
    const std::vector<int>& region,
    ProcessorPointer processor,
    bool thresholdFlag
    )
: OpPropagate(type, redo, region, processor),
  m_ThresholdFlag(thresholdFlag)
{
}


//-----------------------------------------------------------------------------
OpThresholdApply::~OpThresholdApply()
{
}


//-----------------------------------------------------------------------------
bool OpThresholdApply::GetThresholdFlag() const
{
  return m_ThresholdFlag;
}


//-----------------------------------------------------------------------------
OpClean::OpClean(
    mitk::OperationType type,
    bool redo,
    mitk::ContourModelSet::Pointer contourSet
    )
: OpGeneralSegmentorBaseCommand(type, redo),
  m_ContourSet(contourSet)
{
}


//-----------------------------------------------------------------------------
OpClean::~OpClean()
{
}


//-----------------------------------------------------------------------------
mitk::ContourModelSet::Pointer OpClean::GetContourSet() const
{
  return m_ContourSet;
}


//-----------------------------------------------------------------------------
OpWipe::OpWipe(
    mitk::OperationType type,
    bool redo,
    int sliceAxis,
    int sliceIndex,
    const std::vector<int>& region,
    mitk::PointSet::Pointer seeds,
    ProcessorPointer processor
    )
: OpGeneralSegmentorBaseCommand(type, redo),
  m_SliceAxis(sliceAxis),
  m_SliceIndex(sliceIndex),
  m_Region(region),
  m_Seeds(seeds),
  m_Processor(processor)
{
}


//-----------------------------------------------------------------------------
OpWipe::~OpWipe()
{
}


//-----------------------------------------------------------------------------
int OpWipe::GetSliceAxis() const
{
  return m_SliceAxis;
}


//-----------------------------------------------------------------------------
int OpWipe::GetSliceIndex() const
{
  return m_SliceIndex;
}


//-----------------------------------------------------------------------------
std::vector<int> OpWipe::GetRegion() const
{
  return m_Region;
}


//-----------------------------------------------------------------------------
mitk::PointSet::Pointer OpWipe::GetSeeds() const
{
  return m_Seeds;
}


//-----------------------------------------------------------------------------
OpWipe::ProcessorPointer OpWipe::GetProcessor() const
{
  return m_Processor;
}

}
