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
: mitk::Operation(type)
, m_Redo(redo)
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
    int beforeSlice,
    int afterSlice,
    mitk::Point3D beforePoint,
    mitk::Point3D afterPoint
    )
: OpGeneralSegmentorBaseCommand(type, redo)
, m_BeforeSlice(beforeSlice)
, m_AfterSlice(afterSlice)
, m_BeforePoint(beforePoint)
, m_AfterPoint(afterPoint)
{
}


//-----------------------------------------------------------------------------
int OpChangeSliceCommand::GetBeforeSlice() const
{
  return m_BeforeSlice;
}


//-----------------------------------------------------------------------------
int OpChangeSliceCommand::GetAfterSlice() const
{
  return m_AfterSlice;
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
    int sliceNumber,
    int axisNumber,
    mitk::PointSet::Pointer seeds
    )
: OpGeneralSegmentorBaseCommand(type, redo)
, m_SliceNumber(sliceNumber)
, m_AxisNumber(axisNumber)
, m_Seeds(seeds)
{
}


//-----------------------------------------------------------------------------
OpPropagateSeeds::~OpPropagateSeeds()
{
}


//-----------------------------------------------------------------------------
int OpPropagateSeeds::GetSliceNumber() const
{
  return m_SliceNumber;
}


//-----------------------------------------------------------------------------
int OpPropagateSeeds::GetAxisNumber() const
{
  return m_AxisNumber;
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
    int fromSlice,
    int toSlice,
    int axisNumber,
    itk::Orientation orientation,
    std::vector<int> &region,
    ProcessorPointer processor
    )
: OpGeneralSegmentorBaseCommand(type, redo)
, m_FromSlice(fromSlice)
, m_ToSlice(toSlice)
, m_AxisNumber(axisNumber)
, m_Orientation(orientation)
, m_Region(region)
, m_Processor(processor)
{
}


//-----------------------------------------------------------------------------
OpRetainMarks::~OpRetainMarks()
{
}


//-----------------------------------------------------------------------------
int OpRetainMarks::GetFromSlice() const
{
  return m_FromSlice;
}


//-----------------------------------------------------------------------------
int OpRetainMarks::GetToSlice() const
{
  return m_ToSlice;
}


//-----------------------------------------------------------------------------
int OpRetainMarks::GetAxisNumber() const
{
  return m_AxisNumber;
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
    std::vector<int> &region,
    ProcessorPointer processor
    )
: OpGeneralSegmentorBaseCommand(type, redo)
, m_Region(region)
, m_Processor(processor)
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
    std::vector<int> &region,
    ProcessorPointer processor,
    bool thresholdFlag
    )
: OpPropagate(type, redo, region, processor)
, m_ThresholdFlag(thresholdFlag)
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
: OpGeneralSegmentorBaseCommand(type, redo)
, m_ContourSet(contourSet)
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
    int sliceNumber,
    int axisNumber,
    std::vector<int> &region,
    mitk::PointSet::Pointer seeds,
    ProcessorPointer processor
    )
: OpGeneralSegmentorBaseCommand(type, redo)
, m_SliceNumber(sliceNumber)
, m_AxisNumber(axisNumber)
, m_Region(region)
, m_Seeds(seeds)
, m_Processor(processor)
{
}


//-----------------------------------------------------------------------------
OpWipe::~OpWipe()
{
}


//-----------------------------------------------------------------------------
int OpWipe::GetSliceNumber() const
{
  return m_SliceNumber;
}


//-----------------------------------------------------------------------------
int OpWipe::GetAxisNumber() const
{
  return m_AxisNumber;
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
