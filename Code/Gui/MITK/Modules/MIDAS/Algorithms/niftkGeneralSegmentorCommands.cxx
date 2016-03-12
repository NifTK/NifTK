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


//-----------------------------------------------------------------------------
niftk::OpGeneralSegmentorBaseCommand::OpGeneralSegmentorBaseCommand(mitk::OperationType type, bool redo)
: mitk::Operation(type)
, m_Redo(redo)
{
}


//-----------------------------------------------------------------------------
niftk::OpGeneralSegmentorBaseCommand::~OpGeneralSegmentorBaseCommand()
{
}


//-----------------------------------------------------------------------------
bool niftk::OpGeneralSegmentorBaseCommand::IsRedo() const
{
  return m_Redo;
}


//-----------------------------------------------------------------------------
niftk::OpChangeSliceCommand::OpChangeSliceCommand(
    mitk::OperationType type,
    bool redo,
    int beforeSlice,
    int afterSlice,
    mitk::Point3D beforePoint,
    mitk::Point3D afterPoint
    )
: niftk::OpGeneralSegmentorBaseCommand(type, redo)
, m_BeforeSlice(beforeSlice)
, m_AfterSlice(afterSlice)
, m_BeforePoint(beforePoint)
, m_AfterPoint(afterPoint)
{
}


//-----------------------------------------------------------------------------
int niftk::OpChangeSliceCommand::GetBeforeSlice() const
{
  return m_BeforeSlice;
}


//-----------------------------------------------------------------------------
int niftk::OpChangeSliceCommand::GetAfterSlice() const
{
  return m_AfterSlice;
}


//-----------------------------------------------------------------------------
mitk::Point3D niftk::OpChangeSliceCommand::GetBeforePoint() const
{
  return m_BeforePoint;
}


//-----------------------------------------------------------------------------
mitk::Point3D niftk::OpChangeSliceCommand::GetAfterPoint() const
{
  return m_AfterPoint;
}


//-----------------------------------------------------------------------------
niftk::OpPropagateSeeds::OpPropagateSeeds(
    mitk::OperationType type,
    bool redo,
    int sliceNumber,
    int axisNumber,
    mitk::PointSet::Pointer seeds
    )
: niftk::OpGeneralSegmentorBaseCommand(type, redo)
, m_SliceNumber(sliceNumber)
, m_AxisNumber(axisNumber)
, m_Seeds(seeds)
{
}


//-----------------------------------------------------------------------------
niftk::OpPropagateSeeds::~OpPropagateSeeds()
{
}


//-----------------------------------------------------------------------------
int niftk::OpPropagateSeeds::GetSliceNumber() const
{
  return m_SliceNumber;
}


//-----------------------------------------------------------------------------
int niftk::OpPropagateSeeds::GetAxisNumber() const
{
  return m_AxisNumber;
}


//-----------------------------------------------------------------------------
mitk::PointSet::Pointer niftk::OpPropagateSeeds::GetSeeds() const
{
  return m_Seeds;
}


//-----------------------------------------------------------------------------
niftk::OpRetainMarks::OpRetainMarks(
    mitk::OperationType type,
    bool redo,
    int fromSlice,
    int toSlice,
    int axisNumber,
    itk::Orientation orientation,
    std::vector<int> &region,
    ProcessorPointer processor
    )
: niftk::OpGeneralSegmentorBaseCommand(type, redo)
, m_FromSlice(fromSlice)
, m_ToSlice(toSlice)
, m_AxisNumber(axisNumber)
, m_Orientation(orientation)
, m_Region(region)
, m_Processor(processor)
{
}


//-----------------------------------------------------------------------------
niftk::OpRetainMarks::~OpRetainMarks()
{
}


//-----------------------------------------------------------------------------
int niftk::OpRetainMarks::GetFromSlice() const
{
  return m_FromSlice;
}


//-----------------------------------------------------------------------------
int niftk::OpRetainMarks::GetToSlice() const
{
  return m_ToSlice;
}


//-----------------------------------------------------------------------------
int niftk::OpRetainMarks::GetAxisNumber() const
{
  return m_AxisNumber;
}


//-----------------------------------------------------------------------------
itk::Orientation niftk::OpRetainMarks::GetOrientation() const
{
  return m_Orientation;
}


//-----------------------------------------------------------------------------
std::vector<int> niftk::OpRetainMarks::GetRegion() const
{
  return m_Region;
}


//-----------------------------------------------------------------------------
niftk::OpRetainMarks::ProcessorPointer niftk::OpRetainMarks::GetProcessor() const
{
  return m_Processor;
}


//-----------------------------------------------------------------------------
niftk::OpPropagate::OpPropagate(
    mitk::OperationType type,
    bool redo,
    std::vector<int> &region,
    ProcessorPointer processor
    )
: niftk::OpGeneralSegmentorBaseCommand(type, redo)
, m_Region(region)
, m_Processor(processor)
{
}


//-----------------------------------------------------------------------------
niftk::OpPropagate::~OpPropagate()
{
}


//-----------------------------------------------------------------------------
std::vector<int> niftk::OpPropagate::GetRegion() const
{
  return m_Region;
}


//-----------------------------------------------------------------------------
niftk::OpPropagate::ProcessorPointer niftk::OpPropagate::GetProcessor() const
{
  return m_Processor;
}


//-----------------------------------------------------------------------------
niftk::OpThresholdApply::OpThresholdApply(
    mitk::OperationType type,
    bool redo,
    std::vector<int> &region,
    ProcessorPointer processor,
    bool thresholdFlag
    )
: niftk::OpPropagate(type, redo, region, processor)
, m_ThresholdFlag(thresholdFlag)
{
}


//-----------------------------------------------------------------------------
niftk::OpThresholdApply::~OpThresholdApply()
{
}


//-----------------------------------------------------------------------------
bool niftk::OpThresholdApply::GetThresholdFlag() const
{
  return m_ThresholdFlag;
}


//-----------------------------------------------------------------------------
niftk::OpClean::OpClean(
    mitk::OperationType type,
    bool redo,
    mitk::ContourModelSet::Pointer contourSet
    )
: niftk::OpGeneralSegmentorBaseCommand(type, redo)
, m_ContourSet(contourSet)
{
}


//-----------------------------------------------------------------------------
niftk::OpClean::~OpClean()
{
}


//-----------------------------------------------------------------------------
mitk::ContourModelSet::Pointer niftk::OpClean::GetContourSet() const
{
  return m_ContourSet;
}


//-----------------------------------------------------------------------------
niftk::OpWipe::OpWipe(
    mitk::OperationType type,
    bool redo,
    int sliceNumber,
    int axisNumber,
    std::vector<int> &region,
    mitk::PointSet::Pointer seeds,
    ProcessorPointer processor
    )
: niftk::OpGeneralSegmentorBaseCommand(type, redo)
, m_SliceNumber(sliceNumber)
, m_AxisNumber(axisNumber)
, m_Region(region)
, m_Seeds(seeds)
, m_Processor(processor)
{
}


//-----------------------------------------------------------------------------
niftk::OpWipe::~OpWipe()
{
}


//-----------------------------------------------------------------------------
int niftk::OpWipe::GetSliceNumber() const
{
  return m_SliceNumber;
}


//-----------------------------------------------------------------------------
int niftk::OpWipe::GetAxisNumber() const
{
  return m_AxisNumber;
}


//-----------------------------------------------------------------------------
std::vector<int> niftk::OpWipe::GetRegion() const
{
  return m_Region;
}


//-----------------------------------------------------------------------------
mitk::PointSet::Pointer niftk::OpWipe::GetSeeds() const
{
  return m_Seeds;
}


//-----------------------------------------------------------------------------
niftk::OpWipe::ProcessorPointer niftk::OpWipe::GetProcessor() const
{
  return m_Processor;
}
