/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKMIDASGENERALSEGMENTORCOMMANDS_H
#define MITKMIDASGENERALSEGMENTORCOMMANDS_H

#include "mitkOperation.h"
#include "mitkDataNode.h"
#include "mitkTool.h"
#include "mitkPointSet.h"
#include "mitkContourSet.h"
#include "itkMIDASHelper.h"
#include "itkMIDASImageUpdateClearRegionProcessor.h"
#include "itkMIDASImageUpdatePasteRegionProcessor.h"
#include "itkMIDASRetainMarksNoThresholdingProcessor.h"

namespace mitk
{

//-----------------------------------------------------------------------------

/**
 * \class OpGeneralSegmentorBaseCommand
 * \brief Base class for MIDAS GeneralSegmentorView commands.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpGeneralSegmentorBaseCommand: public mitk::Operation
{
public:

  OpGeneralSegmentorBaseCommand(
      mitk::OperationType type,
      bool redo
      )
  : mitk::Operation(type)
  , m_Redo(redo)
  { };

  ~OpGeneralSegmentorBaseCommand()
  { };
  bool IsRedo() const { return m_Redo; }

protected:
  bool m_Redo;
};


//-----------------------------------------------------------------------------

/**
 * \class OpGeneralSegmentorBaseCommand
 * \brief Command class for changing slice.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpChangeSliceCommand : public OpGeneralSegmentorBaseCommand
{
public:
  OpChangeSliceCommand(
      mitk::OperationType type,
      bool redo,
      int beforeSlice,
      int afterSlice,
      mitk::Point3D beforePoint,
      mitk::Point3D afterPoint
      )
  : mitk::OpGeneralSegmentorBaseCommand(type, redo)
  , m_BeforeSlice(beforeSlice)
  , m_AfterSlice(afterSlice)
  , m_BeforePoint(beforePoint)
  , m_AfterPoint(afterPoint)
  { };
  int GetBeforeSlice() const { return m_BeforeSlice; }
  int GetAfterSlice() const { return m_AfterSlice; }
  mitk::Point3D GetBeforePoint() const { return m_BeforePoint; }
  mitk::Point3D GetAfterPoint() const { return m_AfterPoint; }
protected:
  int m_BeforeSlice;
  int m_AfterSlice;
  mitk::Point3D m_BeforePoint;
  mitk::Point3D m_AfterPoint;
};


//-----------------------------------------------------------------------------

/**
 * \class OpPropagateSeeds
 * \brief Command class to store data for propagating seeds from one slice to the next.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpPropagateSeeds: public OpGeneralSegmentorBaseCommand
{
public:

  OpPropagateSeeds(
      mitk::OperationType type,
      bool redo,
      int sliceNumber,
      int axisNumber,
      mitk::PointSet::Pointer seeds
      )
  : mitk::OpGeneralSegmentorBaseCommand(type, redo)
  , m_SliceNumber(sliceNumber)
  , m_AxisNumber(axisNumber)
  , m_Seeds(seeds)
  {
  };
  ~OpPropagateSeeds()
  { };
  int GetSliceNumber() const { return m_SliceNumber; }
  int GetAxisNumber() const { return m_AxisNumber; }
  mitk::PointSet::Pointer GetSeeds() const { return m_Seeds; }
private:
  int m_SliceNumber;
  int m_AxisNumber;
  mitk::PointSet::Pointer m_Seeds;
};


//-----------------------------------------------------------------------------

/**
 * \class OpRetainMarks
 * \brief Command class to store data to copy one slice to the next.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpRetainMarks: public OpGeneralSegmentorBaseCommand
{
public:
  typedef itk::MIDASRetainMarksNoThresholdingProcessor<mitk::Tool::DefaultSegmentationDataType, 3> ProcessorType;
  typedef ProcessorType::Pointer ProcessorPointer;

  OpRetainMarks(
      mitk::OperationType type,
      bool redo,
      int fromSlice,
      int toSlice,
      int axisNumber,
      itk::ORIENTATION_ENUM orientation,
      std::vector<int> &region,
      ProcessorPointer processor
      )
  : mitk::OpGeneralSegmentorBaseCommand(type, redo)
  , m_FromSlice(fromSlice)
  , m_ToSlice(toSlice)
  , m_AxisNumber(axisNumber)
  , m_Orientation(orientation)
  , m_Region(region)
  , m_Processor(processor)
  {
  };
  ~OpRetainMarks()
  { };
  int GetFromSlice() const { return m_FromSlice; }
  int GetToSlice() const { return m_ToSlice; }
  int GetAxisNumber() const { return m_AxisNumber; }
  itk::ORIENTATION_ENUM GetOrientation() const { return m_Orientation; }
  std::vector<int> GetRegion() const { return m_Region; }
  ProcessorPointer GetProcessor() const { return m_Processor; }

private:
  int m_FromSlice;
  int m_ToSlice;
  int m_AxisNumber;
  itk::ORIENTATION_ENUM m_Orientation;
  std::vector<int> m_Region;
  ProcessorPointer m_Processor;
};


//-----------------------------------------------------------------------------

/**
 * \class OpPropagate
 * \brief Class to hold data to do propagate up/down/3D.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpPropagate: public OpGeneralSegmentorBaseCommand
{
public:

  typedef itk::MIDASImageUpdatePasteRegionProcessor<mitk::Tool::DefaultSegmentationDataType, 3> ProcessorType;
  typedef ProcessorType::Pointer ProcessorPointer;

  OpPropagate(
      mitk::OperationType type,
      bool redo,
      std::vector<int> &region,
      ProcessorPointer processor
      )
  : mitk::OpGeneralSegmentorBaseCommand(type, redo)
  , m_Region(region)
  , m_Processor(processor)
  { };

  ~OpPropagate()
  { };
  std::vector<int> GetRegion() const { return m_Region; }
  ProcessorPointer GetProcessor() const { return m_Processor; }
private:
  std::vector<int> m_Region;
  ProcessorPointer m_Processor;
};


//-----------------------------------------------------------------------------

/**
 * \class OpApplyThreshold
 * \brief Class to hold data to apply the threshold region into the segmented image.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpThresholdApply: public OpPropagate
{
public:

  typedef itk::MIDASImageUpdatePasteRegionProcessor<mitk::Tool::DefaultSegmentationDataType, 3> ProcessorType;
  typedef ProcessorType::Pointer ProcessorPointer;

  OpThresholdApply(
      mitk::OperationType type,
      bool redo,
      std::vector<int> &region,
      ProcessorPointer processor,
      bool thresholdFlag
      )
  : mitk::OpPropagate(type, redo, region, processor)
  , m_ThresholdFlag(thresholdFlag)
  { };

  ~OpThresholdApply()
  { };
  bool GetThresholdFlag() const { return m_ThresholdFlag; }
private:
  bool m_ThresholdFlag;
};


//-----------------------------------------------------------------------------

/**
 * \class OpClean
 * \brief Class to hold data for the MIDAS "clean" command, which filters the current contour set.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpClean : public OpGeneralSegmentorBaseCommand
{
public:
  OpClean(
      mitk::OperationType type,
      bool redo,
      mitk::ContourSet::Pointer contourSet
      )
  : mitk::OpGeneralSegmentorBaseCommand(type, redo)
  , m_ContourSet(contourSet)
  {
  };
  ~OpClean() {};
  mitk::ContourSet::Pointer GetContourSet() const { return m_ContourSet; }
private:
  mitk::ContourSet::Pointer m_ContourSet;
};


//-----------------------------------------------------------------------------

/**
 * \class OpWipe
 * \brief Class to hold data to pass back to MIDASGeneralSegmentorView to Undo/Redo the Wipe commands.
 * \see MIDASGeneralSegmentorView::DoWipe
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpWipe: public OpGeneralSegmentorBaseCommand
{
public:
  typedef itk::MIDASImageUpdateClearRegionProcessor<mitk::Tool::DefaultSegmentationDataType, 3> ProcessorType;
  typedef ProcessorType::Pointer ProcessorPointer;

  OpWipe(
      mitk::OperationType type,
      bool redo,
      int sliceNumber,
      int axisNumber,
      std::vector<int> &region,
      mitk::PointSet::Pointer seeds,
      ProcessorPointer processor
      )
  : mitk::OpGeneralSegmentorBaseCommand(type, redo)
  , m_SliceNumber(sliceNumber)
  , m_AxisNumber(axisNumber)
  , m_Region(region)
  , m_Seeds(seeds)
  , m_Processor(processor)
  {
  };
  ~OpWipe()
  { };
  int GetSliceNumber() const { return m_SliceNumber; }
  int GetAxisNumber() const { return m_AxisNumber; }
  std::vector<int> GetRegion() const { return m_Region; }
  mitk::PointSet::Pointer GetSeeds() const { return m_Seeds; }
  ProcessorPointer GetProcessor() const { return m_Processor; }

private:
  int m_SliceNumber;
  int m_AxisNumber;
  std::vector<int> m_Region;
  mitk::PointSet::Pointer m_Seeds;
  ProcessorPointer m_Processor;
};

} // end namespace

#endif
