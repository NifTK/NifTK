/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __GeneralSegmentorCommands_h
#define __GeneralSegmentorCommands_h

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

// Operation constants, used in Undo/Redo framework
const mitk::OperationType OP_CHANGE_SLICE = 9320411;
const mitk::OperationType OP_PROPAGATE_SEEDS = 9320412;
const mitk::OperationType OP_RETAIN_MARKS = 9320413;
const mitk::OperationType OP_THRESHOLD_APPLY = 9320414;
const mitk::OperationType OP_CLEAN = 9320415;
const mitk::OperationType OP_WIPE = 9320416;
const mitk::OperationType OP_PROPAGATE = 9320417;

//-----------------------------------------------------------------------------

/**
 * \class OpGeneralSegmentorBaseCommand
 * \brief Base class for MIDAS General Segmentor commands.
 */
class OpGeneralSegmentorBaseCommand: public mitk::Operation
{
public:

  OpGeneralSegmentorBaseCommand(mitk::OperationType type, bool redo);

  ~OpGeneralSegmentorBaseCommand();

  bool IsRedo() const;

protected:
  bool m_Redo;
};


//-----------------------------------------------------------------------------
/**
 * \class OpChangeSliceCommand
 * \brief Command class for changing slice.
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
      );

  int GetBeforeSlice() const;

  int GetAfterSlice() const;

  mitk::Point3D GetBeforePoint() const;

  mitk::Point3D GetAfterPoint() const;

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
      );

  ~OpPropagateSeeds();

  int GetSliceNumber() const;

  int GetAxisNumber() const;

  mitk::PointSet::Pointer GetSeeds() const;

private:
  int m_SliceNumber;
  int m_AxisNumber;
  mitk::PointSet::Pointer m_Seeds;
};


//-----------------------------------------------------------------------------
/**
 * \class OpRetainMarks
 * \brief Command class to store data to copy one slice to the next.
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
      itk::Orientation orientation,
      std::vector<int> &region,
      ProcessorPointer processor
      );

  ~OpRetainMarks();

  int GetFromSlice() const;

  int GetToSlice() const;

  int GetAxisNumber() const;

  itk::Orientation GetOrientation() const;

  std::vector<int> GetRegion() const;

  ProcessorPointer GetProcessor() const;

private:
  int m_FromSlice;
  int m_ToSlice;
  int m_AxisNumber;
  itk::Orientation m_Orientation;
  std::vector<int> m_Region;
  ProcessorPointer m_Processor;
};


//-----------------------------------------------------------------------------
/**
 * \class OpPropagate
 * \brief Class to hold data to do propagate up/down/3D.
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
      );

  ~OpPropagate();

  std::vector<int> GetRegion() const;

  ProcessorPointer GetProcessor() const;

private:
  std::vector<int> m_Region;
  ProcessorPointer m_Processor;
};


//-----------------------------------------------------------------------------
/**
 * \class OpThresholdApply
 * \brief Class to hold data to apply the threshold region into the segmented image.
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
      );

  ~OpThresholdApply();

  bool GetThresholdFlag() const;

private:
  bool m_ThresholdFlag;
};


//-----------------------------------------------------------------------------
/**
 * \class OpClean
 * \brief Class to hold data for the MIDAS "clean" command, which filters the current contour set.
 */
class OpClean : public OpGeneralSegmentorBaseCommand
{
public:
  OpClean(
      mitk::OperationType type,
      bool redo,
      mitk::ContourModelSet::Pointer contourSet
      );

  ~OpClean();

  mitk::ContourModelSet::Pointer GetContourSet() const;

private:
  mitk::ContourModelSet::Pointer m_ContourSet;
};


//-----------------------------------------------------------------------------
/**
 * \class OpWipe
 * \brief Class to hold data to pass back to MIDASGeneralSegmentorView to Undo/Redo the Wipe commands.
 * \see MIDASGeneralSegmentorView::DoWipe
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
      );

  ~OpWipe();

  int GetSliceNumber() const;

  int GetAxisNumber() const;

  std::vector<int> GetRegion() const;

  mitk::PointSet::Pointer GetSeeds() const;

  ProcessorPointer GetProcessor() const;

private:
  int m_SliceNumber;
  int m_AxisNumber;
  std::vector<int> m_Region;
  mitk::PointSet::Pointer m_Seeds;
  ProcessorPointer m_Processor;
};

}

#endif
