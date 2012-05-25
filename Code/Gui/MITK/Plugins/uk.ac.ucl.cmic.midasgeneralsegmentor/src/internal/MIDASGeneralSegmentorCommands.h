/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKMIDASGENERALSEGMENTORCOMMANDS_H
#define MITKMIDASGENERALSEGMENTORCOMMANDS_H

#include "mitkOperation.h"
#include "mitkDataNode.h"
#include "mitkTool.h"
#include "mitkPointSet.h"
#include "mitkContourSet.h"
#include "itkMIDASHelper.h"
#include "itkImageUpdateClearRegionProcessor.h"
#include "itkImageUpdateCopyRegionProcessor.h"
#include "itkImageUpdatePasteRegionProcessor.h"
#include "itkMIDASRetainMarksNoThresholdingProcessor.h"

namespace mitk
{

/**
 * \class OpGeneralSegmentorBaseCommand
 * \brief Base class for common stuff for MIDAS GeneralSegmentorView commands.
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

/**
 * \class OpPropagate
 * \brief Class to hold data to pass back to MIDASGeneralSegmentorView to Undo/Redo the Propagate commands.
 * \see MIDASGeneralSegmentorView::DoPropagate
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpPropagate: public OpGeneralSegmentorBaseCommand
{
public:
  typedef itk::ImageUpdatePasteRegionProcessor<mitk::Tool::DefaultSegmentationDataType, 3> ProcessorType;
  typedef ProcessorType::Pointer ProcessorPointer;

  OpPropagate(
      mitk::OperationType type,
      bool redo,
      int sliceNumber,
      int axisNumber,
      std::vector<int> &region,
      mitk::PointSet::Pointer seeds,
      ProcessorPointer processor,
      bool copyBackground
      )
  : mitk::OpGeneralSegmentorBaseCommand(type, redo)
  , m_SliceNumber(sliceNumber)
  , m_AxisNumber(axisNumber)
  , m_Region(region)
  , m_Seeds(seeds)
  , m_Processor(processor)
  , m_CopyBackground(copyBackground)
  {
  };
  ~OpPropagate()
  { };
  int GetSliceNumber() const { return m_SliceNumber; }
  int GetAxisNumber() const { return m_AxisNumber; }
  std::vector<int> GetRegion() const { return m_Region; }
  mitk::PointSet::Pointer GetSeeds() const { return m_Seeds; }
  ProcessorPointer GetProcessor() const { return m_Processor; }
  bool GetCopyBackground() const { return m_CopyBackground; }
private:
  int m_SliceNumber;
  int m_AxisNumber;
  std::vector<int> m_Region;
  mitk::PointSet::Pointer m_Seeds;
  ProcessorPointer m_Processor;
  bool m_CopyBackground;
};

/**
 * \class OpApplyThreshold
 * \brief Class to hold data to pass back to the MIDASGeneralSegmentorView to Undo/Redo the ThresholdApply function.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpThresholdApply: public OpPropagate
{
public:

  OpThresholdApply(
      mitk::OperationType type,
      bool redo,
      int sliceNumber,
      int axisNumber,
      std::vector<int> &region,
      mitk::PointSet::Pointer seeds,
      ProcessorPointer processor,
      bool copyBackground,
      bool thresholdFlag,
      int newSliceNumber
      )
  : mitk::OpPropagate(type, redo, sliceNumber, axisNumber, region, seeds, processor, copyBackground)
  , m_ThresholdFlag(thresholdFlag)
  , m_NewSliceNumber(newSliceNumber)
  { };

  ~OpThresholdApply()
  { };
  bool GetThresholdFlag() const { return m_ThresholdFlag; }
  int GetNewSliceNumber() const { return m_NewSliceNumber; }
private:
  bool m_ThresholdFlag;
  int m_NewSliceNumber;
};

/**
 * \class OpWipe
 * \brief Class to hold data to pass back to MIDASGeneralSegmentorView to Undo/Redo the Wipe commands.
 * \see MIDASGeneralSegmentorView::DoWipe
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpWipe: public OpGeneralSegmentorBaseCommand
{
public:
  typedef itk::ImageUpdateClearRegionProcessor<mitk::Tool::DefaultSegmentationDataType, 3> ProcessorType;
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

/**
 * \class OpPropagateSeeds
 * \brief Class to hold data to pass back to MIDASGeneralSegmentorView to Undo/Redo the propagation of seeds that occurs when slice changes.
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

/**
 * \class OpRetainMarks
 * \brief Class to hold data to pass back to MIDASGeneralSegmentorView to Undo/Redo the Retain Marks commands.
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
      mitk::PointSet::Pointer seeds,
      ProcessorPointer processor
      )
  : mitk::OpGeneralSegmentorBaseCommand(type, redo)
  , m_FromSlice(fromSlice)
  , m_ToSlice(toSlice)
  , m_AxisNumber(axisNumber)
  , m_Orientation(orientation)
  , m_Region(region)
  , m_Seeds(seeds)
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
  mitk::PointSet::Pointer GetSeeds() const { return m_Seeds; }
  ProcessorPointer GetProcessor() const { return m_Processor; }

private:
  int m_FromSlice;
  int m_ToSlice;
  int m_AxisNumber;
  itk::ORIENTATION_ENUM m_Orientation;
  std::vector<int> m_Region;
  mitk::PointSet::Pointer m_Seeds;
  ProcessorPointer m_Processor;
};

} // end namespace

#endif
