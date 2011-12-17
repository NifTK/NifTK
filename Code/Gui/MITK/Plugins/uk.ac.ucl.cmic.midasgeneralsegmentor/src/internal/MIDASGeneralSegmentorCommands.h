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
#include "mitkMIDASContourTool.h"
#include "itkMIDASHelper.h"
#include "itkMIDASThresholdApplyProcessor.h"
#include "itkMIDASPropagateProcessor.h"
#include "itkMIDASPropagateUpProcessor.h"
#include "itkMIDASPropagateDownProcessor.h"
#include "itkMIDASWipeSliceProcessor.h"
#include "itkMIDASWipePlusProcessor.h"
#include "itkMIDASWipeMinusProcessor.h"
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
      bool redo,
      mitk::DataNode* targetNode
      )
  : mitk::Operation(type)
  , m_Redo(redo)
  , m_TargetNode(targetNode)
  { };

  ~OpGeneralSegmentorBaseCommand()
  { };
  bool IsRedo() const { return m_Redo; }
  mitk::DataNode* GetTargetNode() const { return m_TargetNode; }

protected:
  bool m_Redo;
  mitk::DataNode* m_TargetNode;

};

/**
 * \class OpApplyThreshold
 * \brief Class to hold data to pass back to the MIDASGeneralSegmentorView to Undo/Redo the ThresholdApply function.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpThresholdApply: public OpGeneralSegmentorBaseCommand
{
public:
  typedef itk::MIDASThresholdApplyProcessor<mitk::Tool::DefaultSegmentationDataType, 3> ProcessorType;

  OpThresholdApply(
      mitk::OperationType type,
      bool redo,
      mitk::DataNode* targetNode,
      mitk::DataNode* sourceNode,
      ProcessorType* processor
      )
  : mitk::OpGeneralSegmentorBaseCommand(type, redo, targetNode)
  , m_SourceNode(sourceNode)
  , m_Processor(processor)
  { };

  ~OpThresholdApply()
  { };
  ProcessorType::Pointer GetProcessor() const { return m_Processor; }
  mitk::DataNode* GetSourceNode() const { return m_SourceNode; }

private:
  mitk::DataNode* m_SourceNode;
  ProcessorType::Pointer m_Processor;
};

/**
 * \class OpPropagate
 * \brief Base class to hold data to pass back to the MIDASGeneralSegmentorView to Undo/Redo the PropagateUp and PropagateDown function.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpPropagate: public OpGeneralSegmentorBaseCommand
{
public:
  typedef itk::MIDASRegionProcessor<mitk::Tool::DefaultSegmentationDataType, 3> RegionProcessorType;
  OpPropagate(
      mitk::OperationType type,
      bool redo,
      mitk::DataNode* targetNode,
      mitk::DataNode* sourceNode,
      int sliceNumber,
      itk::ORIENTATION_ENUM orientation
      )
  : mitk::OpGeneralSegmentorBaseCommand(type, redo, targetNode)
  , m_SourceNode(sourceNode)
  , m_SliceNumber(sliceNumber)
  , m_Orientation(orientation)
  {
  };
  ~OpPropagate()
  { };
  mitk::DataNode* GetSourceNode() const { return m_SourceNode; }
  int GetSliceNumber() const { return m_SliceNumber; }
  itk::ORIENTATION_ENUM GetOrientation() const { return m_Orientation; }
  virtual RegionProcessorType* GetProcessor() const = 0;
private:
  mitk::DataNode* m_SourceNode;
  int m_SliceNumber;
  itk::ORIENTATION_ENUM m_Orientation;

};

/**
 * \class OpPropagateUp
 * \brief Class to hold data to pass back to the MIDASGeneralSegmentorView to Undo/Redo the PropagateUp function.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
template <class TSegmentationPixel, class TGreyScalePixel, class TPointDataType, unsigned int VImageDimension>
class OpPropagateUp: public OpPropagate
{
public:
  typedef itk::MIDASPropagateUpProcessor<TSegmentationPixel, TGreyScalePixel, TPointDataType, VImageDimension> UpProcessorType;
  typedef typename UpProcessorType::Pointer UpProcessorPointer;
  OpPropagateUp(
      mitk::OperationType type,
      bool redo,
      mitk::DataNode* targetNode,
      mitk::DataNode* sourceNode,
      int sliceNumber,
      itk::ORIENTATION_ENUM orientation,
      UpProcessorType* processor
      )
  : mitk::OpPropagate(type, redo, targetNode, sourceNode, sliceNumber, orientation)
  , m_Processor(processor)
  { };
  ~OpPropagateUp() { };
  virtual OpPropagate::RegionProcessorType* GetProcessor() const { return m_Processor; }
private:
  UpProcessorPointer m_Processor;
};

/**
 * \class OpPropagateDown
 * \brief Class to hold data to pass back to the MIDASGeneralSegmentorView to Undo/Redo the PropagateDown function.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
template <class TSegmentationPixel, class TGreyScalePixel, class TPointDataType, unsigned int VImageDimension>
class OpPropagateDown: public OpPropagate
{
public:
  typedef itk::MIDASPropagateDownProcessor<TSegmentationPixel, TGreyScalePixel, TPointDataType, VImageDimension> DownProcessorType;
  typedef typename DownProcessorType::Pointer DownProcessorPointer;
  OpPropagateDown(
      mitk::OperationType type,
      bool redo,
      mitk::DataNode* targetNode,
      mitk::DataNode* sourceNode,
      int sliceNumber,
      itk::ORIENTATION_ENUM orientation,
      DownProcessorType* processor
      )
  : mitk::OpPropagate(type, redo, targetNode, sourceNode, sliceNumber, orientation)
  , m_Processor(processor)
  { };
  ~OpPropagateDown() { };
  virtual OpPropagate::RegionProcessorType* GetProcessor() const { return m_Processor; }
private:
  DownProcessorPointer m_Processor;
};

/**
 * \class OpWipe
 * \brief Base class to hold data to pass back to the MIDASGeneralSegmentorView to Undo/Redo the Wipe Slice/Plus/Minus function.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpWipe: public OpGeneralSegmentorBaseCommand
{
public:

  typedef itk::MIDASWipeProcessor<mitk::Tool::DefaultSegmentationDataType, 3> WipeProcessorType;
  OpWipe(
      mitk::OperationType type,
      bool redo,
      mitk::DataNode* targetNode,
      int sliceNumber,
      itk::ORIENTATION_ENUM orientation
      )
  : mitk::OpGeneralSegmentorBaseCommand(type, redo, targetNode)
  , m_SliceNumber(sliceNumber)
  , m_Orientation(orientation)
  { };

  ~OpWipe()
  { };
  int GetSliceNumber() const { return m_SliceNumber; }
  itk::ORIENTATION_ENUM GetOrientation() const { return m_Orientation; }
  virtual OpWipe::WipeProcessorType* GetProcessor() const = 0;
private:
  int m_SliceNumber;
  itk::ORIENTATION_ENUM m_Orientation;
};

/**
 * \class OpWipeSlice
 * \brief Class to hold data to pass back to the MIDASGeneralSegmentorView to Undo/Redo the WipeSlice function.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpWipeSlice: public OpWipe
{
public:

  typedef itk::MIDASWipeSliceProcessor<mitk::Tool::DefaultSegmentationDataType, 3> WipeSliceProcessorType;

  OpWipeSlice(
      mitk::OperationType type,
      bool redo,
      mitk::DataNode* targetNode,
      int sliceNumber,
      itk::ORIENTATION_ENUM orientation,
      WipeSliceProcessorType* processor
      )
  : mitk::OpWipe(type, redo, targetNode, sliceNumber, orientation)
  , m_Processor(processor)
  { };

  ~OpWipeSlice()
  { };
  virtual OpWipe::WipeProcessorType* GetProcessor() const { return m_Processor; }

private:
  WipeSliceProcessorType::Pointer m_Processor;
};

/**
 * \class OpWipePlus
 * \brief Class to hold data to pass back to the MIDASGeneralSegmentorView to Undo/Redo the WipePlus function.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpWipePlus: public OpWipe
{
public:

  typedef itk::MIDASWipePlusProcessor<mitk::Tool::DefaultSegmentationDataType, 3> WipePlusProcessorType;

  OpWipePlus(
      mitk::OperationType type,
      bool redo,
      mitk::DataNode* targetNode,
      int sliceNumber,
      itk::ORIENTATION_ENUM orientation,
      WipePlusProcessorType* processor
      )
  : mitk::OpWipe(type, redo, targetNode, sliceNumber, orientation)
  , m_Processor(processor)
  { };

  ~OpWipePlus()
  { };
  virtual OpWipe::WipeProcessorType* GetProcessor() const { return m_Processor; }
private:
  WipePlusProcessorType::Pointer m_Processor;
};

/**
 * \class OpWipeMinus
 * \brief Class to hold data to pass back to the MIDASGeneralSegmentorView to Undo/Redo the WipeMinus function.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpWipeMinus: public OpWipe
{
public:

  typedef itk::MIDASWipeMinusProcessor<mitk::Tool::DefaultSegmentationDataType, 3> WipeMinusProcessorType;

  OpWipeMinus(
      mitk::OperationType type,
      bool redo,
      mitk::DataNode* targetNode,
      int sliceNumber,
      itk::ORIENTATION_ENUM orientation,
      WipeMinusProcessorType* processor
      )
  : mitk::OpWipe(type, redo, targetNode, sliceNumber, orientation)
  , m_Processor(processor)
  { };

  ~OpWipeMinus()
  { };
  virtual OpWipe::WipeProcessorType* GetProcessor() const { return m_Processor; }
private:
  WipeMinusProcessorType::Pointer m_Processor;
};

/**
 * \class OpRetainMarksNoThresholding
 * \brief Class to hold data to pass back to the MIDASGeneralSegmentorView to Undo/Redo the RetainMarks function.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class OpRetainMarksNoThresholding: public OpGeneralSegmentorBaseCommand
{
public:
  typedef itk::MIDASRetainMarksNoThresholdingProcessor<mitk::Tool::DefaultSegmentationDataType, 3> ProcessorType;

  OpRetainMarksNoThresholding(
      mitk::OperationType type,
      bool redo,
      mitk::DataNode* targetNode,
      int targetSlice,
      int sourceSlice,
      itk::ORIENTATION_ENUM orientation,
      ProcessorType* processor
      )
  : mitk::OpGeneralSegmentorBaseCommand(type, redo, targetNode)
  , m_TargetSlice(targetSlice)
  , m_SourceSlice(sourceSlice)
  , m_Orientation(orientation)
  , m_Processor(processor)
  { };

  ~OpRetainMarksNoThresholding()
  { };
  ProcessorType::Pointer GetProcessor() const { return m_Processor; }
  int GetTargetSlice() const { return m_TargetSlice; }
  int GetSourceSlice() const { return m_SourceSlice; }
  itk::ORIENTATION_ENUM GetOrientation() const { return m_Orientation; }

private:
  int m_TargetSlice;
  int m_SourceSlice;
  itk::ORIENTATION_ENUM m_Orientation;
  ProcessorType::Pointer m_Processor;
};

} // end namespace

#endif
