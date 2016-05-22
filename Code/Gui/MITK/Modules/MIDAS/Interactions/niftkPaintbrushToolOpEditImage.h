/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPaintbrushToolOpEditImage_h
#define niftkPaintbrushToolOpEditImage_h

#include "niftkMIDASExports.h"

#include <itkImage.h>

#include <mitkOperation.h>
#include <mitkOperationActor.h>
#include <mitkTool.h>

#include <itkMIDASImageUpdatePixelWiseSingleValueProcessor.h>

namespace niftk
{

/**
 * \class PaintbrushToolOpEditImage
 * \brief Operation class to hold data to pass back to this PaintbrushTool,
 * so that this PaintbrushTool can execute the Undo/Redo command.
 */
class NIFTKMIDAS_EXPORT PaintbrushToolOpEditImage: public mitk::Operation
{
public:
  typedef itk::MIDASImageUpdatePixelWiseSingleValueProcessor<mitk::Tool::DefaultSegmentationDataType, 3> ProcessorType;

  PaintbrushToolOpEditImage(
      mitk::OperationType type,
      bool redo,
      int imageNumber,
      unsigned char valueToWrite,
      mitk::Image* imageToEdit,
      mitk::DataNode* nodeToEdit,
      ProcessorType* processor
      );

  ~PaintbrushToolOpEditImage();

  bool IsRedo() const;

  int GetImageNumber() const;

  unsigned char GetValueToWrite() const;

  mitk::Image* GetImageToEdit() const;

  mitk::DataNode* GetNodeToEdit() const;

  ProcessorType::Pointer GetProcessor() const;

private:

  bool m_Redo;
  int m_ImageNumber;
  unsigned char m_ValueToWrite;
  mitk::Image* m_ImageToEdit;
  mitk::DataNode* m_NodeToEdit;
  ProcessorType::Pointer m_Processor;

};

}

#endif
