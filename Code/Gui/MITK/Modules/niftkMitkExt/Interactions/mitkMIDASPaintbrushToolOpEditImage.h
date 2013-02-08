/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKMIDASPAINTBRUSHTOOLOPEDITIMAGE_H
#define MITKMIDASPAINTBRUSHTOOLOPEDITIMAGE_H

#include "niftkMitkExtExports.h"
#include "mitkOperation.h"
#include "mitkOperationActor.h"
#include "mitkTool.h"
#include "itkImage.h"
#include "itkMIDASImageUpdatePixelWiseSingleValueProcessor.h"

namespace mitk
{

/**
 * \class MIDASPaintbrushToolOpEditImage
 * \brief Operation class to hold data to pass back to this MIDASPaintbrushTool,
 * so that this MIDASPaintbrushTool can execute the Undo/Redo command.
 */
class NIFTKMITKEXT_EXPORT MIDASPaintbrushToolOpEditImage: public mitk::Operation
{
public:
  typedef itk::MIDASImageUpdatePixelWiseSingleValueProcessor<mitk::Tool::DefaultSegmentationDataType, 3> ProcessorType;

  MIDASPaintbrushToolOpEditImage(
      mitk::OperationType type,
      bool redo,
      int imageNumber,
      unsigned char valueToWrite,
      mitk::Image* imageToEdit,
      mitk::DataNode* nodeToEdit,
      ProcessorType* processor
      );
  ~MIDASPaintbrushToolOpEditImage() {};
  bool IsRedo() const { return m_Redo; }
  int GetImageNumber() const { return m_ImageNumber; }
  unsigned char GetValueToWrite() const { return m_ValueToWrite; }
  mitk::Image* GetImageToEdit() const { return m_ImageToEdit; }
  mitk::DataNode* GetNodeToEdit() const { return m_NodeToEdit; }
  ProcessorType::Pointer GetProcessor() const { return m_Processor; }

private:
  bool m_Redo;
  int m_ImageNumber;
  unsigned char m_ValueToWrite;
  mitk::Image* m_ImageToEdit;
  mitk::DataNode* m_NodeToEdit;
  ProcessorType::Pointer m_Processor;

};

} // end namespace

#endif
