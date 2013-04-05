/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASPaintbrushToolOpEditImage.h"

mitk::MIDASPaintbrushToolOpEditImage::MIDASPaintbrushToolOpEditImage(
    mitk::OperationType type,
    bool redo,
    int imageNumber,
    unsigned char valueToWrite,
    mitk::Image* imageToEdit,
    mitk::DataNode* nodeToEdit,
    ProcessorType* processor
    )
: mitk::Operation(type)
, m_Redo(redo)
, m_ImageNumber(imageNumber)
, m_ValueToWrite(valueToWrite)
, m_ImageToEdit(imageToEdit)
, m_NodeToEdit(nodeToEdit)
, m_Processor(processor)
{
}
