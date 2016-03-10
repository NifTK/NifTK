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

mitk::MIDASPaintbrushToolOpEditImage::~MIDASPaintbrushToolOpEditImage()
{
}

bool mitk::MIDASPaintbrushToolOpEditImage::IsRedo() const
{
  return m_Redo;
}

int mitk::MIDASPaintbrushToolOpEditImage::GetImageNumber() const
{
  return m_ImageNumber;
}

unsigned char mitk::MIDASPaintbrushToolOpEditImage::GetValueToWrite() const
{
  return m_ValueToWrite;
}

mitk::Image* mitk::MIDASPaintbrushToolOpEditImage::GetImageToEdit() const
{
  return m_ImageToEdit;
}

mitk::DataNode* mitk::MIDASPaintbrushToolOpEditImage::GetNodeToEdit() const
{
  return m_NodeToEdit;
}

mitk::MIDASPaintbrushToolOpEditImage::ProcessorType::Pointer mitk::MIDASPaintbrushToolOpEditImage::GetProcessor() const
{
  return m_Processor;
}
