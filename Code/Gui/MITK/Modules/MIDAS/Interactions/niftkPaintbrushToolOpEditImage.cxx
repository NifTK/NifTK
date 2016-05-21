/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPaintbrushToolOpEditImage.h"

namespace niftk
{

MIDASPaintbrushToolOpEditImage::MIDASPaintbrushToolOpEditImage(
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

MIDASPaintbrushToolOpEditImage::~MIDASPaintbrushToolOpEditImage()
{
}

bool MIDASPaintbrushToolOpEditImage::IsRedo() const
{
  return m_Redo;
}

int MIDASPaintbrushToolOpEditImage::GetImageNumber() const
{
  return m_ImageNumber;
}

unsigned char MIDASPaintbrushToolOpEditImage::GetValueToWrite() const
{
  return m_ValueToWrite;
}

mitk::Image* MIDASPaintbrushToolOpEditImage::GetImageToEdit() const
{
  return m_ImageToEdit;
}

mitk::DataNode* MIDASPaintbrushToolOpEditImage::GetNodeToEdit() const
{
  return m_NodeToEdit;
}

MIDASPaintbrushToolOpEditImage::ProcessorType::Pointer MIDASPaintbrushToolOpEditImage::GetProcessor() const
{
  return m_Processor;
}

}
