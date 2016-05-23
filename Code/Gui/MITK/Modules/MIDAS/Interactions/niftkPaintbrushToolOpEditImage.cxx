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

PaintbrushToolOpEditImage::PaintbrushToolOpEditImage(
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

PaintbrushToolOpEditImage::~PaintbrushToolOpEditImage()
{
}

bool PaintbrushToolOpEditImage::IsRedo() const
{
  return m_Redo;
}

int PaintbrushToolOpEditImage::GetImageNumber() const
{
  return m_ImageNumber;
}

unsigned char PaintbrushToolOpEditImage::GetValueToWrite() const
{
  return m_ValueToWrite;
}

mitk::Image* PaintbrushToolOpEditImage::GetImageToEdit() const
{
  return m_ImageToEdit;
}

mitk::DataNode* PaintbrushToolOpEditImage::GetNodeToEdit() const
{
  return m_NodeToEdit;
}

PaintbrushToolOpEditImage::ProcessorType::Pointer PaintbrushToolOpEditImage::GetProcessor() const
{
  return m_Processor;
}

}
