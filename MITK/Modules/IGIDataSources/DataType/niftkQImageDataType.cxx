/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQImageDataType.h"
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
QImageDataType::QImageDataType()
: m_Image(nullptr)
{
}


//-----------------------------------------------------------------------------
QImageDataType::~QImageDataType()
{
  delete m_Image;
}


//-----------------------------------------------------------------------------
void QImageDataType::DeepCopy(const QImage& image)
{
  if (m_Image != nullptr)
  {
    mitkThrow() << "Image is already set";
  }
  m_Image = new QImage(image.copy());
}


//-----------------------------------------------------------------------------
void QImageDataType::ShallowCopy(const QImage& image)
{
  if (m_Image != nullptr)
  {
    mitkThrow() << "Image is already set";
  }
  m_Image = new QImage(image);
}


//-----------------------------------------------------------------------------
const QImage* QImageDataType::GetImage()
{
  return m_Image;
}

} // end namespace
