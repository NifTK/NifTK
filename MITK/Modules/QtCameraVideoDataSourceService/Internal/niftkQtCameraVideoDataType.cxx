/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQtCameraVideoDataType.h"

namespace niftk
{

//-----------------------------------------------------------------------------
QtCameraVideoDataType::QtCameraVideoDataType()
{
}


//-----------------------------------------------------------------------------
QtCameraVideoDataType::~QtCameraVideoDataType()
{
}


//-----------------------------------------------------------------------------
void QtCameraVideoDataType::CloneImage(const QImage& image)
{
  m_Image = image.copy();
}


//-----------------------------------------------------------------------------
const QImage* QtCameraVideoDataType::GetImage()
{
  return &m_Image;
}

} // end namespace
