/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkOpenCVVideoDataType.h"

namespace niftk
{

//-----------------------------------------------------------------------------
OpenCVVideoDataType::OpenCVVideoDataType()
: m_Image(NULL)
{
}


//-----------------------------------------------------------------------------
OpenCVVideoDataType::~OpenCVVideoDataType()
{
  if (m_Image != NULL)
  {
    cvReleaseImage(&m_Image);
  }
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataType::CloneImage(const IplImage *image)
{
  if (m_Image != NULL)
  {
    cvReleaseImage(&m_Image);
  }
  m_Image = cvCloneImage(image);
}


//-----------------------------------------------------------------------------
const IplImage* OpenCVVideoDataType::GetImage()
{
  return m_Image;
}

} // end namespace

