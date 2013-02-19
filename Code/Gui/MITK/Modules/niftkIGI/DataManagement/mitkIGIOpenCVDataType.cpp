/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkIGIOpenCVDataType.h"

namespace mitk
{

//-----------------------------------------------------------------------------
IGIOpenCVDataType::IGIOpenCVDataType()
: m_Image(NULL)
{
}

//-----------------------------------------------------------------------------
IGIOpenCVDataType::~IGIOpenCVDataType()
{
  if (m_Image != NULL)
  {
    cvReleaseImage(&m_Image);
  }
}


//-----------------------------------------------------------------------------
void IGIOpenCVDataType::CloneImage(const IplImage *image)
{
  if (m_Image != NULL)
  {
    cvReleaseImage(&m_Image);
  }
  m_Image = cvCloneImage(image);
}


//-----------------------------------------------------------------------------
const IplImage* IGIOpenCVDataType::GetImage()
{
  return m_Image;
}

} // end namespace

