/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkUltrasonixDataType.h"

namespace niftk
{

//-----------------------------------------------------------------------------
UltrasonixDataType::UltrasonixDataType()
: m_Image(NULL)
{
}


//-----------------------------------------------------------------------------
UltrasonixDataType::~UltrasonixDataType()
{
  if (m_Image != NULL)
  {
    cvReleaseImage(&m_Image);
  }
}


//-----------------------------------------------------------------------------
void UltrasonixDataType::CloneImage(const IplImage *image)
{
  if (m_Image != NULL)
  {
    cvReleaseImage(&m_Image);
  }
  m_Image = cvCloneImage(image);
}


//-----------------------------------------------------------------------------
const IplImage* UltrasonixDataType::GetImage()
{
  return m_Image;
}

} // end namespace

