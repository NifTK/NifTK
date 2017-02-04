/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkOpenCVVideoDataType.h"
#include <cstring>

namespace niftk
{

//-----------------------------------------------------------------------------
OpenCVVideoDataType::~OpenCVVideoDataType()
{
  if (m_Image != nullptr)
  {
    cvReleaseImage(&m_Image);
  }
}


//-----------------------------------------------------------------------------
OpenCVVideoDataType::OpenCVVideoDataType()
: m_Image(nullptr)
{
}


//-----------------------------------------------------------------------------
OpenCVVideoDataType::OpenCVVideoDataType(IplImage *image)
: m_Image(image)
{
}


//-----------------------------------------------------------------------------
OpenCVVideoDataType::OpenCVVideoDataType(const OpenCVVideoDataType& other)
{
  this->CloneImage(other.m_Image);
}


//-----------------------------------------------------------------------------
OpenCVVideoDataType::OpenCVVideoDataType(OpenCVVideoDataType&& other)
: m_Image(other.m_Image)
{
  other.m_Image = nullptr;
}


//-----------------------------------------------------------------------------
OpenCVVideoDataType& OpenCVVideoDataType::operator=(const OpenCVVideoDataType& other)
{
  IGIDataType::operator=(other);
  this->CloneImage(other.m_Image);
  return *this;
}


//-----------------------------------------------------------------------------
OpenCVVideoDataType& OpenCVVideoDataType::operator=(OpenCVVideoDataType&& other)
{
  IGIDataType::operator=(other);
  if (m_Image != nullptr)
  {
    cvReleaseImage(&m_Image);
  }
  m_Image = other.m_Image;
  other.m_Image = nullptr;
  return *this;
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataType::Clone(const IGIDataType& other)
{
  IGIDataType::Clone(other);
  const OpenCVVideoDataType* tmp = dynamic_cast<const OpenCVVideoDataType*>(&other);
  if (tmp != nullptr)
  {
    this->CloneImage(tmp->GetImage());
  }
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataType::CloneImage(const IplImage *image)
{
  if (m_Image != nullptr)
  {
    if (   m_Image->width == image->width
        && m_Image->height == image->height
        && m_Image->depth == image->depth
        && m_Image->nChannels == image->nChannels
        && m_Image->imageSize == image->imageSize
       )
    {
      std::memcpy(m_Image->imageData, image->imageData, image->imageSize);
    }
    else
    {
      cvReleaseImage(&m_Image);
      m_Image = cvCloneImage(image);
    }
  }
  else
  {
    m_Image = cvCloneImage(image);
  }
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataType::SetImage(const IplImage *image)
{
  this->CloneImage(image);
}


//-----------------------------------------------------------------------------
const IplImage* OpenCVVideoDataType::GetImage() const
{
  return m_Image;
}

} // end namespace
