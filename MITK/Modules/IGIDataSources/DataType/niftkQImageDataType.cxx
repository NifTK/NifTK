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
#include <cstring>

namespace niftk
{

//-----------------------------------------------------------------------------
QImageDataType::~QImageDataType()
{
  if (m_Image != nullptr)
  {
    delete m_Image;
  }
}


//-----------------------------------------------------------------------------
QImageDataType::QImageDataType()
: m_Image(nullptr)
{
}


//-----------------------------------------------------------------------------
QImageDataType::QImageDataType(QImage *image)
: m_Image(image)
{
}


//-----------------------------------------------------------------------------
QImageDataType::QImageDataType(const QImageDataType& other)
: IGIDataType(other)
{
  this->CloneImage(other.m_Image);
}


//-----------------------------------------------------------------------------
QImageDataType::QImageDataType(QImageDataType&& other)
: IGIDataType(other)
{
  this->CloneImage(other.m_Image);
  other.m_Image = nullptr;
}


//-----------------------------------------------------------------------------
QImageDataType& QImageDataType::operator=(const QImageDataType& other)
{
  IGIDataType::operator=(other);
  this->CloneImage(other.m_Image);
  return *this;
}


//-----------------------------------------------------------------------------
QImageDataType& QImageDataType::operator=(QImageDataType&& other)
{
  IGIDataType::operator=(other);
  if (m_Image != nullptr)
  {
    delete m_Image;
  }
  m_Image = other.m_Image;
  other.m_Image = nullptr;
  return *this;
}


//-----------------------------------------------------------------------------
void QImageDataType::Clone(const IGIDataType& other)
{
  IGIDataType::Clone(other);
  const QImageDataType* tmp = dynamic_cast<const QImageDataType*>(&other);
  if (tmp != nullptr)
  {
    this->CloneImage(tmp->GetImage());
  }
  else
  {
    mitkThrow() << "Incorrect data type provided";
  }
}


//-----------------------------------------------------------------------------
void QImageDataType::CloneImage(const QImage *image)
{
  if (m_Image != nullptr)
  {
    if (   m_Image->width() == image->width()
        && m_Image->height() == image->height()
        && m_Image->byteCount() == image->byteCount()
        && m_Image->format() == image->format()
       )
    {
      std::memcpy(m_Image->bits(), image->bits(), image->byteCount());
    }
    else
    {
      delete m_Image;
      m_Image = new QImage(image->copy());
    }
  }
  else
  {
    m_Image = new QImage(image->copy());
  }
}


//-----------------------------------------------------------------------------
void QImageDataType::SetImage(const QImage *image)
{
  this->CloneImage(image);
}


//-----------------------------------------------------------------------------
const QImage* QImageDataType::GetImage() const
{
  return m_Image;
}

} // end namespace
