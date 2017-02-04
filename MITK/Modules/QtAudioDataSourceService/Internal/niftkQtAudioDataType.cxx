/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQtAudioDataType.h"

namespace niftk
{

//-----------------------------------------------------------------------------
QtAudioDataType::QtAudioDataType()
: m_AudioBlob(nullptr)
, m_Length(0)
{
}


//-----------------------------------------------------------------------------
QtAudioDataType::~QtAudioDataType()
{
  delete m_AudioBlob;
}


//-----------------------------------------------------------------------------
QtAudioDataType::QtAudioDataType(const QtAudioDataType& other)
{
  m_Length = other.m_Length;
  std::memcpy(m_AudioBlob, other.m_AudioBlob, m_Length);
}


//-----------------------------------------------------------------------------
QtAudioDataType& QtAudioDataType::operator=(const QtAudioDataType& other)
{
  if (m_AudioBlob != nullptr)
  {
    delete m_AudioBlob;
  }
  m_Length = other.m_Length;
  std::memcpy(m_AudioBlob, other.m_AudioBlob, m_Length);
  return *this;
}


//-----------------------------------------------------------------------------
QtAudioDataType::QtAudioDataType(QtAudioDataType&& other)
{
  m_Length = other.m_Length;
  std::memcpy(m_AudioBlob, other.m_AudioBlob, m_Length);
  other.m_AudioBlob = nullptr;
}


//-----------------------------------------------------------------------------
QtAudioDataType& QtAudioDataType::operator=(QtAudioDataType&& other)
{
  if (m_AudioBlob != nullptr)
  {
    delete m_AudioBlob;
  }
  m_Length = other.m_Length;
  std::memcpy(m_AudioBlob, other.m_AudioBlob, m_Length);
  other.m_AudioBlob = nullptr;
  return *this;
}


//-----------------------------------------------------------------------------
void QtAudioDataType::SetBlob(char* blob, std::size_t length)
{
  delete m_AudioBlob;
  m_AudioBlob = blob;
  m_Length = length;
}


//-----------------------------------------------------------------------------
std::pair<char*, std::size_t> QtAudioDataType::GetBlob() const
{
  return std::make_pair(m_AudioBlob, m_Length);
}

} // end namespace
