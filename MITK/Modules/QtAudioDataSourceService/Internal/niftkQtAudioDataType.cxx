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
: m_AudioBlob(0)
, m_Length(0)
{
}


//-----------------------------------------------------------------------------
QtAudioDataType::~QtAudioDataType()
{
  delete m_AudioBlob;
}


//-----------------------------------------------------------------------------
void QtAudioDataType::SetBlob(const char* blob, std::size_t length)
{
  delete m_AudioBlob;
  m_AudioBlob = blob;
  m_Length = length;
}


//-----------------------------------------------------------------------------
std::pair<const char*, std::size_t> QtAudioDataType::GetBlob() const
{
  return std::make_pair(m_AudioBlob, m_Length);
}

} // end namespace
