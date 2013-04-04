/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkIGINVidiaDataType.h"

namespace mitk
{

//-----------------------------------------------------------------------------
IGINVidiaDataType::IGINVidiaDataType()
  : m_MagicCookie(0), m_SequenceNumber(0), m_GpuArrivalTime(0)
{
}

//-----------------------------------------------------------------------------
IGINVidiaDataType::~IGINVidiaDataType()
{
}


unsigned int IGINVidiaDataType::GetSequenceNumber() const
{
  return m_SequenceNumber;
}


void IGINVidiaDataType::SetValues(unsigned int cookie, unsigned int sn, unsigned __int64 gputime)
{
  m_MagicCookie = cookie;
  m_SequenceNumber = sn;
  m_GpuArrivalTime = gputime;
}

} // end namespace

