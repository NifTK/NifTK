/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkIGIDataType.h"
#include <itkObjectFactory.h>
#include <igtlTimeStamp.h>

namespace mitk
{

//-----------------------------------------------------------------------------
IGIDataType::IGIDataType()
: m_TimeStamp(0)
, m_Duration(0)
, m_FrameId(0)
, m_IsSaved(false)
, m_ShouldBeSaved(false)
, m_FileName("")
{
  m_TimeStamp = igtl::TimeStamp::New();
}

//-----------------------------------------------------------------------------
IGIDataType::~IGIDataType()
{
}


//-----------------------------------------------------------------------------
igtlUint64 IGIDataType::GetTimeStampInNanoSeconds() const
{
  return m_TimeStamp->GetTimeInNanoSeconds();
}


//-----------------------------------------------------------------------------
void IGIDataType::SetTimeStampInNanoSeconds(const igtlUint64& time)
{
  m_TimeStamp->SetTimeInNanoSeconds(time);
}

} // end namespace

