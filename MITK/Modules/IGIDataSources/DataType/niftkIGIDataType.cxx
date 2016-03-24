/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataType.h"

namespace niftk
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
}

//-----------------------------------------------------------------------------
IGIDataType::~IGIDataType()
{
}


//-----------------------------------------------------------------------------
IGIDataType::IGITimeType IGIDataType::GetTimeStampInNanoSeconds() const
{
  return m_TimeStamp;
}


//-----------------------------------------------------------------------------
void IGIDataType::SetTimeStampInNanoSeconds(const IGIDataType::IGITimeType& time)
{
  m_TimeStamp = time;
  this->Modified();
}

} // end namespace
