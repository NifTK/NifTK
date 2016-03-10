/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNVidiaSDIDataType.h"

namespace niftk
{

//-----------------------------------------------------------------------------
NVidiaSDIDataType::NVidiaSDIDataType(unsigned int cookie,
                                     unsigned int sequenceNum,
                                     NVidiaSDITimeType gpuArrivalTime)
: m_MagicCookie(cookie), m_SequenceNumber(sequenceNum), m_GpuArrivalTime(gpuArrivalTime)
{
}


//-----------------------------------------------------------------------------
NVidiaSDIDataType::~NVidiaSDIDataType()
{
}


//-----------------------------------------------------------------------------
unsigned int NVidiaSDIDataType::GetSequenceNumber() const
{
  return m_SequenceNumber;
}


//-----------------------------------------------------------------------------
unsigned int NVidiaSDIDataType::GetCookie() const
{
  return m_MagicCookie;
}

} // end namespace

