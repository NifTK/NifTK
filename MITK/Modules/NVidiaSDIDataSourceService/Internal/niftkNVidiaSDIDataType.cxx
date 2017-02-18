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
NVidiaSDIDataType::~NVidiaSDIDataType()
{
}


//-----------------------------------------------------------------------------
NVidiaSDIDataType::NVidiaSDIDataType(unsigned int cookie,
                                     unsigned int sequenceNum,
                                     NVidiaSDITimeType gpuArrivalTime)
: m_MagicCookie(cookie)
, m_SequenceNumber(sequenceNum)
, m_GpuArrivalTime(gpuArrivalTime)
{
}


//-----------------------------------------------------------------------------
NVidiaSDIDataType::NVidiaSDIDataType(const NVidiaSDIDataType& other)
: m_MagicCookie(other.m_MagicCookie)
, m_SequenceNumber(other.m_SequenceNumber)
, m_GpuArrivalTime(other.m_GpuArrivalTime)
{

}


//-----------------------------------------------------------------------------
NVidiaSDIDataType::NVidiaSDIDataType(const NVidiaSDIDataType&& other)
: m_MagicCookie(std::move(other.m_MagicCookie))
, m_SequenceNumber(std::move(other.m_SequenceNumber))
, m_GpuArrivalTime(std::move(other.m_GpuArrivalTime))
{
}


//-----------------------------------------------------------------------------
NVidiaSDIDataType& NVidiaSDIDataType::operator=(const NVidiaSDIDataType& other)
{
  m_MagicCookie = other.m_MagicCookie;
  m_SequenceNumber = other.m_SequenceNumber;
  m_GpuArrivalTime = other.m_GpuArrivalTime;
  return *this;
}


//-----------------------------------------------------------------------------
NVidiaSDIDataType& NVidiaSDIDataType::operator=(const NVidiaSDIDataType&& other)
{
  m_MagicCookie = std::move(other.m_MagicCookie);
  m_SequenceNumber = std::move(other.m_SequenceNumber);
  m_GpuArrivalTime = std::move(other.m_GpuArrivalTime);
  return *this;
}


//-----------------------------------------------------------------------------
void NVidiaSDIDataType::Clone(const IGIDataType& other)
{
  IGIDataType::Clone(other);
  const NVidiaSDIDataType* tmp = dynamic_cast<const NVidiaSDIDataType*>(&other);
  if (tmp != nullptr)
  {
    m_MagicCookie = tmp->m_MagicCookie;
    m_SequenceNumber = tmp->m_SequenceNumber;
    m_GpuArrivalTime = tmp->m_GpuArrivalTime;
  }
  else
  {
    mitkThrow() << "Incorrect data type provided";
  }
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

