/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "LightweightCUDAImage.h"
#include <CUDAManager/CUDAManager.h>


//-----------------------------------------------------------------------------
LightweightCUDAImage::LightweightCUDAImage()
  : m_RefCount(0)
  , m_Id(0)
{
}


//-----------------------------------------------------------------------------
unsigned int LightweightCUDAImage::GetId() const
{
  return m_Id;
}


//-----------------------------------------------------------------------------
unsigned int LightweightCUDAImage::GetWidth() const
{
  return m_Width;
}


//-----------------------------------------------------------------------------
unsigned int LightweightCUDAImage::GetHeight() const
{
  return m_Height;
}


//-----------------------------------------------------------------------------
unsigned int LightweightCUDAImage::GetBytePitch() const
{
  return m_BytePitch;
}


//-----------------------------------------------------------------------------
cudaEvent_t LightweightCUDAImage::GetReadyEvent() const
{
  return m_ReadyEvent;
}


//-----------------------------------------------------------------------------
LightweightCUDAImage::~LightweightCUDAImage()
{
  if (m_RefCount)
  {
    bool dead = !m_RefCount->deref();
    if (dead)
    {
      CUDAManager::GetInstance()->AllRefsDropped(*this);
    }
  }
}


//-----------------------------------------------------------------------------
LightweightCUDAImage::LightweightCUDAImage(const LightweightCUDAImage& copyme)
  : m_RefCount(copyme.m_RefCount)
  , m_Device(copyme.m_Device)
  , m_Id(copyme.m_Id)
  , m_ReadyEvent(copyme.m_ReadyEvent)
  , m_DevicePtr(copyme.m_DevicePtr)
  , m_SizeInBytes(copyme.m_SizeInBytes)
  , m_Width(copyme.m_Width)
  , m_Height(copyme.m_Height)
  , m_BytePitch(copyme.m_BytePitch)
  , m_LastUsedByStream(copyme.m_LastUsedByStream)
{
  if (m_RefCount)
  {
    m_RefCount->ref();
  }
}


//-----------------------------------------------------------------------------
LightweightCUDAImage& LightweightCUDAImage::operator=(const LightweightCUDAImage& assignme)
{
  if (m_RefCount)
  {
    bool dead = !m_RefCount->deref();
    if (dead)
    {
      CUDAManager::GetInstance()->AllRefsDropped(*this);
    }
  }

  m_RefCount    = assignme.m_RefCount;
  m_Device      = assignme.m_Device;
  m_Id          = assignme.m_Id;
  m_ReadyEvent  = assignme.m_ReadyEvent;
  m_DevicePtr   = assignme.m_DevicePtr;
  m_SizeInBytes = assignme.m_SizeInBytes;
  m_Width       = assignme.m_Width;
  m_Height      = assignme.m_Height;
  m_BytePitch   = assignme.m_BytePitch;
  m_LastUsedByStream = assignme.m_LastUsedByStream;

  if (m_RefCount)
  {
    m_RefCount->ref();
  }

  return *this;
}
