/*=============================================================================

  libvideo: a library for SDI video processing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#pragma once
#ifndef LIBVIDEO_DEVICEIMPL_H_37D12858ED034507AD80F5A4F6F63B7D
#define LIBVIDEO_DEVICEIMPL_H_37D12858ED034507AD80F5A4F6F63B7D


namespace video
{


class SDIDeviceImpl
{
public:
    NvVioHandle     handle;
    // this id is needed to match wgl devices to nvapi devices
    NvU32           id;
    NVVIOCAPS       caps;


    SDIDeviceImpl()
    {
    }
};


} // namespace

#endif // LIBVIDEO_DEVICEIMPL_H_37D12858ED034507AD80F5A4F6F63B7D
