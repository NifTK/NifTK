/*=============================================================================

  libvideo: a library for SDI video processing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#pragma once
#ifndef LIBVIDEO_CONVERSION_H_794A3B56C8CF4B2686CCCB149B3D7749
#define LIBVIDEO_CONVERSION_H_794A3B56C8CF4B2686CCCB149B3D7749

#include <video/dllexport.h>


namespace video
{


// 8 bit only
// pitch in bytes
// width/height in pixels
void convert_nv12_to_rgba(void* src, unsigned int srcpitch, unsigned int srcheight, void* dst, unsigned int dstpitch, unsigned int width, unsigned int height);


} // namespace

#endif // LIBVIDEO_CONVERSION_H_794A3B56C8CF4B2686CCCB149B3D7749
