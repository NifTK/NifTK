/*=============================================================================

  libvideo: a library for SDI video processing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "stdafx.h"
#include "conversion.h"


namespace video
{


void convert_nv12_to_rgba(void* src, unsigned int srcpitch, unsigned int srcheight, void* dst, unsigned int dstpitch, unsigned int width, unsigned int height)
{
    // one full-res plane of Y, one byte per pixel.
    assert(srcpitch >= width);
    assert(dstpitch >= width * 4);
    assert((dstpitch % 4) == 0);

    // src can be higher than output. there's usually encoder padding, or something similar.
    assert(srcheight >= height);

    // padding bits in the source image
    unsigned int  src_trail = srcpitch - width;
    unsigned int  dst_trail = (dstpitch / 4) - width;

    unsigned char*    srcptr_y  = (unsigned char*) src;
    unsigned int*     dstptr = (unsigned int*) dst;

    for (unsigned int y = 0; y < height; ++y)
    {
        unsigned char*   srcptr_uv = &((unsigned char*) src)[srcheight * srcpitch + y / 2 * srcpitch];

        for (unsigned int x = 0; x < width; ++x)
        {
            float   cy = srcptr_y[0]  / 255.0f;
            float   cu = srcptr_uv[0] / 255.0f;
            float   cv = srcptr_uv[1] / 255.0f;

            cu -= 0.5f;
            cv -= 0.5f;

            float cr = cy + 1.403f * cv;
            float cg = cy - 0.344f * cu - 0.714f * cv;
            float cb = cy + 1.770f * cu;

            cr = std::min(1.0f, std::max(0.0f, cr)) * 255.0f;
            cg = std::min(1.0f, std::max(0.0f, cg)) * 255.0f;
            cb = std::min(1.0f, std::max(0.0f, cb)) * 255.0f;

            *dstptr = (((int) cr) << 0) | (((int) cg) << 8) | (((int) cb) << 16) | 0xFF000000;
            ++dstptr;

            ++srcptr_y;
            // uv plane is half the resolution
            if (x & 1)
                srcptr_uv += 2;
        }

        srcptr_y  += src_trail;
        dstptr += dst_trail;
    }
}


} // namespace
