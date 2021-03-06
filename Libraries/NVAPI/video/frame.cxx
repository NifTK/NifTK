/*=============================================================================

  libvideo: a library for SDI video processing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "stdafx.h"
#include <video/frame.h>


namespace video
{


int StreamFormat::get_width() const
{
    switch (format)
    {
        case PF_NONE:
        default:
            return 0;
        case PF_487:
            return 720;
        case PF_576:
            return 720;
        case PF_720:
            return 1280;
        case PF_1035:
            return 1920;
        case PF_1080:
            return 1920;
        case PF_2048:
            return 2048;
    }
}

int StreamFormat::get_height() const
{
    switch (format)
    {
        // by default the enum value is the image height
        default: 
            return (int) format;
        // but unfortunately, video standards are a mess and there are special cases
        case PF_2048:
            return 1080;
    }

    // should never get here
    assert(false);
}

float StreamFormat::get_refreshrate() const
{
    return (float) refreshrate / 1000.0f;
}


} // namespace
