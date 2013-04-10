/*=============================================================================

  libvideo: a library for SDI video processing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#pragma once
#ifndef LIBVIDEO_SDIOUTPUT_H_EA0C3F86122E4C0CAA94B17839B8DB01
#define LIBVIDEO_SDIOUTPUT_H_EA0C3F86122E4C0CAA94B17839B8DB01

#include <video/device.h>
#include <video/dllexport.h>


namespace video
{


class SDIOutputImpl;
class LIBVIDEO_DLL_EXPORTS SDIOutput
{
protected:
    SDIOutputImpl*      pimpl;


public:
    // all streams have same format
    // there can be max 2 streams
    SDIOutput(SDIDevice* dev, StreamFormat format, int streams = 1, int customtextureid1 = 0, int customtextureid2 = 0);
    ~SDIOutput();

private:
    SDIOutput(const SDIOutput& copyme);
    SDIOutput& operator=(const SDIOutput& assignme);


public:
    // stick these into your fbo for rendering
    int get_texture_id(int streamno);

    // beware: the timestamp returned here appears to be unrelated to the capture timestamp!
    unsigned __int64 get_current_hardware_time();

    // zero means asap in queued order
    // FIXME: does this block if we fill up the queue?
    //        write a test tool that outputs a grey screen
    //         and then keeps queueing red frames but a few seconds ahead of gpu time
    //         if old entries simply fall off the queue the red frames will never be displayed
    void present(unsigned __int64 fromwhen = 0);

    // FIXME: need a way to figure out if frames where dropped or repeated
};


} // namespace

#endif // LIBVIDEO_SDIOUTPUT_H_EA0C3F86122E4C0CAA94B17839B8DB01
