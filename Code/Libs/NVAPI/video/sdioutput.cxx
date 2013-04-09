/*=============================================================================

  libvideo: a library for SDI video processing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "stdafx.h"
#include <video/sdioutput.h>
#include "deviceimpl.h"


static std::ostream& operator<<(std::ostream& os, const SYSTEMTIME& s)
{
    os << s.wYear << '_' << s.wMonth << '_' << s.wDay << '_' << s.wHour << '_' << s.wMinute << '_' << s.wSecond << '_' << s.wMilliseconds;
    return os;
}


namespace video
{


static NVVIOSIGNALFORMAT map_streamformat_to_nvsignalformat(StreamFormat format)
{
    if (format.format == StreamFormat::PF_1080 && format.refreshrate == StreamFormat::RR_25)
        return NVVIOSIGNALFORMAT_1080P_25_00_SMPTE274;
    if (format.format == StreamFormat::PF_1080 && format.refreshrate == StreamFormat::RR_30)
        return NVVIOSIGNALFORMAT_1080P_30_00_SMPTE274;
    // so this is ambigous with the progressive and interlaced versions
    // to make our live easier we only support progressive (output)
    if (format.format == StreamFormat::PF_1080 && format.refreshrate == StreamFormat::RR_29_97)
        return NVVIOSIGNALFORMAT_1080P_29_97_SMPTE274;

    // FIXME: i want to know when i'm testing combinations that i havent implemented yet
    assert(false);
    return NVVIOSIGNALFORMAT_NONE;
}


struct FrameStats
{
    unsigned int        frameno;

    unsigned int        gpu_scanout_count;
    unsigned __int64    gpu_scanout_time;

    unsigned __int64    gpu_queued_time;
    SYSTEMTIME          wallclock_queued_time;
};

class SDIOutputImpl
{
public:
    HGLRC               oglrc;
    std::vector<GLuint> textures;
    bool                isusingcustomtextures;

    int                 videoslot;

    // two timer sets for pingpong
    // first: present time
    // second: present duration
    std::pair<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, unsigned int> >    gputimers;
    std::vector<FrameStats>     framestats;

    SDIOutputImpl()
        : oglrc(0), isusingcustomtextures(false), videoslot(0)
    {
    }
};


SDIOutput::~SDIOutput()
{
    if (pimpl)
    {
        assert(wglGetCurrentContext() == pimpl->oglrc);


    // FIXME: unbind device

        std::ofstream   logfile("sdioutput.log");
        for (int i = 0; i < pimpl->framestats.size(); ++i)
        {
            logfile << 
                "frameno=" << pimpl->framestats[i].frameno << 
                ", gpu_scanout_count=" << pimpl->framestats[i].gpu_scanout_count << 
                ", gpu_scanout_time=" << pimpl->framestats[i].gpu_scanout_time << 
                ", gpu_queued_time=" << pimpl->framestats[i].gpu_queued_time << 
                ", wallclock_queued_time=" << pimpl->framestats[i].wallclock_queued_time << 
                std::endl;
        }
        
        delete pimpl;
    }
}

SDIOutput::SDIOutput(SDIDevice* dev, StreamFormat format, int streams, int customtextureid1, int customtextureid2)
    : pimpl(new SDIOutputImpl)
{
    assert(dev != 0);
    assert(dev->get_type() == SDIDevice::OUTPUT);

    if (streams <= 0)
        throw std::logic_error("There has to be at least one output stream");
    // this may change with newer versions of the hardware
    // so if this code never gets updated this check is still fine here
    // caller is supposed to map his manyrequested output streams to the two available
    if (streams > 2)
        throw std::logic_error("Max 2 streams are supported");

    // FIXME: add sanity check: if you are trying to use customtextures then you have to do so for all streams!


    // FIXME: this doesnt work as i expected it: the card is always running!
//  if (NvAPI_VIO_IsRunning(dev->get_pimpl()->handle))
//      throw std::logic_error("SDI output device is already doing stuff");

    // we use the ogl context for debugging: has to be current every time you call into the class
    pimpl->oglrc = wglGetCurrentContext();
    if (pimpl->oglrc == 0)
        throw std::logic_error("There is no OpenGL context current");
    // dc is not that interesting, we just need it around (see further below)
    HDC dc = wglGetCurrentDC();
    assert(dc != 0);

    // should be safe to call this more than once
    if (glewInit() != GLEW_OK)
        throw std::runtime_error("Cannot initialise OpenGL function pointers (via GLEW)");
    // FIXME: check that the necessary extensions are actually present!


    // there is only one device class: sdi (output)
    if (NvAPI_VIO_Open(dev->get_pimpl()->handle, NVVIOCLASS_SDI, NVVIOOWNERTYPE_APPLICATION) != NVAPI_OK)
        throw std::runtime_error("Cannot open SDI output device");

    NVVIOCONFIG config = {0};
    config.version = NVVIOCONFIG_VER;
    config.nvvioConfigType = NVVIOCONFIGTYPE_OUT;
    config.fields = NVVIOCONFIG_SIGNALFORMAT;
    config.vioConfig.outConfig.signalFormat = map_streamformat_to_nvsignalformat(format);
    config.fields |= NVVIOCONFIG_DATAFORMAT;
    // there is only one data format we are using: rgb, which is automatically converted to yuv
    if (streams > 1)
        config.vioConfig.outConfig.dataFormat = NVVIODATAFORMAT_DUAL_R8G8B8_TO_DUAL_YCRCB422;
    else
        config.vioConfig.outConfig.dataFormat = NVVIODATAFORMAT_R8G8B8_TO_YCRCB422;


//  output_config.fields |= NVVIOCONFIG_CSCOVERRIDE;
//  output_config.vioConfig.outConfig.cscOverride = FALSE;
//  output_config.vioConfig.outConfig.gammaCorrection.version = NVVIOGAMMACORRECTION_VER;
//  output_config.vioConfig.outConfig.gammaCorrection.fGammaValueR = 1.0f;
//  output_config.vioConfig.outConfig.gammaCorrection.fGammaValueG = 1.0f;
//  output_config.vioConfig.outConfig.gammaCorrection.fGammaValueB = 1.0f;
//  output_config.fields |= NVVIOCONFIG_GAMMACORRECTION;
//  output_config.fields |= NVVIOCONFIG_FULL_COLOR_RANGE;
//  output_config.vioConfig.outConfig.enableFullColorRange = TRUE;

    // dont know about queue length... 8 fails with error, 2 seems to work fine. dont know about default.
    // sdk says: default is 5, min 2 and max 7.
    config.fields |= NVVIOCONFIG_FLIPQUEUELENGTH;
    config.vioConfig.outConfig.flipQueueLength = 2;

    if (NvAPI_VIO_SetConfig(dev->get_pimpl()->handle, &config) != NVAPI_OK)
        throw std::runtime_error("Cannot configure output parameters");


    HVIDEOOUTPUTDEVICENV video_devices[10];
    // side note: distinction with capture devices is via wglEnumerateVideo*Capture*DevicesNV
    int device_count = wglEnumerateVideoDevicesNV(dc, 0);
    // this should never happen! otherwise driver is confused about what is present in the machine
    if (device_count == 0)
        throw std::runtime_error("Driver is confused about video output devices");

    assert(device_count <= sizeof(video_devices) / sizeof(video_devices[0]));
    if (device_count != wglEnumerateVideoDevicesNV(dc, &video_devices[0]))
        throw std::runtime_error("Driver is confused about video output devices");


    // FIXME: how do i determine whether the wgl device(s) match the nvapi device we want to use here??


    // FIXME: find out how many slots we have: wglQueryCurrentContextNV(WGL_NUM_VIDEO_SLOTS_NV, )
    // FIXME: find out which slot is actually free!
    pimpl->videoslot = 1;
    if (!wglBindVideoDeviceNV(dc, pimpl->videoslot, video_devices[0], 0))
        throw std::runtime_error("Cannot bind video output device");

    // FIXME: we should validate that the current ogl context corresponds to the gpu hosting the sdi card!

    // spec says that wglBindVideoDeviceNV() will enable output straight away
    // so nvapi should know by now that output is running
    // also note at the top of the constructor: this seems to be true all the time, even if nothing is supplied
    if (!NvAPI_VIO_IsRunning(dev->get_pimpl()->handle))
    {
        // FIXME: i want to know when this actually happens
        assert(false);
        // try to enable. no idea if this works or not
        if (NvAPI_VIO_Start(dev->get_pimpl()->handle) != NVAPI_OK)
            throw std::runtime_error("Cannot start video output");
    }



    // ok, so the sdi bits should be working now
    // lets set up a few more opengl thingies for correct rendering
    if (customtextureid1 != 0)
    {
        // FIXME: validate texture dimensions!
        pimpl->isusingcustomtextures = true;
        pimpl->textures.push_back(customtextureid1);
        if (streams == 2)
        {
            assert(customtextureid2 != 0);
            pimpl->textures.push_back(customtextureid2);
        }
    }
    else
    {
        pimpl->isusingcustomtextures = false;
        for (int i = 0; i < streams; ++i)
        {
            GLuint tex = 0;
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, format.get_width(), format.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
            // no mipmapping, for now
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
            if (glGetError() != GL_NO_ERROR)
            {
                // FIXME: figure out how to clean up the other textures later
                glDeleteTextures(1, &tex);
                throw std::runtime_error("Cannot allocate stream storage");
            }

            pimpl->textures.push_back(tex);
            assert(i+1 == pimpl->textures.size());
        }
    }

    // make sure we can store an hours worth of log
    pimpl->framestats.reserve(60 * 60 * 25);
}


void SDIOutput::present(unsigned __int64 fromwhen)
{
    assert(wglGetCurrentContext() == pimpl->oglrc);

    // opengl query objects are a bit of a mess
    // we need to create them on demand because otherwise
    //  it becomes difficult distinguishing between uninit'd timers
    //  and properly finished ones

    if (pimpl->gputimers.first.first == 0)
    {
        assert(pimpl->gputimers.first.second == 0);

        glGenQueries(1, &pimpl->gputimers.first.first);
        glGenQueries(1, &pimpl->gputimers.first.second);
        // may fail... but shouldnt!
        assert(glGetError() == GL_NO_ERROR);
    }

    pimpl->framestats.push_back(FrameStats());
    pimpl->framestats.back().frameno = pimpl->framestats.size() - 1;
    pimpl->framestats.back().gpu_queued_time = get_current_hardware_time();
    GetSystemTime(&pimpl->framestats.back().wallclock_queued_time);


    switch (pimpl->textures.size())
    {
        case 1:
            glPresentFrameKeyedNV(pimpl->videoslot, fromwhen,
                pimpl->gputimers.first.first, // begin present time id
                pimpl->gputimers.first.second, // present duration id
                GL_FRAME_NV, 
                // the extra (zeroed) output images are for interlaced formats
                GL_TEXTURE_2D, pimpl->textures[0], 0,
                GL_NONE, 0, 0);
            break;
        case 2:
            glPresentFrameDualFillNV(pimpl->videoslot, fromwhen,
                pimpl->gputimers.first.first, // begin present time
                pimpl->gputimers.first.second, // present duration
                GL_FRAME_NV, 
                // the extra (zeroed) output images are for interlaced formats
                GL_TEXTURE_2D, pimpl->textures[0], GL_NONE, 0, 
                GL_TEXTURE_2D, pimpl->textures[1], GL_NONE, 0);
            break;

        case 0:
        default:
            assert(false);
    }

    // check if previous frame has finished
    // note: timer will have finished if our frames are actually scanning out
    // they may not scan out however if the present time is totally bogus
    if (pimpl->gputimers.second.first != 0)
    {
        assert(pimpl->gputimers.second.second != 0);
        // if we have timers for the previous frame then it should also have an entry in the log
        assert(pimpl->framestats.size() >= 2);

        glGetQueryObjectui64v(pimpl->gputimers.second.first,  GL_QUERY_RESULT, &(pimpl->framestats[pimpl->framestats.size() - 2].gpu_scanout_time));
        glGetQueryObjectuiv(pimpl->gputimers.second.second, GL_QUERY_RESULT, &(pimpl->framestats[pimpl->framestats.size() - 2].gpu_scanout_count));
        assert(glGetError() == GL_NO_ERROR);
    }

    std::swap(pimpl->gputimers.first, pimpl->gputimers.second);
}

unsigned __int64 SDIOutput::get_current_hardware_time()
{
    assert(wglGetCurrentContext() == pimpl->oglrc);

    GLuint64EXT     time;
    glGetVideoui64vNV(pimpl->videoslot, GL_CURRENT_TIME_NV, &time);
    // FIXME: check for error

    return time;
}


} // namespace
