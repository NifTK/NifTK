#include "stdafx.h"
#include "deviceimpl.h"
#include <video/device.h>
#include <video/sdiinput.h>


static std::ostream& operator<<(std::ostream& os, const SYSTEMTIME& s)
{
    os << s.wYear << '_' << s.wMonth << '_' << s.wDay << '_' << s.wHour << '_' << s.wMinute << '_' << s.wSecond << '_' << s.wMilliseconds;
    return os;
}


namespace video
{


struct FrameTime
{
    unsigned int  frameno;
    GLuint64EXT   arrival_time;
    SYSTEMTIME    pickup_time;

    FrameTime(unsigned int _frameno, GLuint64 _arrival_time)
        : frameno(_frameno), arrival_time(_arrival_time)
    {
    }
};


// pimpl so we dont pollute the header with nvapi dependencies
class SDIInputImpl
{
public:
    HGLRC                 oglrc;
    HDC                   dc;
    std::vector<GLuint>   textures;

    int                   num_streams;
    int                   ringbuffer_size;
    int                   current_ringbuffer_index;

    // there is one pbo for each stream
    //  unless capture goes straight into texture
    std::vector<GLuint>   pbos;
    int                   pbo_pitch;      // in bytes

    int                   videoslot;
    HVIDEOINPUTDEVICENV   videodev;

    std::vector<FrameTime>    frametimes;

    int   width;
    int   height;

    SDIInputImpl()
        : oglrc(0), dc(0), videoslot(0), videodev(0), width(0), height(0), pbo_pitch(0), num_streams(0), ringbuffer_size(0), current_ringbuffer_index(0)
    {
    }
};


void SDIInput::set_log_filename(const std::string& fn)
{
    logfilename = fn;
}

int SDIInput::get_texture_id(int streamno) const
{
    assert(wglGetCurrentContext() == pimpl->oglrc);

    if (streamno >= pimpl->num_streams)
        return 0;
    int texindex = streamno * pimpl->ringbuffer_size + pimpl->current_ringbuffer_index;
    return pimpl->textures[texindex];
}

bool SDIInput::has_frame() const
{
    assert(wglGetCurrentContext() == pimpl->oglrc);

    GLint r = GL_FALSE;
    glGetVideoCaptureivNV(pimpl->videoslot, GL_NEXT_VIDEO_CAPTURE_BUFFER_STATUS_NV, &r);
    return r == GL_TRUE;
}

FrameInfo SDIInput::capture()
{
    assert(wglGetCurrentContext() == pimpl->oglrc);

    GLuint      sequence_num = 0;
    GLuint64EXT capture_time = 0;

    GLenum result = glVideoCaptureNV(pimpl->videoslot, &sequence_num, &capture_time);
    if (result == GL_FAILURE_NV)
        throw std::runtime_error("Capture failed");

    pimpl->frametimes.push_back(FrameTime(sequence_num, capture_time));
    GetSystemTime(&(pimpl->frametimes.back().pickup_time));

    // do this at the beginning so that whatever value it has after this method
    //  it will refer to the most recent slot
    pimpl->current_ringbuffer_index = (pimpl->current_ringbuffer_index + 1) % pimpl->ringbuffer_size;

    // there can be one pbo per stream
    // however, if capture goes straight to texture then there are none
    for (int i = 0; i < pimpl->pbos.size(); ++i)
    {
        // if we do have pbos then we should also have at least one texture for each
        assert(pimpl->textures.size() >= pimpl->pbos.size());

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pimpl->pbos[i]);
        glBindTexture(GL_TEXTURE_2D, pimpl->textures[i * pimpl->ringbuffer_size + pimpl->current_ringbuffer_index]);
        // we have 4 bytes per pixel
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, pimpl->pbo_pitch / 4);

        // using the (possibly halfed) height here takes care of field-drop or stack
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pimpl->width, pimpl->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        // FIXME: does this stall?
        glGenerateMipmapEXT(GL_TEXTURE_2D);
    }
    if (!pimpl->pbos.empty())
    {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    if (glGetError() != GL_NO_ERROR)
        throw std::runtime_error("OpenGL returned error during/after texture update");

    FrameInfo fi = {0};
    if (result != GL_SUCCESS_NV)
        return fi;

    // keep the machine running for as long as we are capturing video
    // presumably requesting display prevents our capture gpu from dozing off
    // (note: only works if called periodically, which is what we want here)
    SetThreadExecutionState(ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED);

    fi.sequence_number = sequence_num;
    fi.arrival_time = capture_time;
    return fi;
}

SDIInput::SDIInput(SDIDevice* dev, InterlacedBehaviour interlaced, int ringbuffersize)
    : pimpl(new SDIInputImpl), logfilename("sdicapture.log")
{
    assert(dev != 0);
    assert(dev->get_type() == SDIDevice::INPUT);

    // we use the ogl context for debugging: has to be current every time you call into the class
    pimpl->oglrc = wglGetCurrentContext();
    if (pimpl->oglrc == 0)
        throw std::logic_error("There is no OpenGL context current");
    // we need the dc for setup and teardown, otherwise its fairly boring
    pimpl->dc = wglGetCurrentDC();
    assert(pimpl->dc != 0);

    // should be safe to call this more than once
    if (glewInit() != GLEW_OK)
        throw std::runtime_error("Cannot initialise OpenGL function pointers (via GLEW)");
    // FIXME: check that the necessary extensions are actually present!

    // all streams have to have same format
    // (because of nvapi, which is most likely resulting from hardware limitations)
    StreamFormat	expectedformat = dev->get_format(0);
    if (expectedformat.format == StreamFormat::PF_NONE)
        throw std::runtime_error("Device has no active streams incoming");

    pimpl->width  = expectedformat.get_width();
    pimpl->height = expectedformat.get_height();
    // just in case somebody requested some interlaced mode but we are actually capturing progressive
    if (!expectedformat.is_interlaced)
        interlaced = DO_NOTHING_SPECIAL;

    // find out what is currently connected and coming in
    NVVIOSTATUS status = {0};
    status.version = NVVIOSTATUS_VER;
    if (NvAPI_VIO_Status(dev->get_pimpl()->handle, &status) != NVAPI_OK)
        throw std::runtime_error("Unable to get input device status");

    // and configure capture accordingly
    NVVIOCONFIG config = {0};
    config.version = NVVIOCONFIG_VER;
    config.nvvioConfigType = NVVIOCONFIGTYPE_IN;

    // key is jack, value is list of channels (at the moment, max 2)
    std::map<int, std::vector<int> >	jackchannelconfig;
    for (int i = 0; i < NVAPI_MAX_VIO_JACKS; ++i)
        for (int j = 0; j < NVAPI_MAX_VIO_CHANNELS_PER_JACK; ++j)
            if (status.vioStatus.inStatus.vidIn[i][j].signalFormat != NVVIOSIGNALFORMAT_NONE)
            {
                // this assumes that all streams have the same format
                // which is true, currently
                config.vioConfig.inConfig.signalFormat = status.vioStatus.inStatus.vidIn[i][j].signalFormat;

                jackchannelconfig[i].push_back(j);
            }

    // FIXME: proper dual link streams are untested!
    //        that is, a stream that comes in over two jacks

    config.fields = NVVIOCONFIG_SIGNALFORMAT;
    config.vioConfig.inConfig.numStreams = jackchannelconfig.size();
    config.fields |= NVVIOCONFIG_STREAMS;

    // FIXME: we may want to set this to 1 (default is 5)
    //        so we will notice immediately if we can keep up or not
    config.vioConfig.inConfig.numRawCaptureImages = NVAPI_GVI_DEFAULT_RAW_CAPTURE_IMAGES;


    std::map<int, std::vector<int> >::const_iterator	jci = jackchannelconfig.begin();
    for (unsigned int i = 0; i < jackchannelconfig.size(); ++i)
    {
        // sampling format is set below
        // never had any input that had more than 8 bits
        config.vioConfig.inConfig.streams[i].bitsPerComponent = 8;
        // expand from colour subsampling to full resolution
        config.vioConfig.inConfig.streams[i].expansionEnable = 1;
        // BEWARE: this one is a bit funny
        // for 3g signals this would be a dual-link stream over two channels (tested, works)
        // but for non-3g, i have no idea how to set this up        
        config.vioConfig.inConfig.streams[i].numLinks = jci->second.size();
        for (unsigned int j = 0; j < config.vioConfig.inConfig.streams[i].numLinks; ++j)
        {
            config.vioConfig.inConfig.streams[i].links[j].jack = jci->first;
            config.vioConfig.inConfig.streams[i].links[j].channel = jci->second[j];
        }
        ++jci;
    }

    // the annoying thing with the nvidia driver is that it does not seem to report
    //  the sampling format that comes off the wire
    // so we need to probe them in order
    // FIXME: after further testing, it appears that the current hardware revision can sample only with 422!
    //        the other ones fail with unsupported even if that's what comes off the wire
    //        e.g.: wire=444, config=444 --> fail     wire=444, config=422 --> ok
    NVVIOCOMPONENTSAMPLING	samplings[] = {NVVIOCOMPONENTSAMPLING_4444, NVVIOCOMPONENTSAMPLING_4224, NVVIOCOMPONENTSAMPLING_444, NVVIOCOMPONENTSAMPLING_422};
    bool foundsamplingformat = false;
    for (int s = 0; s < (sizeof(samplings) / sizeof(samplings[0])); ++s)
    {
        for (unsigned int i = 0; i < jackchannelconfig.size(); ++i)
            config.vioConfig.inConfig.streams[i].sampling = samplings[s];

        if (NvAPI_VIO_SetConfig(dev->get_pimpl()->handle, &config) == NVAPI_OK)
        {
            foundsamplingformat = true;
            break;
        }
    }
    if (!foundsamplingformat)
        throw std::runtime_error("Cannot set up stream parameters");


    std::vector<HVIDEOINPUTDEVICENV> video_devices;
    // ask how many there are
    // side note: output devices are enum'd with wglEnumerateVideoDevicesNV
    //  (notice the lack of "capture" in that function name)
    int device_count = wglEnumerateVideoCaptureDevicesNV(pimpl->dc, 0);
    // this should never happen! otherwise driver is confused about what is present in the machine
    if (device_count == 0)
        throw std::runtime_error("Driver is confused about video capture devices");
    video_devices.resize(device_count, 0);

    if (device_count != wglEnumerateVideoCaptureDevicesNV(pimpl->dc, &video_devices[0]))
        throw std::runtime_error("Driver is confused about video capture devices");

    // the device ids in video_devices are not related to what we did earlier with nvapi
    pimpl->videodev = 0;
    for (int i = 0; i < device_count; ++i)
    {
        int   vid;
        if (!wglQueryVideoCaptureDeviceNV(pimpl->dc, video_devices[i], WGL_UNIQUE_ID_NV, &vid))
          throw std::runtime_error("Driver does not know device ID for capture device");

        if (vid == dev->get_pimpl()->id)
            if (wglLockVideoCaptureDeviceNV(pimpl->dc, video_devices[i]))
            {
                pimpl->videodev = video_devices[i];
                break;
            }
            // should this throw an exception on the else branch?
            // is there only one "video device" per nvapi sdi device?
    }
    if (pimpl->videodev == 0)
        throw std::runtime_error("Could not lock capture device");

    // from now on we need to unlock the device
    //  if something goes wrong
    try
    {
        // FIXME: find out how many slots we have: wglQueryCurrentContextNV(WGL_NUM_VIDEO_CAPTURE_SLOTS_NV
        // FIXME: find out which slot is actually free!
        pimpl->videoslot = 1;
        if (!wglBindVideoCaptureDeviceNV(pimpl->videoslot, pimpl->videodev))
            throw std::runtime_error("Cannot bind capture device to slot");

        try
        {
            // this colour conversion stuff is copied from the nv white paper
            // no idea what the point of this is
            GLfloat mat[4][4];
            GLfloat cmax[] = {5000.0f, 5000.0f, 5000.0f, 5000.0f};
            GLfloat cmin[] = {0.0f, 0.0f, 0.0f, 0.0f};
            GLfloat offset[] = {-0.87f, 0.53026f, -1.08f,  0.0f};
            mat[0][0] = +1.164f; mat[0][1] = +1.164f; mat[0][2] = +1.164f; mat[0][3] = +0.000f;
            mat[1][0] = +0.000f; mat[1][1] = -0.392f; mat[1][2] = +2.017f; mat[1][3] = +0.000f;
            mat[2][0] = +1.596f; mat[2][1] = -0.813f; mat[2][2] = +0.000f; mat[2][3] = +0.000f;
            mat[3][0] = +0.000f; mat[3][1] = +0.000f; mat[3][2] = +0.000f; mat[3][3] = +1.000f;

            pimpl->num_streams = jackchannelconfig.size();
            pimpl->ringbuffer_size = std::max(1, ringbuffersize);

            // BEWARE: this here again assumes that we dont have proper dual-link streams that span across jacks!
            for (int i = 0; i < jackchannelconfig.size(); i++)
            {
                glVideoCaptureStreamParameterfvNV(pimpl->videoslot, i, GL_VIDEO_COLOR_CONVERSION_MATRIX_NV, &mat[0][0]);
                glVideoCaptureStreamParameterfvNV(pimpl->videoslot, i, GL_VIDEO_COLOR_CONVERSION_MAX_NV,    &cmax[0]);
                glVideoCaptureStreamParameterfvNV(pimpl->videoslot, i, GL_VIDEO_COLOR_CONVERSION_MIN_NV,    &cmin[0]);
                glVideoCaptureStreamParameterfvNV(pimpl->videoslot, i, GL_VIDEO_COLOR_CONVERSION_OFFSET_NV, &offset[0]);
                int orientation = GL_LOWER_LEFT;    // this is default and this is how it should be no matter what buffer type is used
                glVideoCaptureStreamParameterivNV(pimpl->videoslot, i, GL_VIDEO_CAPTURE_SURFACE_ORIGIN_NV,  &orientation);

                // we assume it's the same for all channels
                // the previous nvapi acrobatics demand this
                int   capture_width = 0;
                int   capture_height = 0;
                glGetVideoCaptureStreamivNV(pimpl->videoslot, i, GL_VIDEO_CAPTURE_FRAME_WIDTH_NV,  (GLint*) &capture_width);
                glGetVideoCaptureStreamivNV(pimpl->videoslot, i, GL_VIDEO_CAPTURE_FRAME_HEIGHT_NV, (GLint*) &capture_height);

                if ((expectedformat.get_width()  != capture_width) ||
                    (expectedformat.get_height() != capture_height))
                    throw std::runtime_error("Driver reports inconsistent video dimensions");

                // something failed along the way
                if (glGetError() != GL_NO_ERROR)
                    throw std::runtime_error("Cannot set up stream parameters");

                // note: if format is really progressive then the variable had been reset at the top of the constructor
                if (interlaced == DROP_ONE_FIELD)
                {
                    // ntsc is one of the formats with an odd height
                    //  for which the top field has 244 lines and the bottom field 243
                    // so we are rounding towards the top field size here
                    capture_height = (capture_height + 1) / 2;
                    pimpl->height = capture_height;
                }

                int   stream_tex_index = pimpl->textures.size();
                for (int j = 0; j < pimpl->ringbuffer_size; ++j)
                {
                    // no matter what, we'll always have at least one texture per stream
                    GLuint tex = 0;
                    glGenTextures(1, &tex);
                    if (glGetError() != GL_NO_ERROR)
                        throw std::runtime_error("Cannot allocate stream storage");
                    assert(tex != 0);
                    pimpl->textures.push_back(tex);

                    glBindTexture(GL_TEXTURE_2D, tex);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
                    // we do mipmapping
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
                    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 2.0f);

                    // better safe than sorry: pre-allocate levels
                    // while it seems to work without i'm not sure if that's correct. spec doesnt say anything
                    //  but i saw an unrelated(?) bluescreen during testing
                    // (derived from libogltools)
                    {
                        int   d[2] = {capture_width, capture_height};

                        bool  all_min_size = true;
                        for (unsigned int l = 0; ; ++l)
                        {
                            // not sure what happens if filter mode is not mipmap
                            glTexImage2D(GL_TEXTURE_2D, l, GL_RGBA8, d[0], d[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
                            if (glGetError() != GL_NO_ERROR)
                                throw std::runtime_error("Cannot allocate stream storage");

                            all_min_size = true;
                            for (unsigned int k = 0; k < 2; ++k)
                            {
                                // spec says round down
                                d[k] = d[k] / 2;
                                all_min_size &= (d[k] <= 0);
                                d[k] = std::max(1, d[k]);
                            }

                            if (all_min_size)
                                break;
                        }
                    }
                    glGenerateMipmapEXT(GL_TEXTURE_2D);
                    if (glGetError() != GL_NO_ERROR)
                        throw std::runtime_error("Cannot allocate stream storage");
                }

                // bind texture object directly only if we dont have to use pbos
                if ((interlaced == DO_NOTHING_SPECIAL) && (ringbuffersize == 0))
                {
                    glBindVideoCaptureStreamTextureNV(pimpl->videoslot, i, GL_FRAME_NV, GL_TEXTURE_2D, pimpl->textures[stream_tex_index]);
                    if (glGetError() != GL_NO_ERROR)
                        throw std::runtime_error("Cannot bind stream texture");
                }
                else
                {
                    // user requested a ringbuffer style frame storage
                    //  so we need to bounce everything through pbos first
                    //  (irrespective of any interlaced-mode)
                    // transfer to texture happens during capture()

                    // we have a single pbo for each streams stacked on top of another
                    // FIXME: would it make sense to have a single pbo for all streams?
                    //        that would make binding during capture simpler
                    GLuint pbo;
                    glGenBuffers(1, &pbo);
                    if (glGetError() != GL_NO_ERROR)
                        throw std::runtime_error("Cannot allocate stream storage");

                    pimpl->pbos.push_back(pbo);
                    glBindBuffer(GL_VIDEO_BUFFER_NV, pbo);

                    int internalformat = GL_RGBA8;
                    glVideoCaptureStreamParameterivNV(pimpl->videoslot, i, GL_VIDEO_BUFFER_INTERNAL_FORMAT_NV, &internalformat);

                    // we dump both fields into the same pbo, stacked on top of each other
                    // and during capture time we decide whether we drop one or not
                    int     bufferpitch = 0;
                    glGetVideoCaptureStreamivNV(pimpl->videoslot, i, GL_VIDEO_BUFFER_PITCH_NV, &bufferpitch);

                    if (interlaced == DO_NOTHING_SPECIAL)
                    {
                        int     bufferbytes = bufferpitch * capture_height;
                        glBufferData(GL_VIDEO_BUFFER_NV, bufferbytes, 0, GL_STREAM_READ);
                        glBindVideoCaptureStreamBufferNV(pimpl->videoslot, i, GL_FRAME_NV, 0);
                    }
                    else
                    {
                        int     fieldheight[2];
                        glGetVideoCaptureStreamivNV(pimpl->videoslot, i, GL_VIDEO_CAPTURE_FIELD_UPPER_HEIGHT_NV, &fieldheight[0]);
                        glGetVideoCaptureStreamivNV(pimpl->videoslot, i, GL_VIDEO_CAPTURE_FIELD_LOWER_HEIGHT_NV, &fieldheight[1]);
                        int     bufferbytes = bufferpitch * (fieldheight[0] + fieldheight[1]);
                        glBufferData(GL_VIDEO_BUFFER_NV, bufferbytes, 0, GL_STREAM_READ);

                        // docs say we need storage for both fields, even if we are interested in only one
                        // note that capture() will take care of dropping a field, if necessary
                        glBindVideoCaptureStreamBufferNV(pimpl->videoslot, i, GL_FIELD_UPPER_NV, 0);
                        glBindVideoCaptureStreamBufferNV(pimpl->videoslot, i, GL_FIELD_LOWER_NV, bufferpitch * fieldheight[0]);
                    }
                    if (glGetError() != GL_NO_ERROR)
                        throw std::runtime_error("Cannot bind stream buffer");

                    pimpl->pbo_pitch = bufferpitch;
                }

                glBindTexture(GL_TEXTURE_2D, 0);
            }

            // make sure we can store an hours worth of timestamps
            pimpl->frametimes.reserve(25 * 60 * 60);

            glBeginVideoCaptureNV(pimpl->videoslot);
            if (glGetError() != GL_NO_ERROR)
                throw std::runtime_error("Cannot begin video capture");
        }
        catch (...)
        {
            // before cleaning up textures, unbind video stuff
            //  we dont really know (or care) in what state we left off but lets assume its all borked
            //  so stop video bits before cleaning out its storage
            if (!wglBindVideoCaptureDeviceNV(pimpl->videoslot, 0))
                std::cerr << "Warning: cannot unbind capture device from video slot! Subsequent operations might fail." << std::endl;

            if (!pimpl->textures.empty())
            {
                assert(std::numeric_limits<GLsizei>::max() > pimpl->textures.size());
                glDeleteTextures((GLsizei) pimpl->textures.size(), &(pimpl->textures[0]));
                if (glGetError() != GL_NO_ERROR)
                    std::cerr << "Warning: Failed cleaning up video textures! Leaking memory I guess..." << std::endl;
                pimpl->textures.clear();
            }

            if (!pimpl->pbos.empty())
            {
                glDeleteBuffers((GLsizei) pimpl->pbos.size(), &(pimpl->pbos[0]));
                if (glGetError() != GL_NO_ERROR)
                    std::cerr << "Warning: Failed cleaning up video buffers! Leaking memory I guess..." << std::endl;
                pimpl->pbos.clear();
            }

            throw;
        }
    }
    catch (...)
    {
        if (!wglReleaseVideoCaptureDeviceNV(pimpl->dc, pimpl->videodev))
            std::cerr << "Warning: cannot release capture lock! Subsequent operations might fail." << std::endl;
        throw;
    }
}

SDIInput::~SDIInput()
{
    if (pimpl)
    {
        assert(wglGetCurrentContext() == pimpl->oglrc);

        glEndVideoCaptureNV(pimpl->videoslot);
        if (!wglBindVideoCaptureDeviceNV(pimpl->videoslot, 0))
            std::cerr << "Warning: cannot unbind capture device from video slot! Subsequent operations might fail." << std::endl;
        if (!wglReleaseVideoCaptureDeviceNV(pimpl->dc, pimpl->videodev))
            std::cerr << "Warning: cannot release capture lock! Subsequent operations might fail." << std::endl;

        // if we dont have textures then constructor should have never succeeded 
        assert(!pimpl->textures.empty());
        // either way, better guard against it
        if (!pimpl->textures.empty())
        {
            assert(std::numeric_limits<GLsizei>::max() > pimpl->textures.size());
            glDeleteTextures((GLsizei) pimpl->textures.size(), &(pimpl->textures[0]));
            if (glGetError() != GL_NO_ERROR)
                std::cerr << "Warning: cannot free video textures! Leaking memory." << std::endl;
        }

        if (!pimpl->pbos.empty())
        {
            glDeleteBuffers((GLsizei) pimpl->pbos.size(), &(pimpl->pbos[0]));
            if (glGetError() != GL_NO_ERROR)
                std::cerr << "Warning: Failed cleaning up video buffers! Leaking memory I guess..." << std::endl;
        }

        std::ofstream   logfile(logfilename);
        for (int i = 0; i < pimpl->frametimes.size(); ++i)
        {
            logfile << "frameno=" << pimpl->frametimes[i].frameno << ", arrival_gpu_time=" << pimpl->frametimes[i].arrival_time << ", pickup_wall_time=" << pimpl->frametimes[i].pickup_time << std::endl;
        }

        delete pimpl;
    }
}

int SDIInput::get_width() const
{
    return pimpl->width;
}

int SDIInput::get_height() const
{
    return pimpl->height;
}


} // namespace
