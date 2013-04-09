/*=============================================================================

  libvideo: a library for SDI video processing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "stdafx.h"
#include <video/compress.h>


extern "C"
bool rgba2nv12(char* dst, std::size_t dstpitch, cudaArray_t array, int width, int height, int paddedheight);


static std::ostream& operator<<(std::ostream& os, const SYSTEMTIME& s)
{
    os << s.wYear << '_' << s.wMonth << '_' << s.wDay << '_' << s.wHour << '_' << s.wMinute << '_' << s.wSecond << '_' << s.wMilliseconds;
    return os;
}


namespace video
{


static std::string format_error_msg(const std::string& msg, int errorcode)
{
    std::ostringstream  o;
    o << msg << " (Error code: 0x" << std::hex << errorcode << ')';
    return o.str();
}

CompressorFailedException::CompressorFailedException(const std::string& msg, int errorcode)
    : std::runtime_error(format_error_msg(msg, errorcode))
{
}

InteropFailedException::InteropFailedException(const std::string& msg)
    : std::runtime_error(msg)
{
}


static float relative_systime(const SYSTEMTIME& start, const SYSTEMTIME& end)
{
    // slightly annoying process... but this is what msdn recommends doing
    FILETIME    startft, endft;
    SystemTimeToFileTime(&start, &startft);
    SystemTimeToFileTime(&end, &endft);

    ULARGE_INTEGER  s, e;
    s.LowPart = startft.dwLowDateTime;
    s.HighPart = startft.dwHighDateTime;
    e.LowPart = endft.dwLowDateTime;
    e.HighPart = endft.dwHighDateTime;

    return (e.QuadPart - s.QuadPart) / 10000.0f;
}

class CompressorImpl
{
    /** @name Keep some performance stats. */
    //@{
public:
    struct FramePerfStats
    {
        unsigned int        frameno;

        // when the frame was queued to the encoder
        // note: the encoder buffers 5+ frames internally
        SYSTEMTIME          queued;
        // when the frame was finally written to disc
        SYSTEMTIME          finished;

        // low-level cuda processing time to convert from rgba to nv12
        float               formatconversiontime;

        // tsc for trying the video lock
        // we want to know how often we have to wait for the lock
        // note: for these to be meaningful you have to set thread affinity!
        unsigned __int64    trylock_tsc;
        unsigned __int64    gotlock_tsc;

        // 1 = i-frame
        // 2 = p-frame
        // 3 = b-frame
        unsigned char       frametype;

        // frame had to wait because there are no more free slots in the queue
        // writing out to hdd was too slow
        bool                had_to_wait_on_io;

        // there are some limits on how many in-flight async io windows can handle
        // lets see if we ever hit that limit (on my testing on win7 that never happened)
        bool                was_converted_to_synchronous;
    };

    std::vector<FramePerfStats>     frameperstats;

    cudaEvent_t     formatconversion_started;
    cudaEvent_t     formatconversion_finished;
    //@}


    /** @name Encoder callbacks for writing out data. */
    //@{
public:
    HANDLE              outputfile;
    // offset of next nal packet in the output file
    unsigned __int64    outputoffset;
    struct OutputQueue
    {
        unsigned char       buffer[3 * 1024 * 1024];
        OVERLAPPED          overlapped;
    }                   outputqueue[3];
    int                 currentqueueslot;
    // encoder docs suggest that frames are always processed in sequential order
    // we only need this for perf-stats in the bitstream-callbacks, which dont have the frame number on input
    int                 currentframe;

    // accumulated size of the current nal packet
    // the encoder will ask for one or more chunks of storage
    //  and we just add it up
    unsigned int        currentnalsize;

    // a histogram of nal packet sizes supplied by the encoder
    // slots are in kilobyte: 1, 2, 4, 8, 16, 32, 64, ...
    unsigned int        nalsizehistogram[32];

    // called when encoder needs a chunk of memory
    static unsigned char* _stdcall acquirebitstream_callback(int* pBufferSize, void* pUserData)
    {
        CompressorImpl* this_ = (CompressorImpl*) pUserData;

        // simple ring buffer style
        int slot = (this_->currentqueueslot + 1) % (sizeof(this_->outputqueue) / sizeof(this_->outputqueue[0]));

        if ((this_->frameperstats.capacity() - this_->frameperstats.size()) > 1)
        {
            assert(this_->frameperstats.size() >= this_->currentframe);
            assert(this_->frameperstats[this_->currentframe].frameno == this_->currentframe);
        }
        // gotta make sure that this slot has actually finished!
        DWORD slotresult = WaitForSingleObject(this_->outputqueue[slot].overlapped.hEvent, 0);
        if ((this_->frameperstats.capacity() - this_->frameperstats.size()) > 1)
            this_->frameperstats[this_->currentframe].had_to_wait_on_io = (slotresult == WAIT_TIMEOUT);

        slotresult = WaitForSingleObject(this_->outputqueue[slot].overlapped.hEvent, INFINITE);
        // uhm... no idea what to do if wait fails!
        assert(slotresult == WAIT_OBJECT_0);

        this_->currentqueueslot = slot;

        *pBufferSize = sizeof(this_->outputqueue[slot].buffer);
        return &(this_->outputqueue[slot].buffer[0]);
    }

    static void _stdcall releasebitstream_callback(int nBytesInBuffer, unsigned char* cb, void* pUserData)
    {
        CompressorImpl* this_ = (CompressorImpl*) pUserData;

        // add up the size of multiple chunks until endframe_callback()
        this_->currentnalsize += nBytesInBuffer;

        if (this_->outputfile)
        {
            // no idea if the encoder ever hands us back internal pointers!
            assert(cb == &(this_->outputqueue[this_->currentqueueslot].buffer[0]));

            OVERLAPPED*  overlapped = &(this_->outputqueue[this_->currentqueueslot].overlapped);
            overlapped->Offset      = this_->outputoffset & 0xFFFFFFFF;
            overlapped->OffsetHigh  = this_->outputoffset >> 32;

            BOOL e = WriteFile(this_->outputfile, cb, nBytesInBuffer, 0, overlapped);
            DWORD c = GetLastError();
            // if async queueing was successful then we have e==false, c==pending
            // if it got converted to synchronous then we have e==true, c==success
            if ((e == FALSE) && (c == ERROR_IO_PENDING))
            {
                // awesome, nothing to do
            }
            else
            if ((e == TRUE) && (c == ERROR_SUCCESS))
            {
                if ((this_->frameperstats.capacity() - this_->frameperstats.size()) > 1)
                    this_->frameperstats[this_->currentframe].was_converted_to_synchronous = true;
            }
            else
                // anything else means it really did fail
                assert(false);

            this_->outputoffset += nBytesInBuffer;
        }
    }
    
    static void _stdcall beginframe_callback(const NVVE_BeginFrameInfo *pbfi, void *pUserdata)
    {
        CompressorImpl* this_ = (CompressorImpl*) pUserdata;
        this_->currentframe = pbfi->nFrameNumber;

        // starting a new nal packet
        this_->currentnalsize = 0;
    }

    static void _stdcall endframe_callback(const NVVE_EndFrameInfo *pefi, void *pUserdata)
    {
        CompressorImpl* this_ = (CompressorImpl*) pUserdata;

        assert(this_->currentframe == pefi->nFrameNumber);

        if ((this_->frameperstats.capacity() - this_->frameperstats.size()) > 1)
        {
            FramePerfStats&     fps = this_->frameperstats[pefi->nFrameNumber];
            // doing this without sync is a bit dodgy
            // but it works because the vector is never resized/reallocd
            assert(fps.frameno == pefi->nFrameNumber);

            GetSystemTime(&fps.finished);

            // keep track of i and p-frames, just for curiosity
            fps.frametype = pefi->nPicType;
        }

        unsigned int    histindex = std::log(std::max(this_->currentnalsize, 1024u) / 1024.0) / std::log(2.0);
        unsigned int    maxindex = (sizeof(this_->nalsizehistogram) / sizeof(this_->nalsizehistogram[0])) - 1;
        ++(this_->nalsizehistogram[std::min(histindex, maxindex)]);
    }
    //@}

public:
    std::string             filename;
    NVEncoder               encoder;
    NVVE_EncodeFrameParams  frameparams;
    CUdeviceptr             framebuffer;

    CUvideoctxlock          ctxlock;
    CUcontext               cudacontext;
    HGLRC                   oglrc;

    int                     width;
    int                     height;
    int                     paddedheight;

    std::map<int, cudaGraphicsResource*>        gl2cudamap;

    // 0.06 bytes or 0.48 bits per pixel works quite well
    // amounts to around 30 Mb/s for full hd
    static const float      BITRATEESTIMATER_BITSPERPIXEL;


    CompressorImpl(int _width, int _height, int mfps, const std::string& _filename)
        : filename(_filename), encoder(0), framebuffer(0), ctxlock(0), cudacontext(0), oglrc(0), width(_width), height(_height), paddedheight(_height),
            outputfile(INVALID_HANDLE_VALUE), outputoffset(0), currentqueueslot(0),
            formatconversion_started(0), formatconversion_finished(0)
    {
        // FIXME: validate dimensions! has to be at least even!

        // zero-init a few structures so that we can cleanup properly
        std::memset(&outputqueue, 0, sizeof(outputqueue));
        std::memset(&nalsizehistogram, 0, sizeof(nalsizehistogram));

        if (cuCtxGetCurrent(&cudacontext) != CUDA_SUCCESS)
            throw InteropFailedException("Cannot retrieve CUDA context for compression");
        if (cudacontext == 0)
            throw std::logic_error("No current CUDA context");

        CUdevice    dev = -1;
        if (cuCtxGetDevice(&dev) != CUDA_SUCCESS)
            throw InteropFailedException("Cannot retrieve CUDA context information for compression");

        oglrc = wglGetCurrentContext();
        if (oglrc == 0)
            throw std::logic_error("No current OpenGL context");

        // note that zero is a valid device index
        std::vector<int>    cudadevices(10, -1);
        unsigned int        actualcudadevices = 0;
        if (cudaGLGetDevices(&actualcudadevices, &cudadevices[0], cudadevices.size(), cudaGLDeviceListAll) != cudaSuccess)
            throw InteropFailedException("Cannot retrieve OpenGL-CUDA interop information for compression");
        
        bool foundcudadevice = false;
        for (int i = 0; i < actualcudadevices; ++i)
            if (foundcudadevice = (cudadevices[i] == dev))
                break;
        // FIXME: this case, where ogl and cuda are not on the same device has never been tested!
        if (!foundcudadevice)
            throw std::logic_error("OpenGL context is not on current CUDA device");


        outputfile = CreateFileA(filename.c_str(), GENERIC_WRITE, FILE_SHARE_READ, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, 0);
        if (outputfile == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Cannot open output file");

        // from now on we need to undo quite a bit of stuff if something goes wrong
        try
        {
            for (int i = 0; i < (sizeof(outputqueue) / sizeof(outputqueue[0])); ++i)
            {
                // manual-reset = true, initial-state = signaled (because endframe() will wait for it)
                outputqueue[i].overlapped.hEvent = CreateEventA(0, TRUE, TRUE, 0);
                if (outputqueue[i].overlapped.hEvent == 0)
                    throw std::runtime_error("Cannot create I/O event");
            }

            // reserve an hour's worth of log
            frameperstats.reserve(25 * 60 * 60);
            // note: we dont check for error here specifically
            //  because we dont really need this event for anything except perf debugging
            // but we still have to clean them up in the handle further below
            cudaError_t ev = cudaErrorNotSupported;
            ev = cudaEventCreate(&formatconversion_started);
            assert(ev == cudaSuccess);
            ev = cudaEventCreate(&formatconversion_finished);
            assert(ev == cudaSuccess);

            // record some dummy time stamps for initialisation
            if (formatconversion_started)
                cudaEventRecord(formatconversion_started);
            if (formatconversion_finished)
                cudaEventRecord(formatconversion_finished);


            // get a handle on the driver-implemented encoder
            HRESULT hr = NVCreateEncoder(&encoder);
            if (hr != S_OK)
                throw CompressorFailedException("Cannot get handle for encoder interface", hr);

            // we only use avc1, also no other codec is going to be supported on hardware
            hr = NVSetCodec(encoder, NV_CODEC_TYPE_H264); 
            if (hr != S_OK)
                throw CompressorFailedException("Cannot set codec type to h.264", hr);

            // not sure if necessary but lets do it anyway
            hr = NVSetDefaultParam(encoder);
            if (hr != S_OK)
                throw CompressorFailedException("Cannot set default parameters", hr);

            // default = baseline=0xff42, try high=0xff64, main=0x__4d
            // high profile pushes encoding time beyond 150 ms!
            int     profile = 0xff42;
            hr = NVSetParamValue(encoder, NVVE_PROFILE_LEVEL, &profile);
            if (hr != S_OK)
                throw CompressorFailedException("Cannot set encoding profile", hr);

            // cabac is quite expensive!
            int     cabacdisabled = 1;      //default = 0 (enabled)
            hr = NVSetParamValue(encoder, NVVE_DISABLE_CABAC, &cabacdisabled);
            if (hr != S_OK)
                throw CompressorFailedException("Cannot set CABAC", hr);

            // the encoder requires even (not odd) input dimensions!
            // ntsc is a format with an odd height (487)
            // width was previously checked in Compressor constructor
            assert(width % 2 == 0);
            paddedheight = height + height % 2;
            int     inputsize[] = {width, paddedheight};
            hr = NVSetParamValue(encoder, NVVE_IN_SIZE, &inputsize);
            if (hr != S_OK)
                throw CompressorFailedException("Cannot set input size", hr);
            hr = NVSetParamValue(encoder, NVVE_OUT_SIZE, &inputsize);
            if (hr != S_OK)
                throw CompressorFailedException("Cannot set output size", hr);

            int     aspectratio[] = {width, height, ASPECT_RATIO_DAR};
            hr = NVSetParamValue(encoder, NVVE_ASPECT_RATIO, &aspectratio);
            // interval between p-frames. these are b-frames!
            // default = 1, baseline profile supports 1 only (fails further down otherwise)
            // never produces more than 2 b-frames though
            int     pinterval = 1;
            hr = NVSetParamValue(encoder, NVVE_P_INTERVAL, &pinterval);
            if (hr != S_OK)
                throw CompressorFailedException("Cannot set p interval", hr);

            int     deinterlace = DI_OFF;   // default = DI_MEDIAN
            hr = NVSetParamValue(encoder, NVVE_SET_DEINTERLACE, &deinterlace);
            if (hr != S_OK)
                throw CompressorFailedException("Cannot disable deinterlacing", hr);

            // guestimate is: 100kB per frame for 1080p
            // roughly 0.06 bytes per pixel
            // for example: 1920 * 1080 * 0.0625 * 25 * 8 = 26 Mbs
            int     avgbw = BITRATEESTIMATER_BITSPERPIXEL * width * height * ((float) mfps / 1000.0f);
            hr = NVSetParamValue(encoder, NVVE_AVG_BITRATE, &avgbw);
            if (hr != S_OK)
                throw CompressorFailedException("Cannot set average bitrate", hr);
            int     peakbw = 2 * avgbw;
            hr = NVSetParamValue(encoder, NVVE_PEAK_BITRATE, &peakbw);
            if (hr != S_OK)
                throw CompressorFailedException("Cannot set peak bitrate", hr);
            std::clog << "Bit rate for " << width << 'x' << height << '@' << ((float) mfps / 1000.0f) << " Hz "
                      << " (avg, peak): " << avgbw << ", " << peakbw << " bps" << std::endl;
            int     framerate[] = {mfps, 1000};
            hr = NVSetParamValue(encoder, NVVE_FRAME_RATE, &framerate);
            if (hr != S_OK)
                throw CompressorFailedException("Cannot set frame rate", hr);

            // other relevant default parameters are:
            //  NVVE_FIELD_ENC_MODE		= MODE_FRAME
            //  NVVE_RC_TYPE			= RC_VBR
            //
            // profile different values for:
            //  NVVE_DEBLOCK_MODE		default = 1, try 0

            // if fixed gop then encoder always produces a fixed number of p-frames
            int     dynamicgop = 1;     // default 0
            hr = NVSetParamValue(encoder, NVVE_DYNAMIC_GOP, &dynamicgop);
            if (hr != S_OK)
                throw CompressorFailedException("Cannot set dynamic GOP mode", hr);

            // IDR is like a barrier where no other p-frame can ref across
            // so by default every 15th i-frame is an idr? that's quite long...
            // looks like every i-frame is an idr-frame.
            int     idrperiod = 15;     // default 15
            hr = NVSetParamValue(encoder, NVVE_IDR_PERIOD, &idrperiod);
            if (hr != S_OK)
                throw CompressorFailedException("Cannot set IDR period", hr);


            // FIXME: force gpu selection for encoder?
            //        should run on the same gpu that has the input buffer
            //         which would leave copying it from input- to encoder gpu to the user

            // FIXME: profile different offload values!
            int     offloadlevel = NVVE_GPU_OFFLOAD_DEFAULT;//NVVE_GPU_OFFLOAD_ALL;//;
            hr = NVSetParamValue(encoder, NVVE_GPU_OFFLOAD_LEVEL, &offloadlevel);
            // this shouldnt fail normally!
            if (hr != S_OK)
                throw CompressorFailedException("Cannot set GPU offloading", hr);

            if (NVGetHWEncodeCaps() == S_OK)
            {
                // mem copies have to be locked
                if (cuvidCtxLockCreate(&ctxlock, cudacontext) != CUDA_SUCCESS)
                    throw CompressorFailedException("Cannot create compressor lock");
                hr = NVSetParamValue(encoder, NVVE_DEVICE_CTX_LOCK, &ctxlock);
                if (hr != S_OK)
                    throw CompressorFailedException("Cannot set compressor lock", hr);

                int     usedevicemem = 1;
                hr = NVSetParamValue(encoder, NVVE_DEVICE_MEMORY_INPUT, &usedevicemem);
                if (hr != S_OK)
                    throw CompressorFailedException("Cannot tell the compressor to use CUDA memory as input", hr);

                NVVE_CallbackParams cb = {0};
                cb.pfnacquirebitstream = acquirebitstream_callback;
                cb.pfnreleasebitstream = releasebitstream_callback;
                cb.pfnonbeginframe = beginframe_callback;
                cb.pfnonendframe = endframe_callback;
                NVRegisterCB(encoder, cb, this);

                hr = NVCreateHWEncoder(encoder);
                if (hr != S_OK)
                    throw CompressorFailedException("Cannot create hardware encoder", hr);

                // we are encoding from nv12 format
                // this is bi-planar: one plane for full-res luminance (y)
                //  and another plane with interleaved chroma (uv) at 2x2 subsampled res
                //  hence: 1920 pixels per row * 1 byte per pixel * 1080 rows + 1920 / 2 halfres * 2 components * 1 byte each * 1080 / 2 halfres
                //  equals: 2.073.600 + 1.036.800
                std::size_t     framebufferpitch = 0;
                if (cuMemAllocPitch(&framebuffer, &framebufferpitch, width, paddedheight + paddedheight / 2, 16) != CUDA_SUCCESS)
                    throw InteropFailedException("Cannot alloc encoder buffer");
                cuMemsetD2D8(framebuffer, framebufferpitch, 0, width, paddedheight + paddedheight / 2);

                std::memset(&frameparams, 0, sizeof(frameparams));
                frameparams.Height          = paddedheight;
                frameparams.Width           = width;
                frameparams.Pitch           = framebufferpitch;
                frameparams.PictureStruc    = FRAME_PICTURE; 
                frameparams.SurfFmt         = NV12;
                frameparams.progressiveFrame = 1;
                // this would be pointer into cpu-side buffer
                frameparams.picBuf = 0;
            }
        }
        catch (...)
        {
            try_cleanup();
            throw;
        }
    }

    void try_cleanup()
    {
        HRESULT hr = S_OK;
        cudaError_t ev = cudaSuccess;

        if (encoder)
        {
            hr = NVDestroyEncoder(encoder);
            // there isnt anything else we can do?
            if (hr < 0)
                std::cerr << "Failed cleaning up h.264 encoder during exception handler! Leaking memory, I guess." << std::endl;
            encoder = 0;
        }
        if (ctxlock)
        {
            cuvidCtxLockDestroy(ctxlock);
            ctxlock = 0;
        }
        if (framebuffer)
        {
            if (cuMemFree(framebuffer) != CUDA_SUCCESS)
                std::cerr << "Cannot free frame buffer! Leaking memory, I guess." << std::endl;
            framebuffer = 0;
        }
        if (formatconversion_finished)
        {
            ev = cudaEventDestroy(formatconversion_finished);
            assert(ev == cudaSuccess);
            formatconversion_finished = 0;
        }
        if (formatconversion_started)
        {
            ev = cudaEventDestroy(formatconversion_started);
            assert(ev == cudaSuccess);
            formatconversion_started = 0;
        }
        if (outputfile != INVALID_HANDLE_VALUE)
        {
            CloseHandle(outputfile);
            outputfile = INVALID_HANDLE_VALUE;
        }
        for (int i = 0; i < (sizeof(outputqueue) / sizeof(outputqueue[0])); ++i)
        {
            if (outputqueue[i].overlapped.hEvent)
                CloseHandle(outputqueue[i].overlapped.hEvent);
            outputqueue[i].overlapped.hEvent = 0;
        }
    }

    ~CompressorImpl()
    {
        try_cleanup();

        // FIXME: unregister textures??

        // FIXME: this should go somewhere else, really
        std::ofstream   logfile((filename + ".compressorperformance.log").c_str());
        if (logfile)
        {
            char*   frametype[] = {"???", "I", "P", "B", "???"};
            for (int i = 0; i < frameperstats.size(); ++i)
            {
                logfile 
                    << "frameno=" << frameperstats[i].frameno 
                    << ", frametype=" << frametype[std::min((std::size_t) frameperstats[i].frametype, sizeof(frametype) / sizeof(frametype[0]))]
                    << ", queued=" << frameperstats[i].queued 
                    << ", finished=" << frameperstats[i].finished 
                    << ", processtime=" << relative_systime(frameperstats[i].queued, frameperstats[i].finished)
                    << ", had2wait4io=" << frameperstats[i].had_to_wait_on_io
                    << ", formatconversiontime=" << frameperstats[i].formatconversiontime 
                    << ", ioconverted2sync=" << frameperstats[i].was_converted_to_synchronous 
                    << ", trylock_tsc=" << frameperstats[i].trylock_tsc
                    << ", gotlock_tsc=" << frameperstats[i].gotlock_tsc
                    << ", lock_diff=" << (frameperstats[i].gotlock_tsc - frameperstats[i].trylock_tsc)
                    << std::endl;
            }
        }
        logfile.close();

        std::ofstream   nalhistfile((filename + ".nalsizehistogram.log").c_str());
        if (nalhistfile)
        {
            unsigned int    cumulative = 0;
            for (unsigned int i = 0; i < (sizeof(nalsizehistogram) / sizeof(nalsizehistogram[0])); ++i)
            {
                cumulative += nalsizehistogram[i];
                nalhistfile << "smallerthan=" << (1u << i) << ", howoften=" << nalsizehistogram[i] << ", cumulative=" << cumulative << std::endl;
            }
        }
        nalhistfile.close();
    }


    void preparetexture(int gltexture)
    {
        assert(wglGetCurrentContext() == oglrc);

        // lets check that the texture has the right dimensions
        GLuint  cur_tex;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, (GLint*) &cur_tex);
        glBindTexture(GL_TEXTURE_2D, gltexture);
        // FIXME: check format! needs to be compatible with cuda
        
        glBindTexture(GL_TEXTURE_2D, cur_tex);
        if (glGetError() != GL_NO_ERROR)
            throw InteropFailedException("Querying OpenGL texture attributes failed");

        std::map<int, cudaGraphicsResource*>::iterator i = gl2cudamap.find(gltexture);
        if (i == gl2cudamap.end())
        {
            cudaGraphicsResource*   r = 0;
            // note: this function is very picky about format! it will simply crash if it's RGB instead of RGBA, for example
            if (cudaGraphicsGLRegisterImage(&r, gltexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly) != cudaSuccess)
                throw CompressorFailedException("Cannot register OpenGL texture into CUDA for compression");

            gl2cudamap.insert(std::make_pair(gltexture, r));
        }
    }

    void compresstexture(int gltexture)
    {
        assert(wglGetCurrentContext() == oglrc);

        std::map<int, cudaGraphicsResource*>::iterator i = gl2cudamap.find(gltexture);
        if (i == gl2cudamap.end())
            throw std::logic_error("Requested texture has not been prepared for compression");


        // the performance stats thing is for debugging and optimisation only
        //  so if we record for longer that 1h we just ignore later perf timings
        if ((frameperstats.capacity() - frameperstats.size()) > 1)
        {
            if (formatconversion_finished)
            {
                // should have finished long time ago!
                cudaEventSynchronize(formatconversion_finished);
                if (!frameperstats.empty())
                    cudaEventElapsedTime(&(frameperstats.back().formatconversiontime), formatconversion_started, formatconversion_finished);
            }

            frameperstats.push_back(FramePerfStats());
            frameperstats.back().frameno = frameperstats.size() - 1;
            frameperstats.back().formatconversiontime = -1;
            GetSystemTime(&frameperstats.back().queued);
        }

        // we keep a time-stamp to figure out whether there is ever any contention on the lock
        // we dont actually know whether the encoder has an internal ring buffer or anything like it
        // so if an encode is still running while we are trying to queue more, does this block?
        // side note: encoder does buffer input frames internally, around 5 ish or so
        frameperstats.back().trylock_tsc = __rdtsc();

        // i dont know if these vid-locks synchronise with the rest of cuda/opgl
        //  or if they are host side only.
        // if they are not cuda-synchronous than we could risk unlocking the buffer while
        //  the conversion-kernel is still running.
        // to mitigate: lock before mapping and converting the texture. this works
        //  because graphics-interop synchronises on both ogl and cuda
        // FIXME: maybe this is bad for perf?
        if (cuvidCtxLock(ctxlock, 0) != CUDA_SUCCESS)
            throw CompressorFailedException("Cannot lock compressor buffer");

        frameperstats.back().gotlock_tsc = __rdtsc();

        try
        {
            if (cudaGraphicsMapResources(1, &(i->second)) != cudaSuccess)
                throw InteropFailedException("Cannot map OpenGL texture into CUDA for compression");

            try
            {
            //	cudaMipmappedArray_t	mipmaparray;
            //	cudaError_t error = cudaGraphicsResourceGetMappedMipmappedArray(&mipmaparray, i->second);
                cudaArray_t     array;
                cudaError_t error = cudaGraphicsSubResourceGetMappedArray(&array, i->second, 0, 0);
                if (error != cudaSuccess)
                    throw InteropFailedException("Cannot map OpenGL texture into CUDA for compression");

                if (formatconversion_started)
                    cudaEventRecord(formatconversion_started);

                // note: the target-buffer for the conversion might be larger/taller than what we think
                //  the actual video height is. this is fine, the conversion will simply never touch the
                //  bottom bits of it.
                // reason for this is that the encoder seems to have problems with low resolution formats
                //  like ntcs, so we pad the frame at the bottom a bit. (padding is done in constructor)
                rgba2nv12((char*) framebuffer, frameparams.Pitch, array, width, height, paddedheight);

                if (formatconversion_finished)
                    cudaEventRecord(formatconversion_finished);

                // FIXME: ogl-cuda interop synchronises between any ogl ops and cuda ops
                //        but does this mean it syncs to the host too?
                //        in that case we would stall here...
            }
            catch (...)
            {
                if (cudaGraphicsUnmapResources(1, &(i->second)) != cudaSuccess)
                    std::cerr << "Cannot unmap OpenGL texture from compressor during exception handler!" << std::endl;
                throw;
            }

            if (cudaGraphicsUnmapResources(1, &(i->second)) != cudaSuccess)
                throw InteropFailedException("Cannot unmap OpenGL texture from compressor");
        }
        catch (...)
        {
            if (cuvidCtxUnlock(ctxlock, 0) != CUDA_SUCCESS)
                std::cerr << "Cannot unlock compressor buffer during exception handler!" << std::endl;
            throw;
        }

        // can this fail?
        if (cuvidCtxUnlock(ctxlock, 0) != CUDA_SUCCESS)
            throw CompressorFailedException("Cannot unlock compressor buffer");

        HRESULT hr = NVEncodeFrame(encoder, &frameparams, 0, (void*) framebuffer);
        if (hr != S_OK)
            throw CompressorFailedException("Cannot feed frame into compressor", hr);	
    }
};

const float CompressorImpl::BITRATEESTIMATER_BITSPERPIXEL = 0.6f;//0.48f;


Compressor::Compressor(int width, int height, int mfps, const std::string& filename)
    : pimpl(0)
{
    // a more "efficient" test would be to check if the first bit is set
    // but this would just make it harder to read and this codepath is certainly 
    //  not time-critical (all the other stuff takes orders of magnitude longer than this check)
    if (width % 2 != 0)
        throw std::runtime_error("Video width needs to be even");

    pimpl = new CompressorImpl(width, height, mfps, filename);
}

Compressor::~Compressor()
{
    delete pimpl;
}


void Compressor::preparetexture(int gltexture)
{
    pimpl->preparetexture(gltexture);
}

void Compressor::compresstexture(int gltexture)
{
    pimpl->compresstexture(gltexture);
}


} // namespace
