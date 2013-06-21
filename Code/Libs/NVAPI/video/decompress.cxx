/*=============================================================================

  libvideo: a library for SDI video processing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "stdafx.h"
#include <video/decompress.h>
#include <utility>
#include <vector>
#include <fstream>
#include "conversion.h"


namespace video
{


// copied from compress.cxx
static std::string format_error_msg(const std::string& msg, int errorcode)
{
    std::ostringstream  o;
    o << msg << " (Error code: 0x" << std::hex << errorcode << ')';
    return o.str();
}

DecompressorFailedException::DecompressorFailedException(const std::string& msg, int errorcode)
    : std::runtime_error(format_error_msg(msg, errorcode))
{
}


class DecompressorImpl
{
protected:
    std::vector<std::pair<unsigned __int64, FrameType::FT> >   index;

    std::ifstream                 nalfile;
    std::vector<unsigned char>    packetbuffer;

    /** @name Bits for the NVIDIA decoder etc. */
    //@{
protected:
    CUvideosource     nvsource;
    CUVIDEOFORMAT     format;

    CUvideoparser     nvparser;

    CUvideodecoder    nvdecoder;

    unsigned int      session_frame_count;
    unsigned int      requested_frame;
    bool              done_requested_frame;

    // if we are decoding into a texture then this will be non-zero.
    GLuint            target_gl_texture;
    // if we are decoding into a cpu buffer then above is zero and these will have values.
    void*             target_cpu_buffer;
    unsigned int      target_cpu_pitch;
    // the decoded nv12 ouput will always go to our own internal buffer first.
    // only the converted rgba result will be written to the target_cpu_buffer.


    // called by video source
    static int CUDAAPI HandleVideoData(void* pUserData, CUVIDSOURCEDATAPACKET* pPacket)
    {
        // we dont use nvidia's file reader for actual decoding!
        // only for format parsing etc.
        assert(false);
        //CUresult oResult = cuvidParseVideoData(pVideoSourceData->hVideoParser, pPacket);
        return 0;
    }

    // called by video parser
    static int CUDAAPI HandleVideoSequence(void* pUserData, CUVIDEOFORMAT* pFormat)
    {
        DecompressorImpl*   this_ = (DecompressorImpl*) pUserData;

        // FIXME: is this callback called whenever we flush/reset the stream?
        this_->session_frame_count = 0;

        // does it match what we are expecting?
        bool  wrongformat = false;
        wrongformat |= pFormat->codec != this_->format.codec;
        wrongformat |= pFormat->coded_width != this_->format.coded_width;
        wrongformat |= pFormat->coded_height != this_->format.coded_height;
        wrongformat |= pFormat->chroma_format != this_->format.chroma_format;

        return wrongformat ? 0 : 1;
    }

    // called by video parser
    static int CUDAAPI HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams)
    {
        DecompressorImpl*   this_ = (DecompressorImpl*) pUserData;

        CUresult r = cuvidDecodePicture(this_->nvdecoder, pPicParams);
        if (r != CUDA_SUCCESS)
        {
            std::cerr << "cuvidDecodePicture() failed with 0x" << std::hex << r << std::endl;
            // stop further processing
            return 0;
        }
        return 1;
    }

    // called by video parser whenever it thinks that the decoder should be ready.
    static int CUDAAPI HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pPicParams)
    {
        DecompressorImpl*   this_ = (DecompressorImpl*) pUserData;

        // beware: pPicParams->picture_index is an index into a frame queue, it's not a global frame id!
        if (this_->session_frame_count == this_->requested_frame)
        {
            // map it only once
            if (!this_->done_requested_frame)
            {
                // decoder will tell us these
                CUdeviceptr       nv12buffer  = 0;
                unsigned int      bufferpitch = 0;
                CUVIDPROCPARAMS   params = {0};
                // we dont encode interlaced
                assert(pPicParams->progressive_frame);
                params.progressive_frame = pPicParams->progressive_frame;
                CUresult r = cuvidMapVideoFrame(this_->nvdecoder, pPicParams->picture_index, &nv12buffer, &bufferpitch, &params);
                if (r == CUDA_SUCCESS)
                {
                    assert(0 != nv12buffer);
                    assert(0 != bufferpitch);

                    CUcontext   ctx = 0;
                    cuCtxGetCurrent(&ctx);

                    // there is no way to map an arbitrary cuda pointer into cpu address space.
                    // i.e. nothing similar to ogl buffer mapping.
                    // we'd need to use special allocation bits (which ogl is prob using behind the scenes).
                    unsigned int  bufsize = this_->format.coded_height * bufferpitch * 3 / 2;
                    void*  buf = new char[bufsize];
                    r = cuMemcpyDtoH(buf, nv12buffer, bufsize);
                    assert(r == CUDA_SUCCESS);
                    r = cuvidUnmapVideoFrame(this_->nvdecoder, nv12buffer);
                    // no idea what to do if that fails...
                    assert(r == CUDA_SUCCESS);

                    // FIXME: index properly when display area does not start at (0,0)
                    convert_nv12_to_rgba(buf, bufferpitch, this_->format.coded_height, this_->target_cpu_buffer, this_->target_cpu_pitch, this_->get_width(), this_->get_height());
                    delete buf;

                    this_->done_requested_frame = true;
                }
            }

            return 0;
        }

        this_->session_frame_count++;

        // does the return value matter? seems to be ignored...
        return 1;
    }
    //@}


public:
    DecompressorImpl(const std::string& filename)
        : nalfile(filename.c_str(), std::ios::binary), nvsource(0), nvparser(0), nvdecoder(0)
    {
        // make sure the file name is actually valid and we can open it
        if (!nalfile.is_open())
            throw std::runtime_error("Cannot open NAL file " + filename);

        std::memset(&format, 0, sizeof(format));

        // use nvidia's file parser to figure out what format we have.
        // we should be able to get this from the header itself.
        CUVIDSOURCEPARAMS sourceparams = {0};
        // we dont ever expect a callback! but fill this in anyway
        sourceparams.pUserData = this;
        sourceparams.pfnVideoDataHandler = HandleVideoData;
        sourceparams.pfnAudioDataHandler = 0;
        CUresult r = cuvidCreateVideoSource(&nvsource, filename.c_str(), &sourceparams);
        if (r != CUDA_SUCCESS)
            throw DecompressorFailedException("Stream parsing for input file failed", r);

        try
        {
            r = cuvidGetSourceVideoFormat(nvsource, &format, 0);
            if (r != CUDA_SUCCESS)
                throw DecompressorFailedException("Cannot get format from file", r);
        }
        catch (...)
        {
            r = cuvidDestroyVideoSource(nvsource);
            assert(r == CUDA_SUCCESS);
            throw;
        }

        // we are done with the nvidia file parser
        r = cuvidDestroyVideoSource(nvsource);
        assert(r == CUDA_SUCCESS);
        nvsource = 0;
    }

    ~DecompressorImpl()
    {
        if (nvsource)
        {
            CUresult r = cuvidDestroyVideoSource(nvsource);
            if (r != CUDA_SUCCESS)
                std::cerr << "Cleaning up decoder video source failed! Will leak memory." << std::endl;
        }

        if (nvparser)
        {
            CUresult r = cuvidDestroyVideoParser(nvparser);
            if (r != CUDA_SUCCESS)
                std::cerr << "Cleaning up decoder video parser failed! Will leak memory." << std::endl;
        }

        if (nvdecoder)
        {
            CUresult r = cuvidDestroyDecoder(nvdecoder);
            if (r != CUDA_SUCCESS)
                std::cerr << "Cleaning up decoder failed! Will leak memory." << std::endl;
        }

        nalfile.close();
    }

    // returns display dimensions (instead of coded dimensions)
    int get_width() const
    {
        int   w = format.display_area.right - format.display_area.left;
        assert(w >= 0);
        assert((int) format.coded_width >= w);
        return w;
    }

    int get_height() const
    {
        int   h = format.display_area.bottom - format.display_area.top;
        assert(h >= 0);
        assert((int) format.coded_height >= h);
        return h;
    }


    void submit_frame(unsigned int frameno, unsigned int iframebase)
    {
        CUVIDSOURCEDATAPACKET   packet = {0};
        // could happen that a framenumber is passed in that is too large.
        // if that happens we only submit an empty end-of-stream packet.
        if (frameno < index.size())
        {
            unsigned __int64  offset      = index[frameno].first;
            unsigned int      packetsize  = 0;
            // compute size of packet based on where the next one starts
            if ((frameno+1) < index.size())
            {
                unsigned __int64  nextoffset = index[frameno + 1].first;
                // we assume sequential frames.
                // i.e. no b-frames, or arbitrary ordering
                assert(nextoffset > offset);
                // packets
                assert((nextoffset - offset) < std::numeric_limits<unsigned int>::max());

                packetsize = (unsigned int) (nextoffset - offset);
            }
            else
            {
                // the last packet is the rest of the file.
                nalfile.seekg(0, std::ios::end);

                offset = index[frameno].first;
                packetsize = (unsigned int) ((unsigned __int64) nalfile.tellg() - offset);
            }

            packetbuffer.resize(std::max((std::size_t) packetsize, packetbuffer.size()));

            char* ptr = (char*) &packetbuffer[0];
            nalfile.seekg(offset);
            nalfile.read(ptr, packetsize);

            packet.payload = &packetbuffer[0];
            packet.payload_size = packetsize;
        }
        else
            packet.flags |= CUVID_PKT_ENDOFSTREAM;

        if ((frameno - iframebase) > requested_frame)
            packet.flags |= CUVID_PKT_ENDOFSTREAM;
        CUresult r = cuvidParseVideoData(nvparser, &packet);
        if (r != CUDA_SUCCESS)
            // FIXME: throw or return false
            throw DecompressorFailedException("Cannot queue next packet for parser", r);
    }

    bool decompress_internal(unsigned int frameno)
    {
        if (target_gl_texture == 0)
        {
            assert(target_cpu_buffer != 0);
            assert(target_cpu_pitch   > 0);
        }

        if (frameno >= index.size())
            return false;

        // find the most recent i frame
        unsigned int    iframeno = frameno + 1;
        do
        {
            if (index[iframeno - 1].second == FrameType::I)
                break;
            --iframeno;
        }
        while (iframeno > 0);
        iframeno--;
        assert(iframeno <= index.size());

        // we are submitting from an i-frame onwards. so we only count relative to it.
        requested_frame = frameno - iframeno;
        session_frame_count = 0;

        // FIXME: we should figure out whether we can recycle the current instances.
        if (nvparser)
        {
            CUresult r = cuvidDestroyVideoParser(nvparser);
            assert(r == CUDA_SUCCESS);
            nvparser = 0;
        }
        if (nvdecoder)
        {
            CUresult r = cuvidDestroyDecoder(nvdecoder);
            assert(r == CUDA_SUCCESS);
            nvdecoder = 0;
        }


        CUVIDPARSERPARAMS parserparams = {(cudaVideoCodec) 0};
        parserparams.CodecType              = format.codec;
        parserparams.ulMaxNumDecodeSurfaces = 15;   // our idr period is 15 (unrelated, but might as well)
        parserparams.ulMaxDisplayDelay      = 0;
        parserparams.pUserData              = this;
        parserparams.pfnSequenceCallback    = HandleVideoSequence;
        parserparams.pfnDecodePicture       = HandlePictureDecode;
        parserparams.pfnDisplayPicture      = HandlePictureDisplay;
        CUresult r = cuvidCreateVideoParser(&nvparser, &parserparams);
        if (r != CUDA_SUCCESS)
            throw DecompressorFailedException("Cannot create stream parser", r);

        try
        {
            CUVIDDECODECREATEINFO   decoderparams = {0};
            // unscaled input and output
            decoderparams.ulWidth             = format.coded_width;
            decoderparams.ulHeight            = format.coded_height;
            decoderparams.ulTargetWidth       = format.coded_width;
            decoderparams.ulTargetHeight      = format.coded_height;
            decoderparams.CodecType           = format.codec;
            decoderparams.ulNumDecodeSurfaces = 15;     // our idr period is 15 (unrelated, but might as well)
            decoderparams.ChromaFormat        = format.chroma_format;
            decoderparams.OutputFormat        = cudaVideoSurfaceFormat_NV12;
            decoderparams.DeinterlaceMode     = cudaVideoDeinterlaceMode_Weave;   // we dont write out interlaced video
            decoderparams.ulNumOutputSurfaces = 1;
            // use hardware decoder (non-cuda)
            decoderparams.ulCreationFlags     = cudaVideoCreate_PreferCUVID;
            decoderparams.vidLock             = 0;
            r = cuvidCreateDecoder(&nvdecoder, &decoderparams);
            if (r != CUDA_SUCCESS)
                throw DecompressorFailedException("Cannot create decoder", r);

            try
            {
                // we submit stuff until the decoder has finished with our requested frame.
                // this sucks for performance, lots of wasted cycles. but the nv decoder is
                // more suitable for continuous playback instead of picking out a few frames.
                done_requested_frame = false;
                for (unsigned int i = iframeno; i < frameno + 2; ++i)
                {
                    submit_frame(i, iframeno);
                    if (done_requested_frame)
                        break;
                }
            }
            catch (...)
            {
                r = cuvidDestroyDecoder(nvdecoder);
                assert(r == CUDA_SUCCESS);
                nvdecoder = 0;
                throw;
            }
        }
        catch (...)
        {
            r = cuvidDestroyVideoParser(nvparser);
            assert(r == CUDA_SUCCESS);
            nvparser = 0;
            throw;
        }


        return done_requested_frame;
    }

    bool decompress(unsigned int frameno, void* buffer, std::size_t buffersize, unsigned int bufferpitch)
    {
        // validate output buffer size.
        std::size_t   estimatedbuffersize = get_width() * get_height() * 4;
        if (buffersize < estimatedbuffersize)
            throw std::runtime_error("Buffer size is too small");

        if (bufferpitch < ((unsigned int) get_width() * 4))
            throw std::runtime_error("Buffer pitch is too small");

        target_gl_texture = 0;
        target_cpu_buffer = buffer;
        target_cpu_pitch  = bufferpitch;
        return decompress_internal(frameno);
    }

/*
    bool decompress(unsigned int frameno, GLuint targettexture)
    {
        // it's a mistake to pass in an invalid texture id
        if (targettexture == 0)
            throw std::runtime_error("Invalid texture ID passed into decompress()");

        target_gl_texture = targettexture;
        target_cpu_buffer = 0;
        return decompress_internal(frameno);
    }
*/

    void update_index(unsigned int frameno, unsigned __int64 offset, FrameType::FT type)
    {
        if (frameno >= index.size())
            index.resize(frameno + 1);
        index[frameno] = std::make_pair(offset, type);
    }
};


Decompressor::Decompressor(const std::string& _filename)
    : pimpl(new DecompressorImpl(_filename)), filename(_filename)
{
}

Decompressor::~Decompressor()
{
    delete pimpl;
}


void Decompressor::update_index(unsigned int frameno, unsigned __int64 offset, FrameType::FT type)
{
    pimpl->update_index(frameno, offset, type);
}

bool Decompressor::recover_index()
{
    // should this be pimpl specific?

    // naive scanning: just map the whole file and search for suitable start codes
    HANDLE  file = CreateFile(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, 0);
    if (file == INVALID_HANDLE_VALUE)
        // not being able to read the file is a hard error
        throw std::runtime_error("Cannot open/read NAL file " + filename);

    LARGE_INTEGER   size = {0};
    GetFileSizeEx(file, &size);

    HANDLE  mapping = CreateFileMapping(file, 0, PAGE_READONLY, size.HighPart, size.LowPart, 0);
    if (mapping == 0)
    {
        CloseHandle(file);
        throw std::runtime_error("Cannot map NAL file " + filename);
    }

    const char* buffer = (char*) MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (buffer == 0)
    {
        CloseHandle(mapping);
        CloseHandle(file);
        throw std::runtime_error("Cannot read NAL file " + filename);
    }


    static const char iframestartseq[] = {0x00, 0x00, 0x00, 0x01, 0x09, 0x10, 0x00, 0x00, 0x00, 0x01, 0x67};
    static const char pframestartseq[] = {0x00, 0x00, 0x00, 0x01, 0x09, 0x30, 0x00, 0x00, 0x00, 0x01, 0x06};

    unsigned int    currentframe = 0;
    const char*     endptr = buffer + size.QuadPart - sizeof(iframestartseq);
    for (const char* ptr = buffer; ptr < endptr; ++ptr)
    {
        bool foundiframe = std::memcmp(ptr, &iframestartseq[0], sizeof(iframestartseq)) == 0;
        if (foundiframe)
        {
            update_index(currentframe, ptr - buffer, video::FrameType::I);
            ptr += sizeof(iframestartseq) - 1;
            ++currentframe;
        }
        else
        {
            bool foundpframe = std::memcmp(ptr, &pframestartseq[0], sizeof(pframestartseq)) == 0;
            if (foundpframe)
            {
                update_index(currentframe, ptr - buffer, video::FrameType::P);
                ptr += sizeof(pframestartseq) - 1;
                ++currentframe;
            }
        }
    }

    UnmapViewOfFile(buffer);
    CloseHandle(mapping);
    CloseHandle(file);
    return true;
}

/*
bool Decompressor::decompress(unsigned int frameno, GLuint targettexture)
{
    return pimpl->decompress(frameno, targettexture);
}
*/

bool Decompressor::decompress(unsigned int frameno, void* buffer, std::size_t buffersize, unsigned int bufferpitch)
{
    return pimpl->decompress(frameno, buffer, buffersize, bufferpitch);
}


int Decompressor::get_width() const
{
    return pimpl->get_width();
}

int Decompressor::get_height() const
{
    return pimpl->get_height();
}



} // namespace
