#include "nalfilegenerator.h"
#include <cstring>
#include <cassert>
#include <iostream>


//------------------------------------------------------------------------------
static int CUDAAPI handle_video_data(void* user_data,
                                     CUVIDSOURCEDATAPACKET* packet)
{
    if (!user_data || !packet) { return 0; }

    NalFileGenerator* nfg = reinterpret_cast<NalFileGenerator*>(user_data);
    if (nfg->parse_video_data(packet)) {
        return 1;
    }
    return 0;
}

//------------------------------------------------------------------------------
static int CUDAAPI handle_video_sequence(void* user_data, CUVIDEOFORMAT* format)
{
    if (!user_data || !format) { return 0; }

    NalFileGenerator* nfg = reinterpret_cast<NalFileGenerator*>(user_data);
    if (nfg->compare_video_sequence(format)) {
        return 1;
    }

    return 0;
}

//------------------------------------------------------------------------------
static int CUDAAPI handle_picture_decode(void* user_data,
    CUVIDPICPARAMS* pic_params)
{
    if (!user_data || !pic_params) { return 0; }

    NalFileGenerator* nfg = reinterpret_cast<NalFileGenerator*>(user_data);
    CUresult r = cuvidDecodePicture(nfg->video_decoder_, pic_params);
    if (r != CUDA_SUCCESS) {
        return 0;
    }
    return 1;
}

//------------------------------------------------------------------------------
static int CUDAAPI handle_picture_display(void* user_data,
    CUVIDPARSERDISPINFO* pic_params)
{
    return 1;
}

//------------------------------------------------------------------------------
NalFileGenerator::NalFileGenerator()
    :video_source_(0), video_parser_(NULL), video_decoder_(NULL)
{}

//------------------------------------------------------------------------------
NalFileGenerator::~NalFileGenerator()
{
    if (video_source_) {
        cuvidDestroyVideoSource(video_source_);
    }

    if (video_decoder_) { cuvidDestroyDecoder(video_decoder_); }

    h264_file_.close();
}

//------------------------------------------------------------------------------
bool NalFileGenerator::init_CUDA(uint32_t deviceID)
{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    typedef HMODULE CUDADRIVER;
#else
    typedef void *CUDADRIVER;
#endif
    CUDADRIVER hHandleDriver = 0;
    CUresult result;
    result = cuInit(0, __CUDA_API_VERSION, hHandleDriver);
    if (result != CUDA_SUCCESS) {
        return false;
    }
    result = cuvidInit(0);
    if (result != CUDA_SUCCESS) {
        return false;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        return false;
    }

    result = cuCtxCreate(&cu_context_, CU_CTX_SCHED_AUTO, device);
    if (result != CUDA_SUCCESS) {
        return false;
    }

    CUcontext cur_ctx;
    result = cuCtxPopCurrent(&cur_ctx);
    if (result != CUDA_SUCCESS) {
        return false;
    }

    result = cuvidCtxLockCreate(&ctx_lock_, cu_context_);
    if (result != CUDA_SUCCESS) {
        return false;
    }

    return true;
}

//------------------------------------------------------------------------------
bool NalFileGenerator::parse_video_data(CUVIDSOURCEDATAPACKET* packet)
{
    CUresult result = cuvidParseVideoData(video_parser_, packet);
    if ((packet->flags & CUVID_PKT_ENDOFSTREAM) || (result != CUDA_SUCCESS))
        return false;
    return true;
}

//------------------------------------------------------------------------------
bool NalFileGenerator::compare_video_sequence(CUVIDEOFORMAT* format)
{
    if ((format->codec != decoder_info_.CodecType) ||
        (format->coded_width != decoder_info_.ulWidth) ||
        (format->coded_height != decoder_info_.ulHeight) ||
        (format->chroma_format != decoder_info_.ChromaFormat))
    {
        return false;
    }
    return true;
}

//------------------------------------------------------------------------------
bool NalFileGenerator::generate_indexes(const std::string &path)
{
    h264_file_.close();
    h264_file_.open(path, std::ios::binary);
    if (!h264_file_.is_open()) {
        return false;
    }

    init_CUDA(0);

    // Initialize the video source
    CUVIDSOURCEPARAMS vs_params;
    memset(&vs_params, 0, sizeof(CUVIDSOURCEPARAMS));
    vs_params.pUserData = this;
    vs_params.pfnVideoDataHandler = handle_video_data;
    vs_params.pfnAudioDataHandler = NULL; // No audio stuff

    CUresult result = cuvidCreateVideoSource(&video_source_,
                                             path.c_str(), &vs_params);

    if (result != CUDA_SUCCESS) { return false; }
    std::memset(&format_, 0, sizeof(format_));
    cuvidGetSourceVideoFormat(video_source_, &format_, 0);

    // Validate video format
    assert(cudaVideoCodec_MPEG1 == format_.codec ||
           cudaVideoCodec_MPEG2 == format_.codec ||
           cudaVideoCodec_MPEG4 == format_.codec ||
           cudaVideoCodec_VC1 == format_.codec ||
           cudaVideoCodec_H264 == format_.codec ||
           cudaVideoCodec_JPEG == format_.codec ||
           cudaVideoCodec_H264_SVC == format_.codec ||
           cudaVideoCodec_H264_MVC == format_.codec ||
           cudaVideoCodec_HEVC == format_.codec ||
           cudaVideoCodec_VP8 == format_.codec ||
           cudaVideoCodec_VP9 == format_.codec ||
           cudaVideoCodec_YUV420 == format_.codec ||
           cudaVideoCodec_YV12 == format_.codec ||
           cudaVideoCodec_NV12 == format_.codec ||
           cudaVideoCodec_YUYV == format_.codec ||
           cudaVideoCodec_UYVY == format_.codec);

    assert(cudaVideoChromaFormat_Monochrome == format_.chroma_format ||
           cudaVideoChromaFormat_420 == format_.chroma_format ||
           cudaVideoChromaFormat_422 == format_.chroma_format ||
           cudaVideoChromaFormat_444 == format_.chroma_format);

    CUVIDDECODECREATEINFO vid_decode_create_info;
    memset(&vid_decode_create_info, 0, sizeof(CUVIDDECODECREATEINFO));
    vid_decode_create_info.CodecType = format_.codec;
    vid_decode_create_info.ulWidth = format_.coded_width;
    vid_decode_create_info.ulHeight = format_.coded_height;

    {
        const int decode_mem = 16 * 1024 * 1024;
        const int frame_size = format_.coded_width * format_.coded_height;
        vid_decode_create_info.ulNumDecodeSurfaces =
                (decode_mem + frame_size + 1) / frame_size;
    }

    vid_decode_create_info.ChromaFormat = format_.chroma_format;
    vid_decode_create_info.OutputFormat = cudaVideoSurfaceFormat_NV12;
    vid_decode_create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    vid_decode_create_info.ulTargetWidth = format_.coded_width;
    vid_decode_create_info.ulTargetHeight = format_.coded_height;

    vid_decode_create_info.display_area.left   = 0;
    vid_decode_create_info.display_area.right = format_.display_area.right;
    vid_decode_create_info.display_area.top    = 0;
    vid_decode_create_info.display_area.bottom = format_.display_area.bottom;

    vid_decode_create_info.ulNumOutputSurfaces = 2;
    vid_decode_create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    vid_decode_create_info.vidLock = ctx_lock_;

    if (video_decoder_) {
        cuvidDestroyDecoder(video_decoder_);
    }

    // Create the decoder
    result = cuvidCreateDecoder(&video_decoder_, &vid_decode_create_info);
    if (result != CUDA_SUCCESS) {
        return false;
    }

    decoder_info_ = vid_decode_create_info;
    // Initialize the video parser
    CUVIDPARSERPARAMS vp_params;
    memset(&vp_params, 0, sizeof(CUVIDPARSERPARAMS));
    vp_params.CodecType = decoder_info_.CodecType;
    vp_params.ulMaxNumDecodeSurfaces = decoder_info_.ulNumDecodeSurfaces;
    vp_params.ulMaxDisplayDelay = 0;
    vp_params.pUserData = this;
    vp_params.pfnSequenceCallback = handle_video_sequence;
    vp_params.pfnDecodePicture = handle_picture_decode;
    vp_params.pfnDisplayPicture = handle_picture_display;

    result = cuvidCreateVideoParser(&video_parser_, &vp_params);

    if (result != CUDA_SUCCESS) {
        return false;
    }

    result = cuvidSetVideoSourceState(video_source_, cudaVideoState_Started);
    if (result != CUDA_SUCCESS) {
        return false;
    }

    while(cuvidGetVideoSourceState(video_source_) == cudaVideoState_Started);

    return true;

}
