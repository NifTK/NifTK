#include "decoder.h"
#include "cudaprocessframe.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cassert>
#include <iostream>

#ifdef NV12_2_ARGB_PTX_DIR
    #define PTX_DIR NV12_2_ARGB_PTX_DIR
#endif

const int MAX_SIZE = 20;

//------------------------------------------------------------------------------
void Decoder::cuda_process_frame(CUdeviceptr * decoded_frame,
    size_t decoded_pitch, CUdeviceptr * dest_data, size_t dest_pitch)

{
    uint32 w = format_.coded_width;
    uint32 h = format_.coded_height;

    // Upload the Color Space Conversion Matrices
    // CCIR 601/709
    float color_space_mat[9];
    setColorSpaceMatrix(color_space_, color_space_mat, hue_);
    updateConstantMemory_drvapi(cuda_module_->getModule(), color_space_mat);
    cudaLaunchNV12toARGBDrv(*decoded_frame, decoded_pitch,
        *dest_data, dest_pitch, w, h, kernelNV12toARGB_, 0);
}

//------------------------------------------------------------------------------
void BGRA_2_RGBA(unsigned char * data, size_t width, size_t height,
                 size_t pitch_in_bytes)
{
    for (size_t y = 0; y < height; ++y) {
        unsigned int * dstptr = (unsigned int*)(&data[y * pitch_in_bytes]);
        unsigned char * current = &data[y * pitch_in_bytes];
        for (size_t x = 0; x < width * 4; x += 4) {
            unsigned char b = current[x + 0];
            unsigned char g = current[x + 1];
            unsigned char r = current[x + 2];
            unsigned char a = current[x + 3];

            *dstptr = (((int)r) << 0) | (((int)g) << 8) |
                (((int)b) << 16) | 0xFF000000;
            ++dstptr;
        }
    }
}

void BGRA_2_ARGB(unsigned char * data, size_t width, size_t height,
	size_t pitch_in_bytes)
{
	for (size_t y = 0; y < height; ++y) {
		unsigned int * dstptr = (unsigned int*)(&data[y * pitch_in_bytes]);		
		for (size_t x = 0; x < width; x++) {
			int value = dstptr[x];
			dstptr[x] = value & 0xFF;
			dstptr[x] = (dstptr[x] << 8) | ((value >> 8) & 0xFF);
			dstptr[x] = (dstptr[x] << 8) | ((value >> 16) & 0xFF);
			dstptr[x] = (dstptr[x] << 8) | ((value >> 24) & 0xFF);
		}
	}
}

//------------------------------------------------------------------------------
static int CUDAAPI handle_video_data(void* user_data,
                                     CUVIDSOURCEDATAPACKET* packet)
{
    if (!user_data || !packet) { return 0; }

    Decoder* decoder = reinterpret_cast<Decoder*>(user_data);
    if (decoder->parse_video_data(packet)) {
        return 1;
    }

    return 0;
}

//------------------------------------------------------------------------------
bool Decoder::parse_video_data(CUVIDSOURCEDATAPACKET* packet)
{
    CUresult result = cuvidParseVideoData(video_parser_, packet);

    if ((packet->flags & CUVID_PKT_ENDOFSTREAM) || (result != CUDA_SUCCESS))
        return false;
    return true;
}

//------------------------------------------------------------------------------
static int CUDAAPI handle_video_sequence(void* user_data, CUVIDEOFORMAT* format)
{
    if (!user_data || !format) { return 0; }

    Decoder* decoder = reinterpret_cast<Decoder*>(user_data);
    decoder->session_frame_count = 0;

    if (decoder->compare_video_sequence(format)) {
        return 1;
    }

    return 0;
}

//------------------------------------------------------------------------------
static int CUDAAPI handle_picture_decode(void* user_data,
    CUVIDPICPARAMS* pic_params)
{
    if (!user_data || !pic_params) { return 0; }

    Decoder* decoder = reinterpret_cast<Decoder*>(user_data);
    CUresult r = cuvidDecodePicture(decoder->video_decoder_, pic_params);
    if (r != CUDA_SUCCESS) {
        return 0;
    }
    return 1;
}

//------------------------------------------------------------------------------
static int CUDAAPI handle_picture_display(void* user_data,
    CUVIDPARSERDISPINFO* pic_params)
{
    if (!user_data || !pic_params) { return 0; }
    Decoder* decoder = reinterpret_cast<Decoder*>(user_data);
    if (decoder->session_frame_count == decoder->requested_frame_) {
        if (!decoder->done_requested_frame_) {
            CUdeviceptr       nv12buffer = 0;
            unsigned int      bufferpitch = 0;
            CUVIDPROCPARAMS   params = { 0 };
            params.progressive_frame = pic_params->progressive_frame;
            CUresult r = cuvidMapVideoFrame(decoder->video_decoder_,
                pic_params->picture_index, &nv12buffer, &bufferpitch, &params);
            if (r == CUDA_SUCCESS) {
                assert(0 != nv12buffer);
                assert(0 != bufferpitch);

                unsigned int  bufsize = decoder->format_.coded_height *
                        bufferpitch * 3 / 2;

                CudaAutoLock lock(decoder->ctx_lock_);
                CudaAllocator argb_dev(decoder->format_.coded_width *
                                       decoder->format_.coded_height * 4);

                decoder->cuda_process_frame(&nv12buffer, bufferpitch,
                                            argb_dev.get_pointer(),
                                            decoder->dims_.first * 4);

                r = cuvidUnmapVideoFrame(decoder->video_decoder_, nv12buffer);
                assert(r == CUDA_SUCCESS);

                CUDA_MEMCPY2D cp;
                memset(&cp, 0, sizeof(cp));
                cp.dstMemoryType = CU_MEMORYTYPE_HOST;
                cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                cp.srcDevice = *(argb_dev.get_pointer());
                cp.dstHost = (void *)decoder->target_cpu_buffer_;
                cp.dstPitch = decoder->target_cpu_pitch_;
                cp.srcPitch = decoder->format_.coded_width * 4;
                cp.WidthInBytes = decoder->dims_.first * 4;
                cp.Height = decoder->dims_.second;

                r = cuMemcpy2D(&cp);
                assert(r == CUDA_SUCCESS);
                // Convert to RGBA
                BGRA_2_RGBA((unsigned char *)decoder->target_cpu_buffer_,
                    decoder->dims_.first, decoder->dims_.second,
                            decoder->target_cpu_pitch_);
                decoder->done_requested_frame_ = true;
            }
        }
        return 0;
    }


    decoder->session_frame_count++;
    return 1;
}

//------------------------------------------------------------------------------
bool Decoder::compare_video_sequence(CUVIDEOFORMAT* format)
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
Decoder::Decoder(int device_id)
    :video_parser_(NULL), ctx_lock_(0),
    video_decoder_(NULL), requested_frame_(0), session_frame_count(0),
    target_cpu_buffer_(0), target_cpu_pitch_(0), cuda_module_(0),
    kernelNV12toARGB_(0), color_space_(ITU709), hue_(0.0), legacy_(false)
{
    init_CUDA(device_id);
    cuvidCtxLock(ctx_lock_, 0);

    try {
        if (sizeof(void *) == 4) {
            cuda_module_ = new CUmoduleManager("NV12ToARGB_drvapi_Win32.ptx",
                PTX_DIR, 2, 2, 2);
        }
        else {
            cuda_module_ = new CUmoduleManager("NV12ToARGB_drvapi_x64.ptx",
                PTX_DIR, 2, 2, 2);
        }
    }

    catch (...) {
        std::cout << "Could not load the PTX file" << std::endl;
        throw;
    }

    cuda_module_->GetCudaFunction("NV12ToARGB_drvapi", &kernelNV12toARGB_);
    cuvidCtxUnlock(ctx_lock_, 0);
}

//------------------------------------------------------------------------------
Decoder::~Decoder()
{
    if (video_decoder_) { cuvidDestroyDecoder(video_decoder_); }
    h264_file_.close();
}

//------------------------------------------------------------------------------
std::pair<int, int> Decoder::get_dims() const
{
	return dims_;
}

//------------------------------------------------------------------------------
void Decoder::set_legacy(bool legacy)
{
    legacy_ = legacy;
}

//------------------------------------------------------------------------------
bool Decoder::initialize_decoder(const std::string & path)
{
    file_name_ = path;

    h264_file_.close();
    h264_file_.open(file_name_, std::ios::binary);
    if (!h264_file_.is_open()) {
        return false;
    }

    // Initialize the video source
    CUVIDSOURCEPARAMS vs_params;
    memset(&vs_params, 0, sizeof(CUVIDSOURCEPARAMS));
    vs_params.pUserData = this;
    vs_params.pfnVideoDataHandler = handle_video_data;
    vs_params.pfnAudioDataHandler = NULL; // No audio stuff

    CUvideosource  video_source(0);
    CUresult result = cuvidCreateVideoSource(&video_source,
                                             path.c_str(), &vs_params);

    if (result != CUDA_SUCCESS) { return false; }
    std::memset(&format_, 0, sizeof(format_));
    cuvidGetSourceVideoFormat(video_source, &format_, 0);

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

    dims_.first = format_.display_area.right - format_.display_area.left;
    dims_.second = format_.display_area.bottom - format_.display_area.top;

    // We will not use the video source anymore
    if (video_source) {
        cuvidDestroyVideoSource(video_source);
        video_source = 0;
    }

    CUVIDDECODECREATEINFO vid_decode_create_info;
    memset(&vid_decode_create_info, 0, sizeof(CUVIDDECODECREATEINFO));
    vid_decode_create_info.CodecType = format_.codec;
    vid_decode_create_info.ulWidth = format_.coded_width;
    vid_decode_create_info.ulHeight = format_.coded_height;
    {
        const int decode_mem = 16 * 1024 * 1024;
        const int frame_size = dims_.first * dims_.second;
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

    return cuvidCreateVideoParser(&video_parser_, &vp_params) == CUDA_SUCCESS;
}

//------------------------------------------------------------------------------
bool Decoder::init_CUDA(int deviceID)
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
bool Decoder::decompress(unsigned int frame_number, void * buffer,
    std::size_t buffer_size, std::size_t buffer_pitch)
{
    if (frame_number >= indexes_.size()) {
        // Should we throw an exception here?
        return false;
    }

    target_cpu_buffer_ = buffer;
    target_cpu_pitch_ = buffer_pitch;
    target_cpu_size_ = buffer_size;

    std::size_t estimated_size = dims_.first * dims_.second * 4;
    if (buffer_size < estimated_size) {
        throw std::runtime_error("Buffer size is too small");
    }

    if (buffer_pitch < dims_.first * 4) {
        throw std::runtime_error("Buffer pitch is too small");
    }

    // Get the closest IDR frame
    unsigned int idr_frame_num = frame_number;
    for (int i = frame_number; i >= 0; --i) {
        if (indexes_[i].second == NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_IDR) {
            idr_frame_num = i;
            break;
        }
    }

	// We need different processing if we are dealing with kegacy files
	// or the NV7 encoded files

	if (legacy_) {
		requested_frame_ = frame_number - idr_frame_num;
		session_frame_count = 0;
		done_requested_frame_ = false;
		for (int i = idr_frame_num; i < frame_number + 2; ++i) {
			submit_frame(i, idr_frame_num);
			if (done_requested_frame_) {
				return true;
			}
		}
	}

	else {
		// If the closest IDR frame is the first frame, we are fine.
		if (idr_frame_num == 0) {
			// Get the requested frame relative to the IDR frame
			requested_frame_ = frame_number - idr_frame_num;
			session_frame_count = 0;
			done_requested_frame_ = false;
			for (int i = idr_frame_num; i < frame_number + 3; ++i) {
				submit_frame(i, idr_frame_num);
				if (done_requested_frame_) {
					return true;
				}
			}
		}
		else {
			requested_frame_ = frame_number - idr_frame_num + 1;			
			// We need to process the first frame to ensure this is a valid stream
			submit_frame(0, 0);			
			session_frame_count = 0;
			
			done_requested_frame_ = false;
			for (int i = idr_frame_num; i < frame_number + 3; ++i) {
				submit_frame(i, idr_frame_num);
				if (done_requested_frame_) {
					return true;
				}
			}
		}
	}

    return false;
}

//------------------------------------------------------------------------------
void Decoder::submit_frame(unsigned int frame_number, unsigned int frame_base)
{
    CUVIDSOURCEDATAPACKET packet = { 0 };
    if (frame_number < indexes_.size()) {
        uint_least64_t  offset = indexes_[frame_number].first;
        unsigned int packet_size = 0;
        if ((frame_number + 1) < indexes_.size()) {
            uint_least64_t  next_offset = indexes_[frame_number + 1].first;
            packet_size = next_offset - offset;

        }
        else {
            h264_file_.seekg(0, h264_file_.end);
            packet_size = (unsigned int)((uint_least64_t)h264_file_.tellg()
                - offset);
        }

        packet_buffer_.resize(packet_size);
        unsigned char * ptr = (unsigned char*)&packet_buffer_[0];
        h264_file_.seekg(offset);
        h264_file_.read((char *)ptr, packet_size);
        int read_bytes = h264_file_.gcount();
        packet.payload = &packet_buffer_[0];
        packet.payload_size = packet_size;
    }

    else {
        packet.flags |= CUVID_PKT_ENDOFSTREAM;
    }

    if ((frame_number - frame_base) > requested_frame_) {
        packet.flags |= CUVID_PKT_ENDOFSTREAM;
    }

    CUresult r = cuvidParseVideoData(video_parser_, &packet);
    if (r != CUDA_SUCCESS) {
        throw std::runtime_error("Cannot queue packet for decompression");
    }
}

//------------------------------------------------------------------------------
void Decoder::update_index(unsigned int frame_number, uint_least64_t offset,
                           NV_ENC_PIC_TYPE type)
{
    if (frame_number >= indexes_.size()) {
        indexes_.resize(frame_number + 1);
    }
    indexes_[frame_number] = std::make_pair(offset, type);
}
