#include "textureencoder.h"

//------------------------------------------------------------------------------
TextureEncoder::TextureEncoder()
    :frame_buffer_(0), device_ptr_(NULL), width_(0), height_(0), alive_(false)
{
    hw_encoder_ = nullptr;
}

//------------------------------------------------------------------------------
TextureEncoder::~TextureEncoder()
{

}

//------------------------------------------------------------------------------
bool TextureEncoder::initialize_encoder(uint32_t width, uint32_t height,
                                        uint32_t dev_ID, int fps,
                                        const std::string & output_file)
{
    if (width <= 0 || height <= 0) { return false; }

    // Set the width and height
    width_ = width;
    height_ = height;

    // Reset everything
    deinitialize_encoder();
    hw_encoder_.reset(new HWEncoder);

    // Initialize CUDA.
    if (!init_CUDA(dev_ID)) { return false; }

    // Initialize the encoder interface. We only support the CUDA interface
    NVENCSTATUS status = hw_encoder_->Initialize(device_ptr_,
                                                 NV_ENC_DEVICE_TYPE_CUDA);

    if (status != NV_ENC_SUCCESS) { return false; }

    // Create the encoder. We set it to some defaults at the moment.
    encode_config_.width = width_;
    encode_config_.height = height_;
    encode_config_.bitrate = 5000000;
    encode_config_.output_file = output_file;
    encode_config_.rc_mode = NV_ENC_PARAMS_RC_CONSTQP;
    encode_config_.gop_length = 15;
    encode_config_.codec = NV_ENC_H264;
    encode_config_.fps = fps;
    encode_config_.qp = 0;
    encode_config_.i_quant_factor = DEFAULT_I_QFACTOR;
    encode_config_.b_quant_factor = DEFAULT_B_QFACTOR;
    encode_config_.i_quant_offset = DEFAULT_I_QOFFSET;
    encode_config_.b_quant_offset = DEFAULT_B_QOFFSET;
    encode_config_.preset_GUID = NV_ENC_PRESET_LOSSLESS_HP_GUID;
    encode_config_.pic_struct = NV_ENC_PIC_STRUCT_FRAME;
    encode_config_.input_format = NV_ENC_BUFFER_FORMAT_NV12;

    // Create the encoder.
    status = hw_encoder_->CreateEncoder(&encode_config_);

    if (status != NV_ENC_SUCCESS) { return false; }

    bool success = allocate_io_buffers(encode_config_.input_format);

    // Clear any previous index info
    output_nal_.clear();

    // prepare for receiving frames
    if (success) {
        alive_ = true;
        process_compressed_frames();  // start listening for frames
    }
    return success;
}

//------------------------------------------------------------------------------
bool TextureEncoder::init_CUDA(uint32_t deviceID)
{
	CUresult cuResult;
	CUdevice device;
	CUcontext cuContextCurr;
	int  deviceCount = 0;
	int  SMminor = 0, SMmajor = 0;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	typedef HMODULE CUDADRIVER;
#else
	typedef void *CUDADRIVER;
#endif
	CUDADRIVER hHandleDriver = 0;
	cuResult = cuInit(0, __CUDA_API_VERSION, hHandleDriver);
	if (cuResult != CUDA_SUCCESS) {
		return false; // NV_ENC_ERR_NO_ENCODE_DEVICE
	}

	cuResult = cuDeviceGetCount(&deviceCount);
	if (cuResult != CUDA_SUCCESS) {
		return false; // NV_ENC_ERR_NO_ENCODE_DEVICE
	}

	// If dev is negative value, we clamp to 0
	if ((int)deviceID < 0) { deviceID = 0; }

	if (deviceID >(unsigned int)deviceCount - 1) {
		return false; // NV_ENC_ERR_INVALID_ENCODERDEVICE
	}

	cuResult = cuDeviceGet(&device, deviceID);
	if (cuResult != CUDA_SUCCESS) {
		return false; // NV_ENC_ERR_NO_ENCODE_DEVICE;
	}

	cuResult = cuDeviceComputeCapability(&SMmajor, &SMminor, deviceID);
	if (cuResult != CUDA_SUCCESS) {
		return false; //NV_ENC_ERR_NO_ENCODE_DEVICE
	}

	// Check if the GPU has encode capabilities
	if (((SMmajor << 4) + SMminor) < 0x30)
	{
		return false; //NV_ENC_ERR_NO_ENCODE_DEVICE;
	}

	cuResult = cuCtxCreate((CUcontext*)(&device_ptr_), 0, device);
	if (cuResult != CUDA_SUCCESS)
	{
		return false;
	}

	cuResult = cuCtxPopCurrent(&cuContextCurr);
	if (cuResult != CUDA_SUCCESS)
	{
		return false;
	}

	return true;
}

//------------------------------------------------------------------------------
bool TextureEncoder::allocate_io_buffers(NV_ENC_BUFFER_FORMAT format)
{
	return false;
}

//------------------------------------------------------------------------------
bool TextureEncoder::release_io_buffers()
{
	return false;
}

//------------------------------------------------------------------------------
void TextureEncoder::process_compressed_frames()
{}

//------------------------------------------------------------------------------
void TextureEncoder::deinitialize_encoder()
{}