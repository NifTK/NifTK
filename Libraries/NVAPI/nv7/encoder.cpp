#include "encoder.h"
#include "hwencoder.h"
#include "helpers.h"

#include <iostream>
#include <cstring>


//------------------------------------------------------------------------------
void copy_ARGB_with_pitch(unsigned char * dst, unsigned char * src, int width,
                          int height, int dst_stride)
{
    for (int y = 0; y < height; ++y) {
        std::memcpy(dst, src, width * 4);
        src += width * 4;
        dst += dst_stride;
    }
}

//------------------------------------------------------------------------------
Encoder::Encoder()
    :device_ptr_(NULL), width_(0), height_(0), alive_(false)
{
    hw_encoder_ = nullptr;
}

//------------------------------------------------------------------------------
Encoder::~Encoder()
{
    deinitialize_encoder();
}

//------------------------------------------------------------------------------
bool Encoder::initialize_encoder(uint32_t width, uint32_t height,
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
    encode_config_.qp = 28;
    encode_config_.i_quant_factor = DEFAULT_I_QFACTOR;
    encode_config_.b_quant_factor = DEFAULT_B_QFACTOR;
    encode_config_.i_quant_offset = DEFAULT_I_QOFFSET;
    encode_config_.b_quant_offset = DEFAULT_B_QOFFSET;
    encode_config_.preset_GUID = NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID;
    encode_config_.pic_struct = NV_ENC_PIC_STRUCT_FRAME;
    encode_config_.input_format = NV_ENC_BUFFER_FORMAT_ABGR;

    // Create the encoder.
    status = hw_encoder_->CreateEncoder(&encode_config_);

    if (status != NV_ENC_SUCCESS) { return false; }

    buffer_count_ = NUM_ENCODE_QUEUE;

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
void Encoder::deinitialize_encoder()
{
    release_io_buffers();

    if (hw_encoder_) {
        hw_encoder_->NvEncDestroyEncoder();
        hw_encoder_.reset(nullptr);
    }

    if (device_ptr_) {
        CUresult cuResult = CUDA_SUCCESS;
        cuResult = cuCtxDestroy((CUcontext)device_ptr_);
        device_ptr_ = NULL;
    }
}

//------------------------------------------------------------------------------
void Encoder::clear_nals()
{
	output_nal_.clear();
}

//------------------------------------------------------------------------------
bool Encoder::write_nal_to_file(const std::string & file)
{
	std::ofstream nal_file(file);
	if (nal_file.is_open()) {
		for (int i = 0; i < output_nal_.size(); ++i) {
			nal_file << i << ' ' << output_nal_[i].first << ' ' 
				<< output_nal_[i].second << std::endl;
		}
		return true;
	}

	return false;
}

//------------------------------------------------------------------------------
bool Encoder::init_CUDA(uint32_t deviceID)
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

    if (deviceID > (unsigned int)deviceCount - 1) {
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
bool Encoder::allocate_io_buffers(NV_ENC_BUFFER_FORMAT format)
{
    if (width_ <= 0 || height_ <= 0) { return false ; }

    NVENCSTATUS status = NV_ENC_SUCCESS;
    for (uint32_t i = 0; i < NUM_ENCODE_QUEUE; ++i) {
        EncodeBuffer * buf = new EncodeBuffer;
        allocated_buffers_.push_back(buf);

        hw_encoder_->NvEncCreateInputBuffer(width_, height_,
                                           &buf->input_buf.input_surface,
                                           format);

        if (status != NV_ENC_SUCCESS) { return false; }

        buf->input_buf.format = format;
        buf->input_buf.width = width_;
        buf->input_buf.height = height_;
        status = hw_encoder_->NvEncCreateBitstreamBuffer(BITSTREAM_BUFFER_SIZE,
                                            &buf->output_buf.bitstream_buf);

        if (status != NV_ENC_SUCCESS) { return false; }

        buf->output_buf.bitstream_buf_size = BITSTREAM_BUFFER_SIZE;
        buf->output_buf.eos_flag = false;
        buf->output_buf.busy = false;
        buf->output_buf.wait_on_event = false;

        if (encode_config_.enable_async_mode)
        {
            status = hw_encoder_->NvEncRegisterAsyncEvent(
                        &buf->output_buf.output_event);

            buf->output_buf.wait_on_event = true;
            if (status != NV_ENC_SUCCESS) { return false; }
        }
        else {
            buf->output_buf.output_event = NULL;
        }
        // This buffer is now available.
        available_buffers_.push(buf);
    }

    eos_output_buffer_.eos_flag = true;
    eos_output_buffer_.output_event = NULL;

    // TO DO: Perform setup when we have to deal with async events
    if (encode_config_.enable_async_mode) {
        status = hw_encoder_->NvEncRegisterAsyncEvent(
                    &eos_output_buffer_.output_event);
    }

    return (status == NV_ENC_SUCCESS);
}

//------------------------------------------------------------------------------
bool Encoder::release_io_buffers()
{
    alive_ = false;
    // If there is no encoder, silently exit.
    if (!hw_encoder_) return true;

    for (auto i: allocated_buffers_) {
        hw_encoder_->NvEncDestroyInputBuffer(i->input_buf.input_surface);
        i->input_buf.input_surface = NULL;
        hw_encoder_->NvEncDestroyBitstreamBuffer(i->output_buf.bitstream_buf);
        i->output_buf.bitstream_buf = NULL;

        if (encode_config_.enable_async_mode) {
            hw_encoder_->NvEncUnregisterAsyncEvent(i->output_buf.output_event);
            close_file(i->output_buf.output_event);
            i->output_buf.output_event = NULL;
        }
    }

    if (eos_output_buffer_.output_event) {
        if (encode_config_.enable_async_mode) {
            hw_encoder_->NvEncUnregisterAsyncEvent(
                        eos_output_buffer_.output_event);
            close_file(eos_output_buffer_.output_event);
            eos_output_buffer_.output_event = NULL;
        }
    }

    allocated_buffers_.clear();
    std::queue<EncodeBuffer *>().swap(available_buffers_);
    std::queue<EncodeBuffer *>().swap(serializable_buffers_);
    std::queue<EncodeBuffer *>().swap(busy_buffers_);

    return true;
}

//------------------------------------------------------------------------------
bool Encoder::encode_frame(unsigned char *argb)
{
    // Sanity checks
    if (!hw_encoder_ || !argb || width_ <= 0 || height_ <= 0 || !alive_) {
        return false ;
    }
    
    EncodeBuffer * buf = get_available();

    if (!buf) { return false; }  // No free buffers available

    // Lock the buffer
    unsigned char * input_surface;
    uint32_t locked_pitch = 0;

    NVENCSTATUS status = hw_encoder_->NvEncLockInputBuffer(
                buf->input_buf.input_surface, (void**)&input_surface,
                &locked_pitch);

    if (status != NV_ENC_SUCCESS) { return false; }

    // Copy the data over
    copy_ARGB_with_pitch(input_surface, argb, width_, height_, locked_pitch);

    // Unlock the buffer
    status = hw_encoder_->NvEncUnlockInputBuffer(buf->input_buf.input_surface);

    if (status != NV_ENC_SUCCESS) { return false; }

    // Encode the frame
    status = hw_encoder_->NvEncEncodeFrame(buf, NULL, width_, height_,
                                           (NV_ENC_PIC_STRUCT)
                                           encode_config_.pic_struct);

    // Buffer is not busy. So all busy buffers are now ready
    if (!buf->output_buf.busy) {
        while(!busy_buffers_.empty()) {
            EncodeBuffer * busy_buf = busy_buffers_.front();
            busy_buf->output_buf.busy = false;
            busy_buffers_.pop();
            add_for_serialization(busy_buf);
        }
        // Add to the serialization queue
        add_for_serialization(buf);
    }
    // Buffer is busy.
    else {
        busy_buffers_.push(buf);
    }

    return (status == NV_ENC_SUCCESS);
}

//------------------------------------------------------------------------------
bool Encoder::flush_encoder()
{		
    // Sanity checks
    if (!hw_encoder_ || width_ <= 0 || height_ <= 0 || !alive_) {
        return false;
    }

    alive_ = false;

    // Issue EOS event
    NVENCSTATUS status = hw_encoder_->NvEncFlushEncoderQueue(
                eos_output_buffer_.output_event);

    if (status != NV_ENC_SUCCESS) { return false; }

    // Wait for the serialization thread to finish processing.
    process_thread_.join();

    // Look for any leftover frames needed to be flushed
    while (!serializable_buffers_.empty()) {
        EncodeBuffer * buf = serializable_buffers_.front();
        serializable_buffers_.pop();
        SerializedInfo result = hw_encoder_->process_output_buffer(buf);
        if (std::get<0>(result)) {
            output_nal_.push_back(std::make_pair(std::get<1>(result),
                std::get<2>(result)));
        }
    }	

	
#if defined(NV_WINDOWS)	
    if (encode_config_.enable_async_mode) {		
        if (WaitForSingleObject(eos_output_buffer_.output_event, 5000)
                != WAIT_OBJECT_0) {
            return false;
        }
    }
#endif 
    //hw_encoder_->flush();
    return true;
}

//------------------------------------------------------------------------------
EncodeBuffer * Encoder::get_available()
{
    EncodeBuffer * available = NULL;
    auto lock = notstd::get_lock(mutex_);
    if (!available_buffers_.empty()) {
        available = available_buffers_.front();
        available_buffers_.pop();
    }
    return available;
}

//------------------------------------------------------------------------------
void Encoder::add_for_serialization(EncodeBuffer * buf)
{
    if (buf) {
        auto lock = notstd::get_lock(mutex_);
        serializable_buffers_.push(buf);
    }
}

//------------------------------------------------------------------------------
void Encoder::add_for_encoding(EncodeBuffer * buf)
{
    if (buf) {
        auto lock = notstd::get_lock(mutex_);
        available_buffers_.push(buf);
    }
}

//------------------------------------------------------------------------------
EncodeBuffer * Encoder::get_next_for_serialization()
{
    auto lock = notstd::get_lock(mutex_);
    EncodeBuffer * buf = serializable_buffers_.front();
    serializable_buffers_.pop();
    return buf;
}

//------------------------------------------------------------------------------
void Encoder::process_compressed_frames()
{
    // Start a thread that takes care of serializing frames
    process_thread_ = std::thread([this] {
        while (alive_) {
            while (!serializable_buffers_.empty()) {
                auto buf = get_next_for_serialization();
                SerializedInfo result = hw_encoder_->process_output_buffer(buf);
                if (std::get<0>(result)) {
                    output_nal_.push_back(std::make_pair(std::get<1>(result),
                        std::get<2>(result)));
                }
                add_for_encoding(buf);
            }
        }
    });
}

//------------------------------------------------------------------------------
bool Encoder::get_output_info(unsigned int frame_number,
    uint_least64_t & fileoffset, NV_ENC_PIC_TYPE & frame_type)
{
    auto lock = notstd::get_lock(mutex_);
    if (output_nal_.size() > frame_number) {
        fileoffset = output_nal_[frame_number].first;
        frame_type = output_nal_[frame_number].second;
        return true;
    }
    return false;
}
