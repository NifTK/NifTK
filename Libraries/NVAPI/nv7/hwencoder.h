#ifndef HWENCODER_H__
#define HWENCODER_H__

#include "nvEncodeAPI.h"
#include "macros.h"
#include "encoderdefs.h"
#include "dynlink_cuda.h"
#include <fstream>
#include <string>
#include <memory>
#include <tuple>

#if defined(NV_UNIX)
#include <dlfcn.h>
#endif

#define SET_VER(configStruct, type) {configStruct.version = type##_VER;}

//! Some utility functions for unique_ptr
struct LibraryDeleter
{
    typedef HINSTANCE pointer;
    void operator()(HINSTANCE inst) const
    {
#if defined(NV_WINDOWS)
        FreeLibrary(inst);
#else
        dlclose(inst);
#endif
    }
};

//! Compression codecs available.
enum CodecType
{
    NV_ENC_H264 = 0,
    NV_ENC_HEVC = 1,
};

struct PictureCommand
{
    bool res_change_pending;
    bool bitrate_change_pending;
    bool force_idr;
    bool force_intra_refresh;
    bool invalidate_ref_frames;
    uint32_t  intra_ref_dur;
};

struct EncodeInputBuffer
{
    unsigned int            width;
    unsigned int            height;
    CUdeviceptr             nv12_devptr;
    uint32_t                nv12_stride;
    CUdeviceptr             nv12_tmp_devptr;
    uint32_t                nv12_tmp_stride;
    void*                   nv_reg_source;
    NV_ENC_INPUT_PTR        input_surface;
    NV_ENC_BUFFER_FORMAT    format;
};

struct EncodeOutputBuffer
{
    unsigned int          bitstream_buf_size;
    NV_ENC_OUTPUT_PTR     bitstream_buf;
    HANDLE                output_event;
    bool                  wait_on_event;
    bool                  eos_flag;
    bool				  busy; // Need this for synchronous operation
};

struct EncodeBuffer
{
    EncodeOutputBuffer      output_buf;
    EncodeInputBuffer       input_buf;
};

//! Encoder configuration parameters
struct EncodeConfig
{
    EncodeConfig()
    {
        width = height = 0;
        bitrate = vbv_max_brate = vbv_size = rc_mode = qp = fps = 0;
		preset_GUID = { 0 };
        gop_length = num_B = pic_struct = 0;
        i_quant_factor = b_quant_factor = i_quant_offset = b_quant_offset = 0.f;
        intra_ref_enable = intra_ref_period = intra_ref_dur = 0;
        enable_async_mode = enable_temp_AQ = 0;
        enable_async_mode = 1;
        invalidate_ref_frames = 0;
        input_format = NV_ENC_BUFFER_FORMAT_ARGB;
        codec = CodecType::NV_ENC_H264;
    }

    int                                 width;
    int                                 height;
    GUID                                preset_GUID;
    int                                 fps;
    int                                 gop_length;
    int                                 num_B;
    int                                 pic_struct;

    int                                 bitrate;
    int                                 vbv_max_brate;
    int                                 vbv_size;
    int                                 rc_mode;
    int                                 qp;
    float                               i_quant_factor;
    float                               b_quant_factor;
    float                               i_quant_offset;
    float                               b_quant_offset;

    int                                 intra_ref_enable;
    int                                 intra_ref_period;
    int                                 intra_ref_dur;

    int                                 enable_async_mode;
    int                                 enable_temp_AQ;

    int                                 invalidate_ref_frames;

    std::string                         output_file;
    NV_ENC_BUFFER_FORMAT                input_format;
    CodecType                           codec;
};

//! Wrapper around the NVENC interface.
class HWEncoder
{
public:
    HWEncoder();
    virtual ~HWEncoder();
    NVENCSTATUS Initialize(void * device,
                         NV_ENC_DEVICE_TYPE deviceType=NV_ENC_DEVICE_TYPE_CUDA);
    NVENCSTATUS CreateEncoder(EncodeConfig * config);
    NVENCSTATUS ValidateEncodeGUID(GUID input_codec_GUID);

    SerializedInfo process_output_buffer(const EncodeBuffer * buf);

    inline uint_least64_t get_num_encoded_bytes() const
    {
        return encoded_bytes_count_;
    }

    void flush();

public:
    //! Open an encoding session and returns a pointer to the encoder interface
    NVENCSTATUS NvEncOpenEncodeSessionEx(void* device,
                                         NV_ENC_DEVICE_TYPE device_type);

    //! Get the number of supported encode GUIDs
    NVENCSTATUS NvEncGetEncodeGUIDCount(uint32_t* encodeGUIDCount);

    //! Return the array of codec GUIDs supported by the encoder.
    NVENCSTATUS NvEncGetEncodeGUIDs(GUID* GUIDs, uint32_t guid_array_size,
                                    uint32_t* guid_count);

    //! Return the number of supported input formats
    NVENCSTATUS NvEncGetInputFormatCount(GUID encode_GUID,
                                         uint32_t* input_fmt_count);

    //! Return an array of supported input formats
    NVENCSTATUS NvEncGetInputFormats(GUID encode_GUID,
                                     NV_ENC_BUFFER_FORMAT* input_fmts,
                                     uint32_t input_fmt_array_size,
                                     uint32_t* input_fmt_count);

    //! Return the capability value for a given encoder attribute
    NVENCSTATUS NvEncGetEncodeCaps(GUID encode_GUID,
                                   NV_ENC_CAPS_PARAM* caps_param, int* caps_val);

    //! Return the number of preset GUIDs available for a given codec
    NVENCSTATUS NvEncGetEncodePresetCount(GUID encode_GUID,
                                          uint32_t* encode_preset_GUID_count);

    //! Return an array of encode preset guids available for a given codec
    NVENCSTATUS NvEncGetEncodePresetGUIDs(GUID encode_GUID, GUID* preset_GUIDs,
                                          uint32_t guid_array_size,
                                          uint32_t* encode_preset_GUID_count);

    //! Returns a preset config structure for a given preset guid
    NVENCSTATUS NvEncGetEncodePresetConfig(GUID encode_GUID, GUID  preset_GUID,
                                           NV_ENC_PRESET_CONFIG* preset_config);

    //! Allocate an input buffer
    NVENCSTATUS NvEncCreateInputBuffer(uint32_t width, uint32_t height,
                                       void** input_buffer,
                                       NV_ENC_BUFFER_FORMAT input_format);

    //! Free an input buffer
    NVENCSTATUS NvEncDestroyInputBuffer(NV_ENC_INPUT_PTR input_buffer);

    //! Allocate an output bitstream buffer
    NVENCSTATUS NvEncCreateBitstreamBuffer(uint32_t size,
                                           void** bitstreamBuffer);

    //! Free an output bitstream buffer
    NVENCSTATUS NvEncDestroyBitstreamBuffer(NV_ENC_OUTPUT_PTR bitstreamBuffer);

    //! Lock the bit stream buffer to read the encoded data
    NVENCSTATUS NvEncLockBitstream(NV_ENC_LOCK_BITSTREAM*
                                   lockBitstreamBufferParams);

    //! Unlock the output bitstream buffer after the client has read the
    //! encoded data from output buffer
    NVENCSTATUS NvEncUnlockBitstream(NV_ENC_OUTPUT_PTR bitstreamBuffer);

    //! Lock the input buffer to load the uncompressed pixel data into
    //! input buffer memory
    NVENCSTATUS NvEncLockInputBuffer(void* inputBuffer, void** bufferDataPtr,
                                     uint32_t* pitch);

    //! Unlock the input buffer memory previously locked for uploading
    //! pixel data
    NVENCSTATUS NvEncUnlockInputBuffer(NV_ENC_INPUT_PTR inputBuffer);

    //! Register the completion event with NvEncodeAPI interface for
    //! asynchronous mode.
    NVENCSTATUS NvEncRegisterAsyncEvent(void** completionEvent);

    //! Unregister a previously registered completion event
    NVENCSTATUS NvEncUnregisterAsyncEvent(void* completionEvent);

    //! Map an externally allocated input resource
    NVENCSTATUS NvEncMapInputResource(void* registeredResource,
                                      void** mappedResource);

    //! UnMap a previously mapped input buffer
    NVENCSTATUS NvEncUnmapInputResource(NV_ENC_INPUT_PTR mappedInputBuffer);

    //! Destroy an encoder previously created
    NVENCSTATUS NvEncDestroyEncoder();

    //! Registers a resource with the Nvidia Video Encoder Interface for
    //! book keeping
    NVENCSTATUS NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE resourceType,
                                      void* resourceToRegister, uint32_t width,
                                      uint32_t height, uint32_t pitch,
                                      void** registeredResource);

    //! Unregister a previously registered resource
    NVENCSTATUS NvEncUnregisterResource(NV_ENC_REGISTERED_PTR registeredRes);

    NVENCSTATUS NvEncEncodeFrame(EncodeBuffer *pEncodeBuffer,
                         PictureCommand *encPicCommand,
                         uint32_t width, uint32_t height,
                         NV_ENC_PIC_STRUCT ePicStruct = NV_ENC_PIC_STRUCT_FRAME,
                         int8_t *qpDeltaMapArray = NULL,
                         uint32_t qpDeltaMapArraySize = 0);

    //! Wrapper around NvEncEncodePicture API to submit an input picture
    //! for encoding
    NVENCSTATUS NvEncFlushEncoderQueue(void *hEOSEvent);

private:
    //! NV function list interface.
    std::unique_ptr<NV_ENCODE_API_FUNCTION_LIST>		encode_api_;
    //! Encoder interface
    void *                                              encoder_iface_;

    NV_ENC_INITIALIZE_PARAMS                            encoder_params_;
    //! Compressor configuration
    NV_ENC_CONFIG                                       encode_config_;
    //! Input size parameters
    uint32_t                                            width_;
    uint32_t                                            height_;
    uint32_t                                            max_width_;
    uint32_t                                            max_height_;

    //! Output file
    std::ofstream										output_file_;
    //OverlappedFileWriter								output_file_;

    //! Handle to the dynamic library
    std::unique_ptr<HINSTANCE, LibraryDeleter>          hinst_lib_;
    //! Codec type
    CodecType                                           codec;
    //! Counter for keeping track of number of encoded frames
    uint32_t											encode_count;
    //! Counter for number of bytes serialized
    uint_least64_t                                      encoded_bytes_count_;

    // No copy semantics
    DISALLOW_COPY_AND_ASSIGNMENT(HWEncoder);
};

typedef NVENCSTATUS(NVENCAPI *MYPROC)(NV_ENCODE_API_FUNCTION_LIST*);

#endif
