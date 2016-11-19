#include "hwencoder.h"
#include <cassert>
#include <cstring>
#include <cmath>
#include <iostream>

HWEncoder::HWEncoder()
    :encode_api_(nullptr), hinst_lib_(nullptr), width_(0), height_(0),
    max_width_(0), max_height_(0), /*output_file_("output.h264"),*/
    encode_count(0), encoded_bytes_count_(0)
{
    std::memset(&encoder_params_, 0, sizeof(encoder_params_));
    SET_VER(encoder_params_, NV_ENC_INITIALIZE_PARAMS);

    std::memset(&encode_config_, 0, sizeof(encode_config_));
    SET_VER(encode_config_, NV_ENC_CONFIG);
}

HWEncoder::~HWEncoder()
{}

void HWEncoder::flush()
{
    if (output_file_.is_open()) {
        output_file_.flush();
    }
}

NVENCSTATUS HWEncoder::Initialize(void * device, NV_ENC_DEVICE_TYPE device_type)
{
    NVENCSTATUS status = NV_ENC_SUCCESS;
    // function pointer to create instance in nvEncodeAPI
    MYPROC encode_api_instance;

#if defined(NV_WINDOWS)
    #if defined (_WIN64)
        hinst_lib_.reset(LoadLibrary(TEXT("nvEncodeAPI64.dll")));
    #else
        hinst_lib_.reset(LoadLibrary(TEXT("nvEncodeAPI.dll")));
#endif
#else
    hinst_lib_.reset(dlopen("libnvidia-encode.so.1", RTLD_LAZY));
#endif

    if (!hinst_lib_)
        return NV_ENC_ERR_OUT_OF_MEMORY;

#if defined(NV_WINDOWS)
    encode_api_instance =
           (MYPROC)GetProcAddress(hinst_lib_.get(),"NvEncodeAPICreateInstance");
#else
    encode_api_instance = (MYPROC)dlsym(hinst_lib_.get(),
                                              "NvEncodeAPICreateInstance");
#endif

    if (encode_api_instance == NULL)
        return NV_ENC_ERR_OUT_OF_MEMORY;

    encode_api_.reset(new NV_ENCODE_API_FUNCTION_LIST);
    if (!encode_api_)
        return NV_ENC_ERR_OUT_OF_MEMORY;

    memset(encode_api_.get(), 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    encode_api_->version = NV_ENCODE_API_FUNCTION_LIST_VER;
    status = encode_api_instance(encode_api_.get());
    if (status != NV_ENC_SUCCESS)
        return status;

    status = NvEncOpenEncodeSessionEx(device, device_type);
    if (status != NV_ENC_SUCCESS)
        return status;

    return NV_ENC_SUCCESS;
}

NVENCSTATUS HWEncoder::CreateEncoder(EncodeConfig * config)
{
    if (!config) {
        return NV_ENC_ERR_INVALID_PARAM;
    }

    // Setup the image size parameters
    width_ = config->width;
    height_ = config->height;

    max_width_ = config->width;
    max_height_ = config->height;

    if ((width_ > max_width_) || (height_ > max_height_)) {
        return NV_ENC_ERR_INVALID_PARAM;
    }

    // Output file
    output_file_.open(config->output_file, std::ios::binary);

    if (!config->width || !config->height || !output_file_) {
        return NV_ENC_ERR_INVALID_PARAM;
    }

    if ((config->input_format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT ||
         config->input_format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
        && (config->codec == NV_ENC_H264)) {
        // 10 bit is not supported with H264
        return NV_ENC_ERR_INVALID_PARAM;
    }

    GUID in_codec_GUID = config->codec == CodecType::NV_ENC_H264 ?
                NV_ENC_CODEC_H264_GUID : NV_ENC_CODEC_HEVC_GUID;
    codec = config->codec;

    NVENCSTATUS status = ValidateEncodeGUID(in_codec_GUID);
    if (status != NV_ENC_SUCCESS) {
        // Codec is not supported by the underlying device.
        return status;
    }

    encoder_params_.encodeGUID = in_codec_GUID;
    encoder_params_.presetGUID = config->preset_GUID;
    encoder_params_.encodeWidth = config->width;
    encoder_params_.encodeHeight = config->height;

    encoder_params_.darWidth = config->width;
    encoder_params_.darHeight = config->height;
    encoder_params_.frameRateNum = config->fps;
    encoder_params_.frameRateDen = 1;
    encoder_params_.enableEncodeAsync = 0;

    encoder_params_.enablePTD = 1;
    encoder_params_.reportSliceOffsets = 0;
    encoder_params_.enableSubFrameWrite = 0;
    encoder_params_.encodeConfig = &encode_config_;
    encoder_params_.maxEncodeWidth = max_width_;
    encoder_params_.maxEncodeHeight = max_height_;

    // Apply preset
    NV_ENC_PRESET_CONFIG preset_cfg;
    memset(&preset_cfg, 0, sizeof(NV_ENC_PRESET_CONFIG));
    SET_VER(preset_cfg, NV_ENC_PRESET_CONFIG);
    SET_VER(preset_cfg.presetCfg, NV_ENC_CONFIG);

    status = encode_api_->nvEncGetEncodePresetConfig(encoder_iface_,
        encoder_params_.encodeGUID,
        encoder_params_.presetGUID, &preset_cfg);
    if (status != NV_ENC_SUCCESS)
    {
        return status;
    }
    memcpy(&encode_config_, &preset_cfg.presetCfg, sizeof(NV_ENC_CONFIG));

    encode_config_.gopLength = config->gop_length;
    encode_config_.frameIntervalP = config->num_B + 1;
    if (config->pic_struct == NV_ENC_PIC_STRUCT_FRAME)
    {
        encode_config_.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
    }
    else
    {
        encode_config_.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD;
    }

    encode_config_.mvPrecision = NV_ENC_MV_PRECISION_QUARTER_PEL;

    if (config->bitrate || config->vbv_max_brate)
    {
        encode_config_.rcParams.rateControlMode =
                (NV_ENC_PARAMS_RC_MODE)config->rc_mode;

        encode_config_.rcParams.averageBitRate = config->bitrate;
        encode_config_.rcParams.maxBitRate = config->vbv_max_brate;
        encode_config_.rcParams.vbvBufferSize = config->vbv_size;
        encode_config_.rcParams.vbvInitialDelay = config->vbv_size * 9 / 10;
    }
    else
    {
        encode_config_.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
    }

    if (config->rc_mode == 0)
    {
        encode_config_.rcParams.constQP.qpInterP = config->preset_GUID == NV_ENC_PRESET_LOSSLESS_HP_GUID ? 0 : config->qp;
        encode_config_.rcParams.constQP.qpInterB = config->preset_GUID == NV_ENC_PRESET_LOSSLESS_HP_GUID ? 0 : config->qp;
        encode_config_.rcParams.constQP.qpIntra = config->preset_GUID == NV_ENC_PRESET_LOSSLESS_HP_GUID ? 0 : config->qp;
    }

    // set up initial QP value
    if (config->rc_mode == NV_ENC_PARAMS_RC_VBR ||
            config->rc_mode == NV_ENC_PARAMS_RC_VBR_MINQP ||
            config->rc_mode == NV_ENC_PARAMS_RC_2_PASS_VBR)
    {
        encode_config_.rcParams.enableInitialRCQP = 1;
        encode_config_.rcParams.initialRCQP.qpInterP = config->qp;
        if (config->i_quant_factor != 0.0 && config->b_quant_factor != 0.0) {
            encode_config_.rcParams.initialRCQP.qpIntra =
                    (int)(config->qp * std::fabs(config->i_quant_factor) +
                          config->i_quant_offset);

            encode_config_.rcParams.initialRCQP.qpInterB = (int)(config->qp *
                  std::fabs(config->b_quant_factor) + config->b_quant_offset);
        }
        else {
            encode_config_.rcParams.initialRCQP.qpIntra = config->qp;
            encode_config_.rcParams.initialRCQP.qpInterB = config->qp;
        }
    }

    if (config->input_format == NV_ENC_BUFFER_FORMAT_YUV444 ||
            config->input_format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
    {
        if (config->codec == NV_ENC_HEVC) {
            encode_config_.encodeCodecConfig.hevcConfig.chromaFormatIDC = 3;
        }
        else {
            encode_config_.encodeCodecConfig.h264Config.chromaFormatIDC = 3;
        }
    }
    else
    {
        if (config->codec == NV_ENC_HEVC) {
            encode_config_.encodeCodecConfig.hevcConfig.chromaFormatIDC = 1;
        }
        else {
            encode_config_.encodeCodecConfig.h264Config.chromaFormatIDC = 1;
        }
    }

    if (config->input_format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT ||
            config->input_format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) {
        if (config->codec == NV_ENC_HEVC) {
            encode_config_.encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 = 2;
        }
    }

    if (config->intra_ref_enable)
    {
        if (config->codec == NV_ENC_HEVC) {
            encode_config_.encodeCodecConfig.hevcConfig.enableIntraRefresh = 1;
            encode_config_.encodeCodecConfig.hevcConfig.intraRefreshPeriod =
                    config->intra_ref_period;
            encode_config_.encodeCodecConfig.hevcConfig.intraRefreshCnt =
                    config->intra_ref_dur;
        }
        else {
            encode_config_.encodeCodecConfig.h264Config.enableIntraRefresh = 1;
            encode_config_.encodeCodecConfig.h264Config.intraRefreshPeriod =
                    config->intra_ref_period;
            encode_config_.encodeCodecConfig.h264Config.intraRefreshCnt =
                    config->intra_ref_dur;
        }
    }

    if (config->invalidate_ref_frames)
    {
        if (config->codec == NV_ENC_HEVC) {
            encode_config_.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB
                    = 16;
        }
        else {
            encode_config_.encodeCodecConfig.h264Config.maxNumRefFrames = 16;
        }
    }

    if (config->codec == NV_ENC_H264) {
        encode_config_.encodeCodecConfig.h264Config.idrPeriod
                = config->gop_length;
        //encode_config_.encodeCodecConfig.h264Config.qpPrimeYZeroTransformBypassFlag = 1;
    }
    else if (config->codec == NV_ENC_HEVC) {
        encode_config_.encodeCodecConfig.hevcConfig.idrPeriod
                = config->gop_length;
    }

    NV_ENC_CAPS_PARAM caps_param;
    int async_mode = 0;
    memset(&caps_param, 0, sizeof(NV_ENC_CAPS_PARAM));
    SET_VER(caps_param, NV_ENC_CAPS_PARAM);
    caps_param.capsToQuery = NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT;
    encode_api_->nvEncGetEncodeCaps(encoder_iface_, encoder_params_.encodeGUID,
                                    &caps_param, &async_mode);
    encoder_params_.enableEncodeAsync = async_mode;
    config->enable_async_mode = async_mode;

    if (config->enable_temp_AQ == 1) {
        memset(&caps_param, 0, sizeof(NV_ENC_CAPS_PARAM));
        SET_VER(caps_param, NV_ENC_CAPS_PARAM);
        caps_param.capsToQuery = NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ;
        int temp_AQ_supported = 0;
        // Check if temporal AQ is actually supported.
        status = encode_api_->nvEncGetEncodeCaps(encoder_iface_,
                                      encoder_params_.encodeGUID,
                                      &caps_param, &temp_AQ_supported);

        if (status != NV_ENC_SUCCESS) {
            return status;
        }
        else
        {
            if (temp_AQ_supported == 1)
            {
                encode_config_.rcParams.enableTemporalAQ = 1;
            }
            else
            {
                return NV_ENC_ERR_UNSUPPORTED_DEVICE;
            }
        }
    }

    // Finally try and initialize the encoder
    return encode_api_->nvEncInitializeEncoder(encoder_iface_,
                                                &encoder_params_);
}

NVENCSTATUS HWEncoder::ValidateEncodeGUID(GUID input_codec_GUID)
{
    unsigned int i, codec_found, encode_GUID_count, encode_GUID_array_size;
    NVENCSTATUS status = encode_api_->nvEncGetEncodeGUIDCount(encoder_iface_,
                                                          &encode_GUID_count);
    if (status != NV_ENC_SUCCESS)
    {
        return status;
    }

    std::unique_ptr<GUID[]> encodeGUIDArray(new GUID[encode_GUID_count]);
    std::memset(encodeGUIDArray.get(), 0, sizeof(GUID)* encode_GUID_count);
    encode_GUID_array_size = 0;
    status = encode_api_->nvEncGetEncodeGUIDs(encoder_iface_,
                                                encodeGUIDArray.get(),
                                                encode_GUID_count,
                                                &encode_GUID_array_size);

    if (status != NV_ENC_SUCCESS)
    {
        return status;
    }
    assert(encode_GUID_array_size <= encode_GUID_count);
    codec_found = 0;

    for (i = 0; i < encode_GUID_array_size; i++)
    {
        if (input_codec_GUID == encodeGUIDArray[i])
        {
            codec_found = 1;
            break;
        }
    }

    if (codec_found) {
        return NV_ENC_SUCCESS;
    }
    else {
        return NV_ENC_ERR_INVALID_PARAM;
    }
}

NVENCSTATUS HWEncoder::NvEncOpenEncodeSessionEx(void* device,
                                                NV_ENC_DEVICE_TYPE device_type)
{
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open_sess_ex_params;
    memset(&open_sess_ex_params, 0, sizeof(open_sess_ex_params));
    SET_VER(open_sess_ex_params, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS);
    open_sess_ex_params.device = device;
    open_sess_ex_params.deviceType = device_type;
    open_sess_ex_params.apiVersion = NVENCAPI_VERSION;

    return encode_api_->nvEncOpenEncodeSessionEx(&open_sess_ex_params,
                                                 &encoder_iface_);
}

// Encoder API function wrappers follow
NVENCSTATUS HWEncoder::NvEncGetEncodeGUIDCount(uint32_t* encode_GUID_count)
{
    return encode_api_->nvEncGetEncodeGUIDCount(encoder_iface_,
                                                encode_GUID_count);
}

NVENCSTATUS HWEncoder::NvEncGetEncodeGUIDs(GUID* GUIDs, uint32_t guid_array_size,
                                           uint32_t* guid_count)
{
    return encode_api_->nvEncGetEncodeGUIDs(encoder_iface_, GUIDs,
                                            guid_array_size, guid_count);
}

NVENCSTATUS HWEncoder::NvEncGetInputFormatCount(GUID encode_GUID,
                                                uint32_t* input_fmt_count)
{
    return encode_api_->nvEncGetInputFormatCount(encoder_iface_, encode_GUID,
                                                 input_fmt_count);
}

NVENCSTATUS HWEncoder::NvEncGetInputFormats(GUID encode_GUID,
                                            NV_ENC_BUFFER_FORMAT* input_fmts,
                                            uint32_t input_fmt_array_size,
                                            uint32_t* input_fmt_count)
{
    return encode_api_->nvEncGetInputFormats(encoder_iface_, encode_GUID,
                                             input_fmts, input_fmt_array_size,
                                             input_fmt_count);
}

NVENCSTATUS HWEncoder::NvEncGetEncodeCaps(GUID encode_GUID,
                                          NV_ENC_CAPS_PARAM* caps_param,
                                          int* caps_val)
{
    return encode_api_->nvEncGetEncodeCaps(encoder_iface_, encode_GUID,
                                           caps_param, caps_val);
}

NVENCSTATUS HWEncoder::NvEncGetEncodePresetCount(GUID encode_GUID, uint32_t*
                                                 encode_preset_GUID_count)
{
    return encode_api_->nvEncGetEncodePresetCount(encoder_iface_, encode_GUID,
                                                  encode_preset_GUID_count);
}

NVENCSTATUS HWEncoder::NvEncGetEncodePresetGUIDs(GUID encode_GUID,
                                             GUID* preset_GUIDs,
                                             uint32_t guid_array_size,
                                             uint32_t* encode_preset_GUID_count)
{
    return encode_api_->nvEncGetEncodePresetGUIDs(encoder_iface_, encode_GUID,
                                                  preset_GUIDs, guid_array_size,
                                                  encode_preset_GUID_count);
}

NVENCSTATUS HWEncoder::NvEncGetEncodePresetConfig(GUID encode_GUID,
                                                  GUID preset_GUID,
                               NV_ENC_PRESET_CONFIG* preset_config)
{
    return encode_api_->nvEncGetEncodePresetConfig(encoder_iface_, encode_GUID,
                                                   preset_GUID, preset_config);
}

NVENCSTATUS HWEncoder::NvEncCreateInputBuffer(uint32_t width, uint32_t height,
    void** input_buffer, NV_ENC_BUFFER_FORMAT input_format)
{
    NV_ENC_CREATE_INPUT_BUFFER create_input_buffer_params;

    memset(&create_input_buffer_params, 0, sizeof(create_input_buffer_params));
    SET_VER(create_input_buffer_params, NV_ENC_CREATE_INPUT_BUFFER);

    create_input_buffer_params.width = width;
    create_input_buffer_params.height = height;
    create_input_buffer_params.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;
    create_input_buffer_params.bufferFmt = input_format;

    NVENCSTATUS status = NV_ENC_SUCCESS;
    status = encode_api_->nvEncCreateInputBuffer(encoder_iface_,
                                                   &create_input_buffer_params);
    if (status == NV_ENC_SUCCESS) {
        *input_buffer = create_input_buffer_params.inputBuffer;
    }

    return status;
}

NVENCSTATUS HWEncoder::NvEncDestroyInputBuffer(NV_ENC_INPUT_PTR input_buffer)
{
    if (input_buffer) {
        return encode_api_->nvEncDestroyInputBuffer(encoder_iface_,
                                                    input_buffer);
    }

    return NV_ENC_SUCCESS;
}

NVENCSTATUS HWEncoder::NvEncCreateBitstreamBuffer(uint32_t size,
                                                  void** bitstreamBuffer)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBufferParams;
    memset(&createBitstreamBufferParams, 0,
           sizeof(createBitstreamBufferParams));

    SET_VER(createBitstreamBufferParams, NV_ENC_CREATE_BITSTREAM_BUFFER);

    createBitstreamBufferParams.size = size;
    createBitstreamBufferParams.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;

    nvStatus = encode_api_->nvEncCreateBitstreamBuffer(encoder_iface_,
                                               &createBitstreamBufferParams);
    if (nvStatus == NV_ENC_SUCCESS)
    {
        *bitstreamBuffer = createBitstreamBufferParams.bitstreamBuffer;
    }
    return nvStatus;
}

NVENCSTATUS HWEncoder::NvEncDestroyBitstreamBuffer(NV_ENC_OUTPUT_PTR
                                                   bitstreamBuffer)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    if (bitstreamBuffer)
    {
        nvStatus = encode_api_->nvEncDestroyBitstreamBuffer(encoder_iface_,
                                                            bitstreamBuffer);
    }
    return nvStatus;
}

NVENCSTATUS HWEncoder::NvEncLockBitstream(NV_ENC_LOCK_BITSTREAM*
                                          lockBitstreamBufferParams)
{
    return encode_api_->nvEncLockBitstream(encoder_iface_,
                                           lockBitstreamBufferParams);
}

NVENCSTATUS HWEncoder::NvEncUnlockBitstream(NV_ENC_OUTPUT_PTR bitstreamBuffer)
{
    return encode_api_->nvEncUnlockBitstream(encoder_iface_, bitstreamBuffer);
}

NVENCSTATUS HWEncoder::NvEncLockInputBuffer(void* inputBuffer,
                                            void** bufferDataPtr,
                                            uint32_t* pitch)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_LOCK_INPUT_BUFFER lockInputBufferParams;
    memset(&lockInputBufferParams, 0, sizeof(lockInputBufferParams));
    SET_VER(lockInputBufferParams, NV_ENC_LOCK_INPUT_BUFFER);

    lockInputBufferParams.inputBuffer = inputBuffer;
    nvStatus = encode_api_->nvEncLockInputBuffer(encoder_iface_,
                                                 &lockInputBufferParams);
    if (nvStatus == NV_ENC_SUCCESS)
    {
        *bufferDataPtr = lockInputBufferParams.bufferDataPtr;
        *pitch = lockInputBufferParams.pitch;
    }
    return nvStatus;
}

NVENCSTATUS HWEncoder::NvEncUnlockInputBuffer(NV_ENC_INPUT_PTR inputBuffer)
{
    return encode_api_->nvEncUnlockInputBuffer(encoder_iface_, inputBuffer);
}

NVENCSTATUS HWEncoder::NvEncRegisterAsyncEvent(void** completionEvent)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_EVENT_PARAMS eventParams;

    memset(&eventParams, 0, sizeof(eventParams));
    SET_VER(eventParams, NV_ENC_EVENT_PARAMS);

#if defined(NV_WINDOWS)
    eventParams.completionEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
#else
    eventParams.completionEvent = NULL;
#endif

    nvStatus = encode_api_->nvEncRegisterAsyncEvent(encoder_iface_,
                                                    &eventParams);
    if (nvStatus == NV_ENC_SUCCESS)
    {
        *completionEvent = eventParams.completionEvent;
    }
    return nvStatus;
}

NVENCSTATUS HWEncoder::NvEncUnregisterAsyncEvent(void* completionEvent)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_EVENT_PARAMS eventParams;

    if (completionEvent)
    {
        memset(&eventParams, 0, sizeof(eventParams));
        SET_VER(eventParams, NV_ENC_EVENT_PARAMS);
        eventParams.completionEvent = completionEvent;
        nvStatus = encode_api_->nvEncUnregisterAsyncEvent(encoder_iface_,
                                                          &eventParams);
    }
    return nvStatus;
}

NVENCSTATUS HWEncoder::NvEncMapInputResource(void* registeredResource,
                                             void** mappedResource)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_MAP_INPUT_RESOURCE mapInputResParams;

    memset(&mapInputResParams, 0, sizeof(mapInputResParams));
    SET_VER(mapInputResParams, NV_ENC_MAP_INPUT_RESOURCE);
    mapInputResParams.registeredResource = registeredResource;
    nvStatus = encode_api_->nvEncMapInputResource(encoder_iface_,
                                                  &mapInputResParams);
    if (nvStatus != NV_ENC_SUCCESS)
    {
        *mappedResource = mapInputResParams.mappedResource;
    }

    return nvStatus;
}

NVENCSTATUS HWEncoder::NvEncUnmapInputResource(NV_ENC_INPUT_PTR
                                               mappedInputBuffer)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    if (mappedInputBuffer)
    {
        nvStatus = encode_api_->nvEncUnmapInputResource(encoder_iface_,
                                                        mappedInputBuffer);
    }

    return nvStatus;
}

// TO DO: There is a lot of book keeping we need to add
// before we destroy the encoder.
NVENCSTATUS HWEncoder::NvEncDestroyEncoder()
{
    return encode_api_->nvEncDestroyEncoder(encoder_iface_);
}

NVENCSTATUS HWEncoder::NvEncRegisterResource(
        NV_ENC_INPUT_RESOURCE_TYPE resourceType, void* resourceToRegister,
        uint32_t width, uint32_t height, uint32_t pitch,
        void** registeredResource)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_REGISTER_RESOURCE registerResParams;

    memset(&registerResParams, 0, sizeof(registerResParams));
    SET_VER(registerResParams, NV_ENC_REGISTER_RESOURCE);

    registerResParams.resourceType = resourceType;
    registerResParams.resourceToRegister = resourceToRegister;
    registerResParams.width = width;
    registerResParams.height = height;
    registerResParams.pitch = pitch;
    registerResParams.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12_PL;

    nvStatus = encode_api_->nvEncRegisterResource(encoder_iface_,
                                                  &registerResParams);
    if (nvStatus == NV_ENC_SUCCESS)
    {
        *registeredResource = registerResParams.registeredResource;
    }
    return nvStatus;
}

NVENCSTATUS HWEncoder::NvEncUnregisterResource(NV_ENC_REGISTERED_PTR
                                               registeredRes)
{
    return encode_api_->nvEncUnregisterResource(encoder_iface_, registeredRes);
}

NVENCSTATUS HWEncoder::NvEncEncodeFrame(EncodeBuffer *pEncodeBuffer,
                                        PictureCommand *encPicCommand,
                                        uint32_t width, uint32_t height,
                                        NV_ENC_PIC_STRUCT ePicStruct,
                                        int8_t *qpDeltaMapArray,
                                        uint32_t qpDeltaMapArraySize)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_PIC_PARAMS encPicParams;

    memset(&encPicParams, 0, sizeof(encPicParams));
    SET_VER(encPicParams, NV_ENC_PIC_PARAMS);


    encPicParams.inputBuffer = pEncodeBuffer->input_buf.input_surface;
    encPicParams.bufferFmt = pEncodeBuffer->input_buf.format;
    encPicParams.inputWidth = width;
    encPicParams.inputHeight = height;
    encPicParams.outputBitstream = pEncodeBuffer->output_buf.bitstream_buf;
    encPicParams.completionEvent = pEncodeBuffer->output_buf.output_event;
    encPicParams.inputTimeStamp = encode_count;
    encPicParams.pictureStruct = ePicStruct;
    encPicParams.qpDeltaMap = qpDeltaMapArray;
    encPicParams.qpDeltaMapSize = qpDeltaMapArraySize;


    if (encPicCommand) {
        if (encPicCommand->force_idr) {
            encPicParams.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
        }

        if (encPicCommand->force_intra_refresh) {
            if (codec == CodecType::NV_ENC_HEVC) {
                encPicParams.codecPicParams.hevcPicParams.
                        forceIntraRefreshWithFrameCnt =
                        encPicCommand->intra_ref_dur;
            }
            else {
                encPicParams.codecPicParams.h264PicParams.
                        forceIntraRefreshWithFrameCnt =
                        encPicCommand->intra_ref_dur;
            }
        }
    }

    NVENCSTATUS status = encode_api_->nvEncEncodePicture(encoder_iface_,
                                                         &encPicParams);
    // This is not an error.
    if (status == NV_ENC_ERR_NEED_MORE_INPUT) {
        pEncodeBuffer->output_buf.busy = true;
    }

    if (status != NV_ENC_SUCCESS && status != NV_ENC_ERR_NEED_MORE_INPUT) {
        return nvStatus;
    }

    encode_count++;
    return NV_ENC_SUCCESS;
}

NVENCSTATUS HWEncoder::NvEncFlushEncoderQueue(void *hEOSEvent)
{
    NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
    NV_ENC_PIC_PARAMS encPicParams;
    memset(&encPicParams, 0, sizeof(encPicParams));
    SET_VER(encPicParams, NV_ENC_PIC_PARAMS);
    encPicParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    encPicParams.completionEvent = hEOSEvent;
    return encode_api_->nvEncEncodePicture(encoder_iface_, &encPicParams);
}

SerializedInfo HWEncoder::process_output_buffer(const EncodeBuffer *buf)
{
    NVENCSTATUS status = NV_ENC_SUCCESS;
    SerializedInfo result = std::make_tuple(false, 0, NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_UNKNOWN);

    if (buf->output_buf.bitstream_buf == NULL &&
        buf->output_buf.eos_flag == false) {
        std::cout << "CRAP 1" << std::endl;
        return result;
    }

    if (buf->output_buf.wait_on_event == true) {
        if (!buf->output_buf.output_event) {
            std::cout << "CRAP 2" << std::endl;
            return result;
        }

#if defined(NV_WINDOWS)
        WaitForSingleObject(buf->output_buf.output_event, INFINITE);
#endif
    }

    if (buf->output_buf.eos_flag) {
        std::get<0>(result) = true;
        return result;
    }

    NV_ENC_LOCK_BITSTREAM lock_bitstr_data;
    memset(&lock_bitstr_data, 0, sizeof(lock_bitstr_data));
    SET_VER(lock_bitstr_data, NV_ENC_LOCK_BITSTREAM);
    lock_bitstr_data.outputBitstream = buf->output_buf.bitstream_buf;
    lock_bitstr_data.doNotWait = false;

    status = encode_api_->nvEncLockBitstream(encoder_iface_,
        &lock_bitstr_data);
    if (status == NV_ENC_SUCCESS) {
        output_file_.seekp(encoded_bytes_count_, std::ios_base::beg);
        output_file_.write(reinterpret_cast<char *>(lock_bitstr_data.bitstreamBufferPtr),
            lock_bitstr_data.bitstreamSizeInBytes);

        // Write the offset where this frame is.
        std::get<1>(result) = encoded_bytes_count_;
        std::get<2>(result) = lock_bitstr_data.pictureType;
        // Increment the count for the next frame offset
        encoded_bytes_count_ += lock_bitstr_data.bitstreamSizeInBytes;
        status = encode_api_->nvEncUnlockBitstream(encoder_iface_,
            buf->output_buf.bitstream_buf);
        if (status == NV_ENC_SUCCESS) {
            // Mark everything is good.
            std::get<0>(result) = true;
        }
        else {
            std::cout << "CRAP" << std::endl;
        }
    }

    return result;
}
