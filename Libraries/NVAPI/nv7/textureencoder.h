#ifndef TEXTURE_ENCODER_H__
#define TEXTURE_ENCODER_H__

#include "macros.h"
#include "encoderdefs.h"
#include "hwencoder.h"
#include "dynlink_nvcuvid.h"
#include <memory>

class TextureEncoder
{
public:
    TextureEncoder();
    virtual ~TextureEncoder();

    /**
     * @brief initialize_encoder
     * @param width
     * @param height
     * @param dev_ID
     * @param fps
     * @param output_file
     * @return
     */
    bool initialize_encoder(uint32_t width, uint32_t height, uint32_t dev_ID,
                            int fps, const std::string & output_file);
private:

	/**
	* @brief init_CUDA
	* @param deviceID
	* @return
	*/
	bool init_CUDA(uint32_t deviceID);

    /**
     * @brief allocate_io_buffers
     * @param format
     * @return
     */
    bool allocate_io_buffers(NV_ENC_BUFFER_FORMAT format);
    /**
     * @brief release_io_buffers
     * @return
     */
    bool release_io_buffers();

    /**
     * @brief process_compressed_frames
     */
    void process_compressed_frames();

    /**
     * @brief deinitialize_encoder
     */
    void deinitialize_encoder();

    std::unique_ptr<HWEncoder>                          hw_encoder_;
    CUdeviceptr                                         frame_buffer_;
    CUvideoctxlock                                      ctx_lock_;
    NalType												output_nal_;

    // Pointer to client device
    void *												device_ptr_;
    uint32_t											width_;
    uint32_t											height_;

    // Encoder configuration
    EncodeConfig										encode_config_;

    bool												alive_;

    // No copy semantics
    DISALLOW_COPY_AND_ASSIGNMENT(TextureEncoder);
};

#endif
