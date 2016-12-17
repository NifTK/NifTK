#ifndef ENCODER_H__
#define ENCODER_H__

#include "macros.h"
#include "encoderdefs.h"
#include "hwencoder.h"
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <thread>

namespace notstd {
    template<class Mutex>
    std::unique_lock<Mutex> get_lock(Mutex & m)
    {
        return std::unique_lock<Mutex>(m);
    }
}

class Encoder
{
public:
    Encoder();
    virtual ~Encoder();

    /**
     * @brief initialize_encoder
     * @param width
     * @param height
     * @param dev_ID
     * @param output_file
     * @return
     */
    bool initialize_encoder(uint32_t width, uint32_t height, uint32_t dev_ID,
                            int fps, const std::string & output_file);
    /**
     * @brief encode_frame
     * @param argb
     * @return
     */
    bool encode_frame(unsigned char * argb);

    /**
    * @brief flush_encoder
    * @return
    */
    bool flush_encoder();

    bool get_output_info(unsigned int frame_number, uint_least64_t & fileoffset,
        NV_ENC_PIC_TYPE & frame_type);

	bool write_nal_to_file(const std::string & file);

	void clear_nals();

private:
    /**
     * @brief allocate_io_buffers
     * @param width
     * @param height
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
     * @brief init_CUDA
     * @param deviceID
     * @return
     */
    bool init_CUDA(uint32_t deviceID);

    /**
     * @brief deinitialize_encoder
     */
    void deinitialize_encoder();

    /**
    * @brief get_available
    * @return
    */
    EncodeBuffer * get_available();

    /**
    * @brief add_for_serialization
    * @param buf
    * @return
    */
    void add_for_serialization(EncodeBuffer * buf);

    /**
    * @brief add_for_encoding
    * @param buf
    * @return
    */
    void add_for_encoding(EncodeBuffer * buf);

    /**
    * @brief get_next_for_serialization
    * @return
    */
    EncodeBuffer * get_next_for_serialization();

    /**
    * @brief process_comnpressed_frames
    */
    void process_compressed_frames();

    // Member variables follow
    std::unique_ptr<HWEncoder>							hw_encoder_;
    std::vector<EncodeBuffer *>							allocated_buffers_;

    std::queue<EncodeBuffer *>							available_buffers_;
    std::queue<EncodeBuffer *>							serializable_buffers_;
    std::queue<EncodeBuffer *>							busy_buffers_;

    uint32_t											buffer_count_;
    EncodeOutputBuffer									eos_output_buffer_;

    // Pointer to client device
    void *												device_ptr_;
    uint32_t											width_;
    uint32_t											height_;

    // Encoder configuration
    EncodeConfig										encode_config_;

    // Multithreading bits
    std::mutex											mutex_;
    std::thread											process_thread_;
    bool												alive_;

    NalType												output_nal_;

    // No copy semantics
    DISALLOW_COPY_AND_ASSIGNMENT(Encoder);
};

#endif
