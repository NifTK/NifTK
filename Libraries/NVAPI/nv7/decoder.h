#ifndef DECODER_H__
#define DECODER_H__

#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
#include <map>
#include "nvEncodeAPI.h"
#include "dynlink_cuda.h"
#include "dynlink_nvcuvid.h"
#include "cudamodulemgr.h"
#include "cudalloc.h"
#include "cudaprocessframe.h"
#include "macros.h"

typedef std::vector<std::pair<uint_least64_t, NV_ENC_PIC_TYPE> > NalType;

// Here for legacy reasons
enum OldFrameTypes
{
    I = 1,
    P = 2,
    B = 3
};


class Decoder
{
public:
    /**
     * @brief Decoder
     */
    Decoder(int device_id);

    /**
     * @brief ~Decoder
     */
    virtual ~Decoder();

    /**
     * @brief set_legacy
     * @param legacy
     */
    void set_legacy(bool legacy);

    /**
     * @brief initialize_decoder
     * @param path
     * @param ctx_lock
     * @param target_width
     * @param target_height
     */
    bool initialize_decoder(const std::string & path);

    /**
    * @brief parse_video_data
    * Called by the callback function
    * @param packet
    */
    bool parse_video_data(CUVIDSOURCEDATAPACKET * packet);

    /**
    * @brief compare_video_sequence
    * Called by the callback function
    * @param format
    */
    bool compare_video_sequence(CUVIDEOFORMAT * format);

    /**
    * @brief get_dims
    */
    std::pair<int, int> get_dims() const;

    /**
     * @brief update_index
     * @param frame_number
     * @param offset
     * @param type
     */
    void update_index(unsigned int frame_number, uint_least64_t offset,
        NV_ENC_PIC_TYPE type);

    /**
     * @brief decompress
     * @param frame_number
     * @param buffer
     * @param buffer_size
     * @param buffer_pitch
     * @return
     */
    bool decompress(unsigned int frame_number, void * buffer,
                    std::size_t buffer_size, size_t buffer_pitch);

    /**
     * @brief cuda_process_frame
     * @param decoded_frame
     * @param decoded_pitch
     * @param dest_data
     * @param dest_pitch
     */
    void cuda_process_frame(CUdeviceptr * decoded_frame, size_t decoded_pitch,
        CUdeviceptr * dest_data, size_t dest_pitch);

    unsigned int						requested_frame_;
    unsigned int						session_frame_count;
    CUVIDEOFORMAT						format_;
    CUvideodecoder                      video_decoder_;
    std::pair<int, int>					dims_;
    void*								target_cpu_buffer_;
    std::size_t 						target_cpu_pitch_;
    std::size_t							target_cpu_size_;
    bool								done_requested_frame_;
    CUcontext							cu_context_;
    CUvideoctxlock						ctx_lock_;

private:
    CUvideoparser                       video_parser_;
    CUVIDDECODECREATEINFO               decoder_info_;
    NalType								indexes_;
    std::vector<unsigned char>			packet_buffer_;
    std::string							file_name_;
    std::ifstream						h264_file_;
    CUmoduleManager *					cuda_module_;
    CUfunction							kernelNV12toARGB_;
    eColorSpace							color_space_;
    float								hue_;
    bool                                legacy_;

    bool init_CUDA(int deviceID);
    void submit_frame(unsigned int frame_number, unsigned int frame_base);

    DISALLOW_COPY_AND_ASSIGNMENT(Decoder);
};

#endif
