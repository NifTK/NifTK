#ifndef NAL_FILE_GENERATOR__
#define NAL_FILE_GENERATOR__

#include <fstream>
#include "macros.h"
#include "nvEncodeAPI.h"
#include "dynlink_cuda.h"
#include "dynlink_nvcuvid.h"

class NalFileGenerator
{
public:
    NalFileGenerator();
    ~NalFileGenerator();
    bool generate_indexes(const std::string & path);
    bool compare_video_sequence(CUVIDEOFORMAT * format);
    bool parse_video_data(CUVIDSOURCEDATAPACKET * packet);

    CUvideodecoder                      video_decoder_;

private:
    std::ifstream						h264_file_;
    CUvideosource                       video_source_;
    CUvideoparser                       video_parser_;
    CUVIDDECODECREATEINFO               decoder_info_;
    CUVIDEOFORMAT						format_;
    CUcontext							cu_context_;
    CUvideoctxlock						ctx_lock_;

    bool init_CUDA(uint32_t deviceID);

    DISALLOW_COPY_AND_ASSIGNMENT(NalFileGenerator);
};

#endif
