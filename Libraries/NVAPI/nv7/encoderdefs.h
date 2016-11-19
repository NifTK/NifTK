#ifndef ENCODER_DEFS_H__
#define ENCODER_DEFS_H__

#include <vector>
#include <tuple>
#include <cstdint>
#include "nvEncodeAPI.h"

typedef std::vector<std::pair<uint_least64_t, NV_ENC_PIC_TYPE> > NalType;
typedef std::tuple<bool, uint_least64_t, NV_ENC_PIC_TYPE> SerializedInfo;

#define BITSTREAM_BUFFER_SIZE 10 * 1024 * 1024

#define DEFAULT_I_QFACTOR -0.8f
#define DEFAULT_B_QFACTOR 1.25f
#define DEFAULT_I_QOFFSET 0.f
#define DEFAULT_B_QOFFSET 1.25f

#define NUM_ENCODE_QUEUE 16

#endif
