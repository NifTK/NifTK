#include "stdafx.h"
#include <glutminitk/glutminitk.h>
#include <video/compress.h>
#include <video/decompress.h>
#include <ogltools/texture.h>
#include <ogltools/glewsupport.h>


class CompressorTestWindow : public glutminitk::Window, public glutminitk::Timer, ogltools::GLEWInitHelper
{
protected:
    video::Compressor*    compressor;
    CUcontext             cuContext;

    ogltools::Texture<boost::gil::rgba8_pixel_t, 2>     noiseinput[10];


public:
    CompressorTestWindow()
        : glutminitk::Window("CompressorTestWindow", 800, 600),
          compressor(0), cuContext(0)
    {
        // a bit of standard cuda/ogl setup
        int           cudadevices[10];
        unsigned int  actualcudadevices = 0;
        // note that zero is a valid device index
        std::memset(&cudadevices[0], -1, sizeof(cudadevices));
        if (cudaGLGetDevices(&actualcudadevices, &cudadevices[0], sizeof(cudadevices) / sizeof(cudadevices[0]), cudaGLDeviceListAll) != cudaSuccess)
            throw std::runtime_error("No CUDA devices for OpenGL context???");

        if (cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cudadevices[0]) != CUDA_SUCCESS)
            std::cerr << "Cannot create CUDA driver context!" << std::endl;

        boost::gil::rgba8_image_t   noiseimg(1920, 1080);
        for (int t = 0; t < (sizeof(noiseinput) / sizeof(noiseinput[0])); ++t)
        {
            for (int y = 0; y < noiseimg.height(); ++y)
                for (int x = 0; x < noiseimg.width(); ++x)
                {
                    boost::gil::rgba8_pixel_t& p = boost::gil::view(noiseimg)(x, y);
                    p[0] = ((float) std::rand() / (float) RAND_MAX) * 255;
                    p[1] = ((float) std::rand() / (float) RAND_MAX) * 255;
                    p[2] = ((float) std::rand() / (float) RAND_MAX) * 255;
                }
            noiseinput[t].redefine(boost::gil::const_view(noiseimg));
        }


        compressor = new video::Compressor(noiseimg.width(), noiseimg.height(), 25000, "testcompressor.264");
        for (int t = 0; t < (sizeof(noiseinput) / sizeof(noiseinput[0])); ++t)
            compressor->preparetexture(noiseinput[t]);

        // the compressor uses tsc for lock contention estimation
        // for this to be of any use we need to set thread affinity, otherwise we could get totally random tscs from different cores
        SetThreadAffinityMask(GetCurrentThread(), 0x01);

        register_timer(this, 1);
    }

    void kill_compressor()
    {
        std::ofstream   offsetfile("offset.txt");
        for (unsigned int i = 0; ; ++i)
        {
            unsigned __int64        offset = 0;
            video::FrameType::FT    type;

            bool ok = compressor->get_output_info(i, &offset, &type);
            if (!ok)
                break;
            offsetfile << i << ' ' << offset << ' ' << ((int) type) << std::endl;
        }
        offsetfile.close();
        delete compressor;
    }

    ~CompressorTestWindow()
    {
        if (compressor)
            kill_compressor();
    }


protected:
    virtual void keyboard(unsigned char key, int x, int y)
    {
    }

    virtual void elapsed()
    {
    //  for (int i = 0; i < (sizeof(noiseinput) / sizeof(noiseinput[0])); ++i)
        {
            int j = std::rand() % (sizeof(noiseinput) / sizeof(noiseinput[0]));
            compressor->compresstexture(noiseinput[j]);
        }

        register_timer(this, 1);
    }

    virtual void display()
    {
    }
};


int main(int argc, char* argv[])
{
    CompressorTestWindow* wnd = new CompressorTestWindow;
    wnd->runloop();

    return 0;
}
