#include "stdafx.h"
#include <glutminitk/glutminitk.h>
#include <video/compress.h>
#include <video/decompress.h>
#include <ogltools/texture.h>
#include <ogltools/glewsupport.h>
#include <boost/typeof/typeof.hpp>
#include <boost/gil/extension/io/png_io.hpp>


class DecompressorTestWindow : public glutminitk::Window, public glutminitk::Timer, ogltools::GLEWInitHelper
{
protected:
    video::Compressor*    compressor;
    CUcontext             cuContext;

    unsigned int                                        compressedframes;
    ogltools::Texture<boost::gil::rgba8_pixel_t, 2>     testinput;

public:
    DecompressorTestWindow()
        : glutminitk::Window("DecompressorTestWindow", 800, 600),
          compressor(0), cuContext(0), compressedframes(0), testinput(1920, 1080)
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


        compressor = new video::Compressor(testinput.width(), testinput.height(), 25000, "testcompressor.264");
        compressor->preparetexture(testinput);

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
        compressor = 0;
    }

    ~DecompressorTestWindow()
    {
        if (compressor)
            kill_compressor();
    }

    void generate_testdata(unsigned int frame, boost::gil::rgba8_view_t output)
    {
        std::srand(0);
        int   c = 0;
        for (int i = 0; i < frame; ++i)
            c = std::rand() & 0xFF;

        boost::gil::fill_pixels(output, boost::gil::rgba8_pixel_t(c, 255 - c, c / 2, 255));
    }

    void test_decomp(bool rebuildindex)
    {
        video::Decompressor*    decomp = new video::Decompressor("testcompressor.264");
        if (!rebuildindex)
        {
            std::ifstream   offsetfile("offset.txt");
            while (offsetfile.good())
            {
                unsigned int            frameno = -1;
                unsigned __int64        offset = 0;
                int                     type;

                offsetfile >> frameno;
                offsetfile >> offset;
                offsetfile >> type;

                if (frameno == -1)
                    break;

                decomp->update_index(frameno, offset, (video::FrameType::FT) type);
            }
        }
        else
            decomp->recover_index();

        assert(decomp->get_width()  == testinput.width());
        assert(decomp->get_height() == testinput.height());

        std::srand(0);
        // we decompress frames in an arbitrary order
        unsigned int    frames2decompress[60];
        for (int i = 0; i < (sizeof(frames2decompress) / sizeof(frames2decompress[0])); ++i)
            // note: potentially decompress more frames than there are in the file!
            frames2decompress[i] = ((float) std::rand() / (float) RAND_MAX) * 61;

        boost::gil::rgba8_image_t   result(decomp->get_width(), decomp->get_height());
        for (int i = 0; i < (sizeof(frames2decompress) / sizeof(frames2decompress[0])); ++i)
        {
            unsigned int  f = frames2decompress[i];

            void*         buf   = &boost::gil::view(result)(0, 0);
            unsigned int  pitch = (char*) &boost::gil::view(result)(0, 1) - (char*) &boost::gil::view(result)(0, 0);
            std::size_t   size  = 4 + (char*) &boost::gil::view(result)(result.width() - 1, result.height() - 1) - (char*) &boost::gil::view(result)(0, 0);

            bool ok = decomp->decompress(f, buf, size, pitch);
            if (ok)
            {
                boost::gil::rgba8_image_t   expected(result.width(), result.height());
                generate_testdata(f, boost::gil::view(expected));
                // difference between all pixels
                unsigned int    diff = 0;
                for (int y = 0; y < result.height(); ++y)
                    for (int x = 0; x < result.width(); ++x)
                    {
                        const BOOST_AUTO(& r, boost::gil::const_view(result)(x, y));
                        const BOOST_AUTO(& e, boost::gil::const_view(expected)(x, y));

                        diff += std::abs(r[0] - e[0]);
                        diff += std::abs(r[1] - e[1]);
                        diff += std::abs(r[2] - e[2]);
                    }
                std::cerr << "Max diff for frame " << f << ": " << diff << std::endl;

                std::ostringstream  filename;
                filename << "testcompressor-decoded-f=" << f << ".png";
                boost::gil::png_write_view(filename.str().c_str(), boost::gil::const_view(result));
            }
            else
                std::cerr << "Could not decompress frame " << f << std::endl;
        }

        delete decomp;
    }


protected:

    virtual void elapsed()
    {
        boost::gil::rgba8_image_t   testimg(testinput.width(), testinput.height());
        generate_testdata(compressedframes, boost::gil::view(testimg));
        testinput.upload(boost::gil::const_view(testimg));
        // FIXME: necessary?
        glFinish();

        compressor->compresstexture(testinput);
        ++compressedframes;
        std::cerr << '.';

        // the default idr period is 15 frames.
        // so lets compress 4 chunks of idr: 4 * 15 = 60.
        if (compressedframes > 60)
        {
            std::cerr << std::endl;
            kill_compressor();

            try
            {
                test_decomp(true);
            }
            catch (const std::exception& e)
            {
                std::cerr << "Exception: decompression failed: " << e.what() << std::endl;
            }


            try
            {
                test_decomp(false);
            }
            catch (const std::exception& e)
            {
                std::cerr << "Exception: decompression failed: " << e.what() << std::endl;
            }

            // we are done, one way or the other.
            selfdestroy();
        }

        register_timer(this, 1);
    }

    virtual void display()
    {
    }
};


int main(int argc, char* argv[])
{
    DecompressorTestWindow* wnd = new DecompressorTestWindow;
    wnd->runloop();

    return 0;
}
