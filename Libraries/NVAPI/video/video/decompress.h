/*=============================================================================

  libvideo: a library for SDI video processing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#pragma once
#ifndef LIBVIDEO_DECOMPRESS_H_924E20500947462F889F5AB92DA2C0FF
#define LIBVIDEO_DECOMPRESS_H_924E20500947462F889F5AB92DA2C0FF

#include <video/dllexport.h>
#include <video/compress.h>
#include <string>


namespace video
{


#pragma warning(push)
#pragma warning(disable: 4275)      // non dll-interface class '...' used as base for dll-interface class '...'

class LIBVIDEO_DLL_EXPORTS DecompressorFailedException : public std::runtime_error
{
public:
    DecompressorFailedException(const std::string& msg, int errorcode = 1);
};

#pragma warning(pop)


// forward-decl
class DecompressorImpl;

/**
 * Use this to decompress the output of Compressor.
 * @warning Not a general-purpose video decompressor!
 */
class LIBVIDEO_DLL_EXPORTS Decompressor
{
private:
    // avoid header pollution
    DecompressorImpl*     pimpl;


#pragma warning(push)
#pragma warning(disable: 4251)      // class ... needs to have dll-interface to be used by clients of class ...

    std::string           filename;

#pragma warning(pop)


public:
    Decompressor(const std::string& filename);
    ~Decompressor();


private:
    // not implemented
    Decompressor(const Decompressor& copyme);
    Decompressor& operator=(const Decompressor& assignme);


public:
    // to be able to seek, or decompress an arbitrary frame, we need to know
    // where in the file the relevant data has been written to.
    void update_index(unsigned int frameno, unsigned __int64 offset, FrameType::FT type);

    // retrieve what is already in the index, or was recovered.
    bool get_index(unsigned int frameno, unsigned __int64* offset, FrameType::FT* type);

    // try to rebuild index by scanning the input file for start codes.
    bool recover_index();

    // decompresses the requested frame into the targettexture.
    // this is a blocking call: it will return once decompression has finished.
//  bool decompress(unsigned int frameno, GLuint targettexture);

    // buffer is RGBA (4 bytes per pixel).
    // buffersize is in bytes, pitch is bytes as well.
    // returns true if successfully decoded.
    // throws if the decoder barfs.
    bool decompress(unsigned int frameno, void* buffer, std::size_t buffersize, unsigned int bufferpitch);

    // FIXME: coded dimensions are different from display dimensions! which one to return?
    //        i would go for display dimension because that is what compressor takes as input.
    int get_width() const;
    int get_height() const;
};


} // namespace

#endif // LIBVIDEO_DECOMPRESS_H_924E20500947462F889F5AB92DA2C0FF
