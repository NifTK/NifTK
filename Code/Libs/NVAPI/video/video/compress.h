
#pragma once
#ifndef LIBVIDEO_COMPRESS_H_51C60B61F0E2444388408FB2BD1F737A
#define LIBVIDEO_COMPRESS_H_51C60B61F0E2444388408FB2BD1F737A

#include <string>
#include <exception>
#include <video/dllexport.h>


namespace video
{


#pragma warning(push)
#pragma warning(disable: 4275)      // non dll-interface class '...' used as base for dll-interface class '...'

// specifically the compressor engine returned error codes
//  for functions that should not fail during normal operations
class LIBVIDEO_DLL_EXPORTS CompressorFailedException : public std::runtime_error
{
public:
	CompressorFailedException(const std::string& msg, int errorcode = 1);
};

// functionality required to map buffers/textures/etc for compression failed
class LIBVIDEO_DLL_EXPORTS InteropFailedException : public std::runtime_error
{
public:
	InteropFailedException(const std::string& msg);
};

#pragma warning(pop)


// forward-decl for pimpl
class CompressorImpl;

/**
 * @warning The constructor of this class will query the current OpenGL rendering context
 * and the current CUDA context and store both internally. Every time you call a method
 * it will check whether these contexts are still current!
 *
 * @warning For the CUDA compression to work properly you need to make sure the CUDA
 * context that runs the compression stream is on the same device as the OpenGL context
 * that owns the textures! In debug mode, this class will check for this and barf if not!
 */
class LIBVIDEO_DLL_EXPORTS Compressor
{
private:
	CompressorImpl*		pimpl;

public:
	/**
	 * @param mfps Refresh rate in milli-Hz, e.g. for NTSC 29.97 Hz this would be 29970.
	 * @param width The input and output width of the video. Needs to be even! All formats in StreamFormat have even widths.
	 * @param height The input and output height of the video. If it's not even then this class automatically pads it. This is because NTSC has an odd height (487) and this class should be able to take it directly.
	 * @throws InteropFailedException if there are problems with CUDA
	 * @throws CompressorFailedException if the compressor engine specifically returns unexpected errors
	 * @throws std::logic_error if there's no CUDA or OpenGL context, or both are not on the same device
	 */
	Compressor(int width, int height, int mfps, const std::string& filename);	
	~Compressor();

private:
	// not implemented
	Compressor(const Compressor& copyme);
	Compressor& operator=(const Compressor& assignme);


public:

	/**
	 * @throws std::logic_error if you did not call preparetexture() beforehand
	 */
	void compresstexture(int gltexture);

	void preparetexture(int gltexture);
};


} // namespace

#endif // LIBVIDEO_COMPRESS_H_51C60B61F0E2444388408FB2BD1F737A
