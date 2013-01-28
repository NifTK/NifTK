#pragma once
#ifndef LIBVIDEO_SDIINPUT_H_3CD0DE9000FA4846B097E35979071A3B
#define LIBVIDEO_SDIINPUT_H_3CD0DE9000FA4846B097E35979071A3B

#include <video/device.h>
#include <video/frame.h>
#include <video/dllexport.h>
#include <string>


namespace video
{


class SDIInputImpl;
class LIBVIDEO_DLL_EXPORTS SDIInput
{
private:

protected:
	SDIInputImpl*		pimpl;


public:
	SDIInput(SDIDevice* dev);
	~SDIInput();

private:
	// not implemented
	SDIInput(const SDIInput& copyme);
	SDIInput& operator=(const SDIInput& assignme);


public:
	// format is always RGBA!
	// returns zero if no stream with that index
	int get_texture_id(int streamno);

	/**
	 * @brief Transfers one set of frames over all streams into texture objects.
	 * @throws std::runtime_error if capture setup has become invalid
	 * @post Texture binding on the currently active unit will have changed
	 * @detail This method will actively try to prevent the system from entering any power-savings mode!
	 */
	FrameInfo capture();


	/**
	 * @warning Caveat: this will not throw if the capture state has become invalid,
	 * it will simply continue returning false! So if too much time has passed since last frame
	 * you may want to call capture() anyway to see whether there still is anything connected.
	 */
	bool has_frame();


protected:
	std::string		logfilename;

public:
	void set_log_filename(const std::string& fn);
};


} // namespace

#endif // LIBVIDEO_SDIINPUT_H_3CD0DE9000FA4846B097E35979071A3B
