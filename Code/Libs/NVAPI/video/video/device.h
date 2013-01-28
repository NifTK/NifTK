
#pragma once
#ifndef LIBVIDEO_DEVICE_H_71D5A234EE5E44D996C19C2FC05E80C5
#define LIBVIDEO_DEVICE_H_71D5A234EE5E44D996C19C2FC05E80C5

#include <video/frame.h>
#include <video/dllexport.h>
#include <vector>


namespace video
{


class SDIDeviceImpl;
class SDIDevice;
class LIBVIDEO_DLL_EXPORTS SDIDevice
{
public:
	enum Type
	{
		INPUT,
		OUTPUT
	}				type;

protected:
	SDIDeviceImpl*	pimpl;

public:
	SDIDeviceImpl* get_pimpl();


private:
	SDIDevice();
	~SDIDevice();

private:
	// not implemented
	SDIDevice(const SDIDevice& copyme);
	SDIDevice& operator=(const SDIDevice& assignme);


public:
	// note: hardware limitation means that all streams have the same signal format
	StreamFormat get_format(int streamno);

	Type get_type() const;


private:
	static std::vector<SDIDevice*>		devices;

public:
	/**
	 * @throws std::runtime_error if the driver returns errors
	 * @returns either null ptr if no device, or a pointer
	 * @warning SDIDevice owns the returned pointer!
	 */
	static SDIDevice* get_device(int devno);
};

}

#endif // LIBVIDEO_DEVICE_H_71D5A234EE5E44D996C19C2FC05E80C5
