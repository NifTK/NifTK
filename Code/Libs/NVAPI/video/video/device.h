/*=============================================================================

  libvideo: a library for SDI video processing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
    }               type;

protected:
    SDIDeviceImpl*  pimpl;

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

    // returns a technical string of what actually comes off the wire, irrespective of what StreamFormat thinks it is.
    // note that the return value is a statically allocated read-only string; do not free!
    // only valid after probing with get_format()!
    const char* get_wireformat();

    Type get_type() const;


#pragma warning(push)
#pragma warning(disable: 4251)      //  class '...' needs to have dll-interface to be used by clients of class '...'

private:
    static std::vector<SDIDevice*>      devices;

#pragma warning(pop)


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
