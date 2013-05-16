/*=============================================================================

  libvideo: a library for SDI video processing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "stdafx.h"
#include <video/device.h>
#include "deviceimpl.h"


namespace video
{


SDIDevice::SDIDevice()
    : pimpl(new SDIDeviceImpl)
{
    pimpl->wireformat = NVVIOSIGNALFORMAT_NONE;
}

SDIDevice::~SDIDevice()
{
    delete pimpl;
}


static bool map_nvformat_to_interlaced(NVVIOSIGNALFORMAT signalformat)
{
    switch (signalformat)
    {
        case NVVIOSIGNALFORMAT_NONE:
            return false;

        case NVVIOSIGNALFORMAT_487I_59_94_SMPTE259_NTSC:
        case NVVIOSIGNALFORMAT_576I_50_00_SMPTE259_PAL:
        case NVVIOSIGNALFORMAT_1080I_59_94_SMPTE274:
            return true;

        case NVVIOSIGNALFORMAT_720P_60_00_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_59_94_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_50_00_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_30_00_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_29_97_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_25_00_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_24_00_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_23_98_SMPTE296:
        case NVVIOSIGNALFORMAT_1080P_30_00_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_29_97_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_25_00_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_24_00_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_23_976_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_50_00_SMPTE274_3G_LEVEL_A:
        case NVVIOSIGNALFORMAT_1080P_59_94_SMPTE274_3G_LEVEL_A:
        case NVVIOSIGNALFORMAT_1080P_60_00_SMPTE274_3G_LEVEL_A:
        case NVVIOSIGNALFORMAT_1080P_60_00_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_50_00_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_30_00_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_25_00_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_24_00_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_59_94_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_29_97_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_23_98_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_2048P_30_00_SMPTE372:
            return false;

        // FIXNE: fill in all the others
    }

    return false;
}

static StreamFormat::RefreshRate map_nvformat_to_rr(NVVIOSIGNALFORMAT signalformat)
{
    switch (signalformat)
    {
        case NVVIOSIGNALFORMAT_NONE:
            return StreamFormat::RR_NONE;

        // interlaced at twice framerate (what a mess)
        case NVVIOSIGNALFORMAT_487I_59_94_SMPTE259_NTSC:
        case NVVIOSIGNALFORMAT_1080I_59_94_SMPTE274:
            return StreamFormat::RR_29_97;
        case NVVIOSIGNALFORMAT_576I_50_00_SMPTE259_PAL:
            return StreamFormat::RR_25;

        case NVVIOSIGNALFORMAT_720P_50_00_SMPTE296:
        case NVVIOSIGNALFORMAT_1080P_50_00_SMPTE274_3G_LEVEL_A:
        case NVVIOSIGNALFORMAT_1080P_50_00_SMPTE274_3G_LEVEL_B:
            return StreamFormat::RR_50;

        case NVVIOSIGNALFORMAT_720P_30_00_SMPTE296:
        case NVVIOSIGNALFORMAT_1080P_30_00_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_30_00_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_2048P_30_00_SMPTE372:
            return StreamFormat::RR_30;

        case NVVIOSIGNALFORMAT_720P_60_00_SMPTE296:
        case NVVIOSIGNALFORMAT_1080P_60_00_SMPTE274_3G_LEVEL_A:
        case NVVIOSIGNALFORMAT_1080P_60_00_SMPTE274_3G_LEVEL_B:

        case NVVIOSIGNALFORMAT_720P_59_94_SMPTE296:
        case NVVIOSIGNALFORMAT_1080P_59_94_SMPTE274_3G_LEVEL_A:
        case NVVIOSIGNALFORMAT_1080P_59_94_SMPTE274_3G_LEVEL_B:

        case NVVIOSIGNALFORMAT_720P_29_97_SMPTE296:
        case NVVIOSIGNALFORMAT_1080P_29_97_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_29_97_SMPTE274_3G_LEVEL_B:

        case NVVIOSIGNALFORMAT_720P_25_00_SMPTE296:
        case NVVIOSIGNALFORMAT_1080P_25_00_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_25_00_SMPTE274_3G_LEVEL_B:

        case NVVIOSIGNALFORMAT_720P_24_00_SMPTE296:
        case NVVIOSIGNALFORMAT_1080P_24_00_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_24_00_SMPTE274_3G_LEVEL_B:

        case NVVIOSIGNALFORMAT_720P_23_98_SMPTE296:
        case NVVIOSIGNALFORMAT_1080P_23_98_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_23_976_SMPTE274:

        // FIXNE: fill in all the others
        default:
            return StreamFormat::RR_NONE;
    }

    return StreamFormat::RR_NONE;
}


static StreamFormat::PictureFormat map_nvformat_to_pf(NVVIOSIGNALFORMAT signalformat)
{
    switch (signalformat)
    {
        case NVVIOSIGNALFORMAT_NONE:
            return StreamFormat::PF_NONE;

        case NVVIOSIGNALFORMAT_487I_59_94_SMPTE259_NTSC:
            return StreamFormat::PF_487;
        case NVVIOSIGNALFORMAT_576I_50_00_SMPTE259_PAL:
            return StreamFormat::PF_576;

        case NVVIOSIGNALFORMAT_720P_60_00_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_59_94_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_50_00_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_30_00_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_29_97_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_25_00_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_24_00_SMPTE296:
        case NVVIOSIGNALFORMAT_720P_23_98_SMPTE296:
            return StreamFormat::PF_720;

        case NVVIOSIGNALFORMAT_1080P_30_00_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_29_97_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_25_00_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_24_00_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_23_976_SMPTE274:
        case NVVIOSIGNALFORMAT_1080P_50_00_SMPTE274_3G_LEVEL_A:
        case NVVIOSIGNALFORMAT_1080P_59_94_SMPTE274_3G_LEVEL_A:
        case NVVIOSIGNALFORMAT_1080P_60_00_SMPTE274_3G_LEVEL_A:
        case NVVIOSIGNALFORMAT_1080P_60_00_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_50_00_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_30_00_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_25_00_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_24_00_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_59_94_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_29_97_SMPTE274_3G_LEVEL_B:
        case NVVIOSIGNALFORMAT_1080P_23_98_SMPTE274_3G_LEVEL_B:
        // not 100% sure what to do about the interlaced format here...
        case NVVIOSIGNALFORMAT_1080I_59_94_SMPTE274:
            return StreamFormat::PF_1080;

        case NVVIOSIGNALFORMAT_2048P_30_00_SMPTE372:
            return StreamFormat::PF_2048;

        // FIXNE: fill in all the others
    }

    return StreamFormat::PF_NONE;
}


SDIDevice::Type SDIDevice::get_type() const
{
    return type;
}

SDIDeviceImpl* SDIDevice::get_pimpl()
{
    return pimpl;
}

StreamFormat SDIDevice::get_format(int streamno)
{
    // device status is volatile
    //  meaning it can change all the time
    //  e.g. when user unplugs the cables
    // this returns meaningful data only for the input card
    NVVIOSTATUS status = {0};
    status.version = NVVIOSTATUS_VER;
    if (NvAPI_VIO_Status(pimpl->handle, &status) != NVAPI_OK)
        throw std::runtime_error("Unable to get input device status");

    // is it ok to call this for the output card?
    NVVIOCONFIG config = {0};
    config.version = NVVIOCONFIG_VER;
    // even though this says _IN below, this only works for the output card
    // for input the only usable value returned by the driver is what type it is
    // and even though it's _IN the driver will return _OUT type correctly
    //  but vice-versa is not true! specifying _OUT for the input card will result in garbage
    config.nvvioConfigType = NVVIOCONFIGTYPE_IN;
    config.fields = NVVIOCONFIG_SIGNALFORMAT | NVVIOCONFIG_DATAFORMAT | NVVIOCONFIG_STREAMS;
    // WARNING: for input almost everything returned is bogus! need to check with NVVIOSTATUS above!
    if (NvAPI_VIO_GetConfig(pimpl->handle, &config) != NVAPI_OK)
        throw std::runtime_error("Unable to query stream configuration");

    // lets check if we and the driver agree on what type of card we have here
    switch (config.nvvioConfigType)
    {
        case NVVIOCONFIGTYPE_IN:
        {
            assert(this->type == INPUT);

            // number of streams reported in config is bogus
            // we have to count ourselfs
            int                 streamcount = 0;
            // used for debugging: currently all streams on the nvidia card have to have the same format
            NVVIOSIGNALFORMAT   foundformat = NVVIOSIGNALFORMAT_NONE;
            for (int i = 0; i < NVAPI_MAX_VIO_JACKS; ++i)
            {
                // also note: there cannot be two independent streams on the same jack
                // so if there's a signal on the first channel of a jack then any other signal on the second channel
                //  is part of the dual-link(?) stream on that jack
                if (status.vioStatus.inStatus.vidIn[i][0].signalFormat != NVVIOSIGNALFORMAT_NONE)
                {
                    ++streamcount;
                    if (foundformat == NVVIOSIGNALFORMAT_NONE)
                        foundformat = status.vioStatus.inStatus.vidIn[i][0].signalFormat;
                    assert(foundformat == status.vioStatus.inStatus.vidIn[i][0].signalFormat);
                }
            }

            // we store this for later
            pimpl->wireformat = foundformat;

            if (streamno >= streamcount)
                return StreamFormat();
            // FIXME: this should use the actual format of the stream (despite the above mentioned limitation, which could go away with newer hardware revisions)
            return StreamFormat(map_nvformat_to_pf(foundformat), map_nvformat_to_rr(foundformat), map_nvformat_to_interlaced(foundformat));
        }
        case NVVIOCONFIGTYPE_OUT:
            assert(this->type == OUTPUT);
            // currently there's only one output format (both possible streams have to have same format)
            // FIXME: untested what this would return if we do custom-sdi-out with two streams
            if (streamno >= 1)
                return StreamFormat();
            return StreamFormat(map_nvformat_to_pf(config.vioConfig.outConfig.signalFormat), map_nvformat_to_rr(config.vioConfig.outConfig.signalFormat), map_nvformat_to_interlaced(config.vioConfig.outConfig.signalFormat));
    }

    return StreamFormat();
}


static const char*  wireformat_stringtable[] =
{
    "NONE",
    "487I_59_94_SMPTE259_NTSC",
    "576I_50_00_SMPTE259_PAL",
    "1035I_60_00_SMPTE260",
    "1035I_59_94_SMPTE260",
    "1080I_50_00_SMPTE295",
    "1080I_60_00_SMPTE274",
    "1080I_59_94_SMPTE274",
    "1080I_50_00_SMPTE274",
    "1080P_30_00_SMPTE274",
    "1080P_29_97_SMPTE274",
    "1080P_25_00_SMPTE274",
    "1080P_24_00_SMPTE274",
    "1080P_23_976_SMPTE274",
    "720P_60_00_SMPTE296",
    "720P_59_94_SMPTE296",
    "720P_50_00_SMPTE296",
    "1080I_48_00_SMPTE274",
    "1080I_47_96_SMPTE274",
    "720P_30_00_SMPTE296",
    "720P_29_97_SMPTE296",
    "720P_25_00_SMPTE296",
    "720P_24_00_SMPTE296",
    "720P_23_98_SMPTE296",
    "2048P_30_00_SMPTE372",
    "2048P_29_97_SMPTE372",
    "2048I_60_00_SMPTE372",
    "2048I_59_94_SMPTE372",
    "2048P_25_00_SMPTE372",
    "2048I_50_00_SMPTE372",
    "2048P_24_00_SMPTE372",
    "2048P_23_98_SMPTE372",
    "2048I_48_00_SMPTE372",
    "2048I_47_96_SMPTE372",
    "1080PSF_25_00_SMPTE274",
    "1080PSF_29_97_SMPTE274",
    "1080PSF_30_00_SMPTE274",
    "1080PSF_24_00_SMPTE274",
    "1080PSF_23_98_SMPTE274",
    "1080P_50_00_SMPTE274_3G_LEVEL_A",
    "1080P_59_94_SMPTE274_3G_LEVEL_A",
    "1080P_60_00_SMPTE274_3G_LEVEL_A",
    "1080P_60_00_SMPTE274_3G_LEVEL_B",
    "1080I_60_00_SMPTE274_3G_LEVEL_B",
    "2048I_60_00_SMPTE372_3G_LEVEL_B",
    "1080P_50_00_SMPTE274_3G_LEVEL_B",
    "1080I_50_00_SMPTE274_3G_LEVEL_B",
    "2048I_50_00_SMPTE372_3G_LEVEL_B",
    "1080P_30_00_SMPTE274_3G_LEVEL_B",
    "2048P_30_00_SMPTE372_3G_LEVEL_B",
    "1080P_25_00_SMPTE274_3G_LEVEL_B",
    "2048P_25_00_SMPTE372_3G_LEVEL_B",
    "1080P_24_00_SMPTE274_3G_LEVEL_B",
    "2048P_24_00_SMPTE372_3G_LEVEL_B",
    "1080I_48_00_SMPTE274_3G_LEVEL_B",
    "2048I_48_00_SMPTE372_3G_LEVEL_B",
    "1080P_59_94_SMPTE274_3G_LEVEL_B",
    "1080I_59_94_SMPTE274_3G_LEVEL_B",
    "2048I_59_94_SMPTE372_3G_LEVEL_B",
    "1080P_29_97_SMPTE274_3G_LEVEL_B",
    "2048P_29_97_SMPTE372_3G_LEVEL_B",
    "1080P_23_98_SMPTE274_3G_LEVEL_B",
    "2048P_23_98_SMPTE372_3G_LEVEL_B",
    "1080I_47_96_SMPTE274_3G_LEVEL_B",
    "2048I_47_96_SMPTE372_3G_LEVEL_B",
    // this one last, get_wireformat() will clamp to here
    "Unknown/invalid"
};

const char* SDIDevice::get_wireformat()
{
    return wireformat_stringtable[std::min((int) pimpl->wireformat, (int) (sizeof(wireformat_stringtable) / sizeof(wireformat_stringtable[0]) - 1))];
}



std::vector<SDIDevice*>     SDIDevice::devices;

SDIDevice* SDIDevice::get_device(int devno)
{
    // we already have enumerated all suitable hardware...
    if (!devices.empty())
    {
        // ...so just pick one from the list
        if (devno >= devices.size())
            return 0;
        return devices[devno];
    }

    // no hardware there (yet)
    // could either mean there's just none in the user's machine
    //  or we haven't tried enum yet


    // we never unload it, so no need to worry about nesting, etc
    if (NvAPI_Initialize() != NVAPI_OK)
        throw std::runtime_error("Failed to initialise NvAPI");


    NvAPI_ShortString nvapi_version;
    NvAPI_GetInterfaceVersionString(nvapi_version);

    NV_DISPLAY_DRIVER_VERSION driver_version = {0};
    driver_version.version = NV_DISPLAY_DRIVER_VERSION_VER;
    NvAPI_GetDisplayDriverVersion(NVAPI_DEFAULT_HANDLE, &driver_version);

    std::cout << "NvAPI version: " << nvapi_version << std::endl;
    std::cout << "Driver version: " << driver_version.drvVersion << " (build " << driver_version.bldChangeListNum << ")" << std::endl;

    // find out how many suitable capture cards there are
    NVVIOTOPOLOGY topology;
    std::memset(&topology, 0, sizeof(topology));
    topology.version = NVVIOTOPOLOGY_VER;
    if (NvAPI_VIO_QueryTopology(&topology) != NVAPI_OK)
        throw std::runtime_error("Unable to query available video I/O topologies");


    for (unsigned int i = 0; i < topology.vioTotalDeviceCount; i++)
    {
        NVVIOCAPS caps;
        std::memset(&caps, 0, sizeof(caps));
        caps.version = NVVIOCAPS_VER;
        if (NvAPI_VIO_GetCapabilities(topology.vioTarget[i].hVioHandle, &caps) != NVAPI_OK)
            throw std::runtime_error("Unable to get video I/O capabilities");

        SDIDevice* dev = new SDIDevice;
        if (caps.adapterCaps & NVVIOCAPS_VIDIN_SDI)
            dev->type = INPUT;
        else
        if (caps.adapterCaps & NVVIOCAPS_VIDOUT_SDI)
            dev->type = OUTPUT;

        dev->pimpl->caps = caps;
        dev->pimpl->handle = topology.vioTarget[i].hVioHandle;
        dev->pimpl->id = topology.vioTarget[i].vioId;
        devices.push_back(dev);
    }


    // after probing around
    //  do we have any devices?
    // if not just bail out, otherwise let the logic at the beginning of this method deal with this
    if (devices.empty())
        return 0;
    return get_device(devno);
}


} // namespace
