/*=============================================================================

  libvideo: a library for SDI video processing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#pragma once
#ifndef LIBVIDEO_SDIINPUT_H_3CD0DE9000FA4846B097E35979071A3B
#define LIBVIDEO_SDIINPUT_H_3CD0DE9000FA4846B097E35979071A3B

#include <video/device.h>
#include <video/frame.h>
#include <video/dllexport.h>
#include <string>


namespace video
{


/** @brief Holds nvapi related stuff, pimpl'd to avoid header pollution. */
class SDIInputImpl;
class LIBVIDEO_DLL_EXPORTS SDIInput
{
private:

protected:
    SDIInputImpl*   pimpl;


public:
    // what to do if format is interlaced
    // these are ignored if incoming video is progressive
    enum InterlacedBehaviour
    {
        // interlaced video is treated the same way as progressive
        DO_NOTHING_SPECIAL,
        // one of the fields (not specifying which one) is dropped
        DROP_ONE_FIELD,
        // both fields are captured and stacked vertically
        STACK_FIELDS
    };


public:
    /**
     * @brief Initialises SDI capture into texture memory.
     * @param dev the device on which to capture
     * @param interlaced what to do with interlaced video, ignored of format is progressive
     * @param ringbuffersize how may slots to allocate in the texture ringbuffer
     * @throws std::runtime_error if anything goes wrong with setup
     * @throws std::logic_error if you passed in the wrong parameters
     */
    SDIInput(SDIDevice* dev, InterlacedBehaviour interlaced = DO_NOTHING_SPECIAL, int ringbuffersize = 0);
    ~SDIInput();

private:
    // not implemented
    SDIInput(const SDIInput& copyme);
    SDIInput& operator=(const SDIInput& assignme);


    /** 
     * @name Actual video dimensions. 
     * @detail These may be different from the reported capture format if we are dropping a field, for example.
     */
    //@{
public:
    /** @brief Returns the width in pixels of the texture object that receives video data. */
    int get_width() const;
    /** @brief Returns the height in pixels of the texture object that receives video data. */
    int get_height() const;
    //@}


protected:
    // if true then capture() will prepare the textures, otherwise data is only in pbos
    bool  preparetextures;


public:
    /** @brief Returns the most up-to-date index into the ring buffer. */
    int get_current_ringbuffer_slot() const;

    /**
     * @brief Returns the texture ID for a specific ring buffer slot, or for the current slot.
     * @detail The texture has always RGBA format, and dimensions reported by get_width()/get_height().
     * @param streamno the index of the stream, counting from zero
     * @param ringbufferslot a specific ringbuffer slot, otherwise the currently active slot
     * @throws nothing should not throw
     * @returns The OpenGL object name (i.e. ID) of the requested texture map; or zero if there is none for the requested stream.
     */
    int get_texture_id(int streamno, int ringbufferslot = -1) const;

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
    bool has_frame() const;


#pragma warning(push)
#pragma warning(disable: 4251)      //  class '...' needs to have dll-interface to be used by clients of class '...'

protected:
    std::string     logfilename;

#pragma warning(pop)


public:
    void set_log_filename(const std::string& fn);
    void flush_log();
};


} // namespace

#endif // LIBVIDEO_SDIINPUT_H_3CD0DE9000FA4846B097E35979071A3B
