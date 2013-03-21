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
    SDIInput(SDIDevice* dev, InterlacedBehaviour interlaced = DO_NOTHING_SPECIAL, int ringbuffersize = 0);
    ~SDIInput();

private:
    // not implemented
    SDIInput(const SDIInput& copyme);
    SDIInput& operator=(const SDIInput& assignme);


public:
    // these may be different from the reported capture format
    //  if we are dropping a field, for example
    int get_width() const;
    int get_height() const;


protected:
    // if true then capture() will prepare the textures, otherwise data is only in pbos
    bool  preparetextures;


public:
    /** @brief Returns the most up-to-date index into the ring buffer. */
    int get_current_ringbuffer_slot() const;

    // format is always RGBA!
    // returns zero if no stream with that index
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
};


} // namespace

#endif // LIBVIDEO_SDIINPUT_H_3CD0DE9000FA4846B097E35979071A3B
