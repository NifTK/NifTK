#pragma once
#ifndef LIBVIDEO_FRAME_H_E9E717C1DAB64B1D8E7E877FF5C02FE7
#define LIBVIDEO_FRAME_H_E9E717C1DAB64B1D8E7E877FF5C02FE7

#include <video/dllexport.h>


namespace video
{


struct LIBVIDEO_DLL_EXPORTS StreamFormat
{
	enum PictureFormat
	{
		PF_NONE		= 0,
		PF_487		= 487,	// 720 x 487
		PF_576		= 576,	// 720 x 576
		PF_720		= 720,	// FIXME: never tested!
		PF_1035		= 1035,	// FIXME: never tested!
		PF_1080		= 1080,	// 1920 x 1080
		PF_2048		= 2048	// 2048 x 1080
	}		format;

	int get_width() const;
	int get_height() const;

	enum RefreshRate
	{
		RR_NONE		= 0,
		RR_23_976	= 23976,
		RR_24		= 24000,	// FIXME: never tested!
		RR_25		= 25000,
		RR_29_97	= 29970,	// FIXME: never tested!
		RR_30		= 30000,
		RR_47_96	= 47960,	// FIXME: never tested!
		RR_48_00	= 48000,	// FIXME: never tested!
		RR_50		= 50000,
		RR_59_94	= 59940,	// FIXME: never tested!
		RR_60		= 60000	// FIXME: never tested!
	}		refreshrate;

	bool is_interlaced;

	// in hz or fps
	float get_refreshrate() const;

	StreamFormat(PictureFormat pf = PF_NONE, RefreshRate rr = RR_NONE, bool _is_interlaced = false)
		: format(pf), refreshrate(rr), is_interlaced(_is_interlaced)
	{
	}

	// FIXME: conversion to bool for simple x != NONE check?
};


struct FrameInfo
{
	// your id for this frame, not used by libvideo
	unsigned __int64	id;

	// (arbitrary) time-stamp of when this frame arrived at the capture hardware
	unsigned __int64	arrival_time;
	// another (arbitrary) time-stamp when it was scanned out by the hardware
	unsigned __int64	present_time;

	// sequence number determined by capture hardware
	unsigned int		sequence_number;
};


} // namespace

#endif // LIBVIDEO_FRAME_H_E9E717C1DAB64B1D8E7E877FF5C02FE7
