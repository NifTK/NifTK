/*=============================================================================

NifTK: A software platform for medical image computing.

Copyright (c) University College London (UCL). All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.

See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <gl/glew.h>
#include "niftkNV7SDIDataSourceImpl.h"
#include <iostream>
#include <cuda_gl_interop.h>

#include "dynlink_cuda.h"
#include "dynlink_nvcuvid.h"

#include <Mmsystem.h>
#include <opencv2/core/core_c.h>
#include <igtlTimeStamp.h>
#include <NiftyLinkUtils.h>
#include <fstream>

namespace niftk
{

	//-----------------------------------------------------------------------------
	static bool CUDADelayLoadCheck()
	{
		// the cuda dlls are delay-loaded, that means they are only mapped into our process
		// on demand, i.e. when we try to call an exported function for the first time.
		// this is what we do here.
		// if it fails (e.g. dll not found) then the runtime linker will throw a SEH exception.
		CUresult r;
		__try
		{	

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
			typedef HMODULE CUDADRIVER;
#else
			typedef void *CUDADRIVER;
#endif
			CUDADRIVER hHandleDriver = 0;

			CUresult result;
			result = cuInit(0, __CUDA_API_VERSION, hHandleDriver);
			if (result != CUDA_SUCCESS) {
				return false;
			}
			result = cuvidInit(0);
			if (result != CUDA_SUCCESS) {
				return false;
			}

			int cudadevices[10];
			unsigned int  actualcudadevices = 0;
			// note that zero is a valid device index
			std::memset(&cudadevices[0], -1, sizeof(cudadevices));
			return (cudaGLGetDevices(&actualcudadevices, &cudadevices[0], sizeof(cudadevices) / sizeof(cudadevices[0]), 
				cudaGLDeviceListAll) == CUDA_SUCCESS);
		}
		__except (1)
		{
			return false;
		}

		// unreachable
		assert(false);
	}


	//-----------------------------------------------------------------------------
	static void check_oglerror(const std::string& detailedmsg)
	{
		GLenum ec = glGetError();
		if (ec != GL_NO_ERROR)
		{
			throw std::runtime_error("OpenGL failed: " + detailedmsg);
		}
	}


	//-----------------------------------------------------------------------------
	NVidiaSDIDataSourceImpl::NVidiaSDIDataSourceImpl()
		: IGITimerBasedThread(0),
		sdidev(0), sdiin(0), streamcount(0), oglwin(0), oglshare(0), encoder(0), decoder(0),
		lock(QMutex::Recursive), 
		current_state(PRE_INIT), m_LastSuccessfulFrame(0), m_NumFramesCompressed(0), wireformat(""),
		m_CaptureWidth(0), m_CaptureHeight(0),
		m_Cookie(0),
		state_message("Starting up")
	{
		// check whether cuda dlls are available
		if (!CUDADelayLoadCheck())
		{
			throw std::runtime_error("No suitable CUDA available");
		}

		// helps with debugging
		setObjectName("NVidiaSDIDataSourceImpl");
		// sample interval in milliseconds that a timer will kick in to check for sdi hardware.
		// once capture has been setup, it will trigger much more often to keep up with video refresh rate.
		SetInterval(100);

		std::memset(&textureids[0], 0, sizeof(textureids));
		// we create the opengl widget on the ui thread once
		// and then never modify or signal/etc again

		oglwin = new QGLWidget(0, 0, Qt::WindowFlags(Qt::Window | Qt::FramelessWindowHint));
		oglwin->hide();
		assert(oglwin->isValid());
		// hack to get context sharing to work while the capture thread is cracking away
		oglshare = new GLHiddenWidget(0, oglwin, Qt::WindowFlags(Qt::Window | Qt::FramelessWindowHint));
		oglshare->hide();
		assert(oglshare->isValid());
		assert(oglwin->isSharing());		
		oglwin->makeCurrent();
		
		// Find out which gpu is rendering our window.
		// We need to know because this is where the video comes in
		// and on which we do compression.
		int           cudadevices[10];
		unsigned int  actualcudadevices = 0;
		// note that zero is a valid device index
		std::memset(&cudadevices[0], -1, sizeof(cudadevices));
		if (cudaGLGetDevices(&actualcudadevices, &cudadevices[0], sizeof(cudadevices) / sizeof(cudadevices[0]), cudaGLDeviceListAll) == cudaSuccess)
		{
			device_id = cudadevices[0];
		}	
		
		oglwin->doneCurrent();
		oglshare->context()->moveToThread(this);
		oglshare->moveToThread(this);


		// we want signal/slot processing to happen on our background thread.
		// for that to work we need to explicitly move this object because
		// it is currently owned by the gui thread.
		// beware: i wonder how well that works with start() and quit(). our thread instance stays
		// the same so it should be ok?				
		this->moveToThread(this);		
	}


	//-----------------------------------------------------------------------------
	NVidiaSDIDataSourceImpl::~NVidiaSDIDataSourceImpl()
	{
		// we dont really support concurrent destruction
		// if some other thread is still holding on to a pointer
		//  while we are cleaning up here then things are going to blow up anyway
		// might as well fail fast
		bool wasnotlocked = lock.tryLock();
		assert(wasnotlocked);

		QGLContext* ctx = const_cast<QGLContext*>(QGLContext::currentContext());
		try
		{
			// we need the capture context for proper cleanup
			oglwin->makeCurrent();
			// final attempt to not loose data.
			DumpNALIndex();			

			delete decoder;
			delete encoder;

			// we do not own sdidev!
			sdidev = 0;

			for (int i = m_ReadbackPBOs.size() - 1; i >= 0; --i)
			{
				glDeleteBuffers(1, (GLuint*)&m_ReadbackPBOs[i]);
			}
			m_ReadbackPBOs.clear();

			if (ctx)
				ctx->makeCurrent();
			else
				oglwin->doneCurrent();
			
		}
		catch (...)
		{
			std::cerr << "sdi cleanup threw exception" << std::endl;
		}
		
		delete oglshare;
		delete oglwin;
		

		// they'll get cleaned up now
		// if someone else is currently waiting on these
		//  deleting but not unlocking does what to their waiting?
		lock.unlock();
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::InitVideo()
	{		
		// we do not own the device!
		sdidev = 0;
		// but we gotta clear up this one
		delete sdiin;
		sdiin = 0;
		format = video::StreamFormat();
		// try not to loose any still compressing info
		DumpNALIndex();
		
		delete encoder;
		encoder = 0;

		// note: do not kill off decompressor here.
		// see description in SetPlayback() as for why.

		// clear these off, we'll allocate new ones below.
		for (int i = m_ReadbackPBOs.size() - 1; i >= 0; --i)
		{
			glDeleteBuffers(1, (GLuint*)&m_ReadbackPBOs[i]);
		}
		m_ReadbackPBOs.clear();

		// find our capture card
		for (int i = 0;; ++i)
		{
			video::SDIDevice* d = video::SDIDevice::get_device(i);
			if (d == 0)
				break;

			if (d->get_type() == video::SDIDevice::INPUT)
			{
				sdidev = d;
				break;
			}
		}

		// so we have a card, check the incoming video format and hook up capture
		if (sdidev)
		{
			streamcount = 0;
			for (int i = 0;; ++i, ++streamcount)
			{
				video::StreamFormat f = sdidev->get_format(i);
				wireformat = sdidev->get_wireformat();
				if (f.format == video::StreamFormat::PF_NONE)
				{
					break;
				}

				format = f;
			}

			if (format.format != video::StreamFormat::PF_NONE)
			{
				// we have one second worth of ringbuffer.
				int   ringbufferslots = format.get_refreshrate();
				sdiin = new video::SDIInput(sdidev, m_FieldMode, ringbufferslots);

				m_CaptureWidth = sdiin->get_width();
				m_CaptureHeight = sdiin->get_height();

				if (m_FieldMode == video::SDIInput::SPLIT_LINE_INTERLEAVED_STEREO)
					streamcount *= 2;

				// allocate a bunch of pbos that we can copy the incoming video frames into.
				// we do that during the capture loop (OnTimeoutImpl), so that when there's a request for a frame
				// copy has finished and we can map the pbo straight in.
				try
				{
					// width * height * pixelsize * channelcount
					std::size_t   pbosize = m_CaptureWidth * 4 * m_CaptureHeight * streamcount;
					for (int i = 0; i < ringbufferslots; ++i)
					{
						GLuint id = 0;
						glGenBuffers(1, &id);
						check_oglerror("Cannot create new buffer object.");

						glBindBuffer(GL_PIXEL_PACK_BUFFER, id);
						check_oglerror("Cannot bind newly created buffer object.");

						// GL_STREAM_READ is a hint to the driver that we want to read back from the gpu.
						glBufferData(GL_PIXEL_PACK_BUFFER, pbosize, 0, GL_STREAM_READ);
						check_oglerror("Cannot initialise newly created buffer object.");

						m_ReadbackPBOs.push_back(id);
					}
				}
				catch (...)
				{
					for (int i = m_ReadbackPBOs.size() - 1; i >= 0; --i)
					{
						glDeleteBuffers(1, (GLuint*)&m_ReadbackPBOs[i]);
					}
					m_ReadbackPBOs.clear();
					glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
					throw;
				}
				glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

				m_Cookie = (unsigned int)((((std::size_t) ((void*)sdiin)) >> 4) & 0xFFFFFFFF);

			}
		}

		// assuming everything went fine
		//  we now have texture objects that will receive video data everytime we call capture()
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::ReadbackViaPBO(char* buffer, std::size_t bufferpitch, int width, int height, int slot)
	{	
		assert(sdiin != 0);
		assert(bufferpitch >= width * 4);

		assert(m_CaptureWidth <= width);
		assert(m_CaptureHeight <= height);
		assert(slot < m_ReadbackPBOs.size());

		glBindBuffer(GL_PIXEL_PACK_BUFFER, m_ReadbackPBOs[slot]);
		check_oglerror("Cannot bind buffer object to be mapped.");
		const char* data = (const char*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);		
		check_oglerror("Cannot map buffer object.");
		assert(data != 0);

		
		// note: this would not work with fieldmode set the STACK_FIELDS.
		// but we dont support that anymore, so no problem.
		IplImage  subimgInput;
		cvInitImageHeader(&subimgInput, cvSize(std::min(m_CaptureWidth, width), m_CaptureHeight), IPL_DEPTH_8U, 4);
		IplImage  subimgOutput;
		cvInitImageHeader(&subimgOutput, cvSize(std::min(m_CaptureWidth, width), m_CaptureHeight), IPL_DEPTH_8U, 4);

		for (int i = 0; i < streamcount; ++i)
		{
			// remaining buffer too small?
			if (height - (i * m_CaptureHeight) < m_CaptureHeight)
				break;

			cvSetData(&subimgInput, (void*)&(data[m_CaptureHeight * (m_CaptureWidth * 4) * i]), m_CaptureWidth * 4);
			cvSetData(&subimgOutput, (void*)&(buffer[m_CaptureHeight * (m_CaptureWidth * 4) * i]), bufferpitch);

			cvFlip(&subimgInput, &subimgOutput);
		}

		glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::ReadbackRGBA(char* buffer, std::size_t bufferpitch, int width, int height, int slot)
	{
		assert(sdiin != 0);
		assert(bufferpitch >= width * 4);

		assert(m_CaptureWidth <= width);


		// fortunately we have 4 bytes per pixel
		glPixelStorei(GL_PACK_ALIGNMENT, 4);
		// row length is in pixels
		assert((bufferpitch % 4) == 0);
		// specifying a negative row-length would allow us to flip the texture
		// from bottom-left to top-left orientation on the fly.
		// but negative row-length is specifically mentioned in the spec as an error.
		glPixelStorei(GL_PACK_ROW_LENGTH, bufferpitch / 4);

		for (int i = 0; i < 4; ++i)
		{
			int texid = sdiin->get_texture_id(i, slot);
			if (texid == 0)
				break;

			// while we have the lock, texture dimensions are not going to change

			// remaining buffer too small?
			if (height - (i * m_CaptureHeight) < m_CaptureHeight)
				return;

			char* subbuf = &buffer[i * m_CaptureHeight * bufferpitch];

			glBindTexture(GL_TEXTURE_2D, texid);
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, subbuf);

			assert(glGetError() == GL_NO_ERROR);
		}
	}


	//-----------------------------------------------------------------------------
	bool NVidiaSDIDataSourceImpl::IsRunning() const
	{
		QMutexLocker    l(&lock);
		return current_state == RUNNING;
	}


	//-----------------------------------------------------------------------------
	bool NVidiaSDIDataSourceImpl::HasHardware() const
	{
		QMutexLocker    l(&lock);
		return sdidev != 0;
	}


	//-----------------------------------------------------------------------------
	bool NVidiaSDIDataSourceImpl::HasInput() const
	{
		QMutexLocker    l(&lock);
		return sdiin != 0;
	}


	//-----------------------------------------------------------------------------
	unsigned int NVidiaSDIDataSourceImpl::GetCookie() const
	{
		QMutexLocker    l(&lock);
		return m_Cookie;
	}


	//-----------------------------------------------------------------------------
	const char* NVidiaSDIDataSourceImpl::GetWireFormatString() const
	{
		QMutexLocker    l(&lock);
		return wireformat;
	}


	//-----------------------------------------------------------------------------
	video::FrameInfo NVidiaSDIDataSourceImpl::GetNextSequenceNumber(unsigned int ihavealready) const
	{
		QMutexLocker    l(&lock);

		video::FrameInfo  fi = { 0 };
		fi.sequence_number = ihavealready;

		const auto i = sn2slot_map.upper_bound(fi);
		if (i == sn2slot_map.end())
		{
			video::FrameInfo  empty = { 0 };
			return empty;
		}

		return i->first;
	}


	//-----------------------------------------------------------------------------
	// this is the format reported by sdi
	// the actual capture format might be different!
	video::StreamFormat NVidiaSDIDataSourceImpl::GetFormat() const
	{
		QMutexLocker    l(&lock);
		return format;
	}


	//-----------------------------------------------------------------------------
	QGLWidget* NVidiaSDIDataSourceImpl::GetCaptureContext()
	{
		QMutexLocker    l(&lock);
		assert(oglshare != 0);
		return oglshare;
	}


	//-----------------------------------------------------------------------------
	std::pair<int, int> NVidiaSDIDataSourceImpl::GetCaptureFormat() const
	{
		QMutexLocker    l(&lock);
		return std::make_pair(m_CaptureWidth, m_CaptureHeight);
	}


	//-----------------------------------------------------------------------------
	int NVidiaSDIDataSourceImpl::GetTextureId(unsigned int stream) const
	{
		QMutexLocker    l(&lock);
		if (sdiin == 0)
			return 0;
		if (stream >= 4)
			return 0;
		return textureids[stream];
	}


	//-----------------------------------------------------------------------------
	int NVidiaSDIDataSourceImpl::GetStreamCount() const
	{
		QMutexLocker    l(&lock);
		return streamcount;
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::SetFieldMode(video::SDIInput::InterlacedBehaviour mode)
	{
		QMutexLocker    l(&lock);
		m_FieldMode = mode;
	}


	//-----------------------------------------------------------------------------
	video::SDIInput::InterlacedBehaviour NVidiaSDIDataSourceImpl::GetFieldMode() const
	{
		QMutexLocker    l(&lock);
		return m_FieldMode;
	}


	//-----------------------------------------------------------------------------
	NVidiaSDIDataSourceImpl::CaptureState NVidiaSDIDataSourceImpl::GetCaptureState() const
	{
		QMutexLocker    l(&lock);
		return current_state;
	}


	//-----------------------------------------------------------------------------
	std::string NVidiaSDIDataSourceImpl::GetStateMessage() const
	{
		QMutexLocker    l(&lock);
		return state_message;
	}


	//-----------------------------------------------------------------------------
	std::string NVidiaSDIDataSourceImpl::GetCompressionOutputFilename() const
	{
		QMutexLocker    l(&lock);
		return m_CompressionOutputFilename;
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::setCompressionOutputFilename(const std::string& name)
	{
		QMutexLocker    l(&lock);
		m_CompressionOutputFilename = name;
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::run()
	{	
		m_Cookie = 0;				
		// it's possible for someone else to start and stop our thread.
		// just make sure we start clean if that happens.
		oglshare->makeCurrent();
		Reset();

		bool ok = connect(this, SIGNAL(SignalBump()), this, SLOT(DoWakeUp()), Qt::QueuedConnection);
		assert(ok);
		ok = connect(this, SIGNAL(SignalCompress(unsigned int, unsigned int*)), this, SLOT(DoCompressFrame(unsigned int, unsigned int*)), Qt::BlockingQueuedConnection);
		assert(ok);
		ok = connect(this, SIGNAL(SignalStopCompression()), this, SLOT(DoStopCompression()), Qt::BlockingQueuedConnection);
		assert(ok);
		ok = connect(this, SIGNAL(SignalGetRGBAImage(unsigned int, IplImage**, unsigned int*)), this, SLOT(DoGetRGBAImage(unsigned int, IplImage**, unsigned int*)), 
			Qt::BlockingQueuedConnection);
		assert(ok);
		ok = connect(this, SIGNAL(SignalTryPlayback(const char*, bool*, const char**)), this, SLOT(DoTryPlayback(const char*, bool*, const char**)), 
			Qt::DirectConnection);
		assert(ok);


		// current_state is reset by Reset() called above.
		// but is this a requirement for our message loop to run properly?
		// i think it is: whenever somebody (re-)starts our sdi thread we can assume
		// that our previous capture-setup is now dead/stale.
		assert(current_state == PRE_INIT);

		// let base class deal with timer and event loop and stuff
		IGITimerBasedThread::run();		

		// reset the libvideo cookie so that in-flight IGINVidiaDataType get rejected early
		m_Cookie = 0;

		ok = disconnect(this, SIGNAL(SignalBump()), this, SLOT(DoWakeUp()));
		assert(ok);
		ok = disconnect(this, SIGNAL(SignalCompress(unsigned int, unsigned int*)), this, SLOT(DoCompressFrame(unsigned int, unsigned int*)));
		assert(ok);
		ok = disconnect(this, SIGNAL(SignalStopCompression()), this, SLOT(DoStopCompression()));
		assert(ok);
		ok = disconnect(this, SIGNAL(SignalGetRGBAImage(unsigned int, IplImage**, unsigned int*)), this, SLOT(DoGetRGBAImage(unsigned int, IplImage**, unsigned int*)));
		assert(ok);
		ok = disconnect(this, SIGNAL(SignalTryPlayback(const char*, bool*, const char**)), this, SLOT(DoTryPlayback(const char*, bool*, const char**)));
		assert(ok);

		oglshare->makeCurrent();
		delete sdiin;
		sdiin = 0;
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::DoWakeUp()
	{
		OnTimeoutImpl();
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::OnTimeoutImpl()
	{
		QMutexLocker    l(&lock);

		if (current_state == NVidiaSDIDataSourceImpl::FAILED)
		{
			return;
		}
		if (current_state == NVidiaSDIDataSourceImpl::DEAD)
		{
			return;
		}

		// not much to do in case of playback.
		// it's all triggered by GetRGBAImage().
		if (current_state == PLAYBACK)
		{
			return;
		}		

		if (current_state == NVidiaSDIDataSourceImpl::PRE_INIT)
		{
			//oglwin->makeCurrent();			
			oglshare->makeCurrent();
			current_state = NVidiaSDIDataSourceImpl::HW_ENUM;

			// side note: we could try to avoid a race-condition of PRE_INIT-make-current and signal delivery
			// by connecting the signals only after HW_ENUM, when we know that it is safe to deliver them.
			// however, i dont know how well that mixes with signal drop-out recovery and repeated HW_ENUM,
			// which would then connect signals multiple times.
		}		

		// make sure nobody messes around with contexts
		//assert(QGLContext::currentContext() == oglwin->context());
		//assert(QGLContext::currentContext() == oglshare->context());
		
		if (current_state == NVidiaSDIDataSourceImpl::HW_ENUM)
		{
			try
			{	
				// libvideo does its own glew init, so we can get cracking straight away
				InitVideo();

				// once we have an input setup
				//  grab at least one frame (not quite sure why)
				if (sdiin)
				{
					sdiin->capture();

					// make sure we are checking with twice the frame rate.
					// sampling theorem and stuff
					SetInterval((unsigned int)std::max(1, (int)(500.0f / format.get_refreshrate())));
				}
			}
			// getting an exception means something is broken
			// during normal operation this should never happen
			//  even if there's no hardware or signal
			catch (const std::exception& e)
			{
				state_message = std::string("Failed: ") + e.what();
				current_state = NVidiaSDIDataSourceImpl::FAILED;
				// no point trying check often, it's not going to recover
				SetInterval(1000);
				return;
			}
			catch (...)
			{
				state_message = "Failed";
				current_state = NVidiaSDIDataSourceImpl::FAILED;
				// no point trying check often, it's not going to recover
				SetInterval(1000);
				return;
			}
		}

		if (!HasHardware())
		{
			state_message = "No SDI hardware";
			// no hardware then nothing to do
			current_state = NVidiaSDIDataSourceImpl::DEAD;
			// no point trying check often, it's not going to recover
			SetInterval(1000);
			return;
		}

		if (!HasInput())
		{
			state_message = "No input signal";
			// no signal, try again next round
			current_state = NVidiaSDIDataSourceImpl::HW_ENUM;
			// dont re-setup too quickly
			SetInterval(500);
			return;
		}

		// if we get to here then we should be good to go!
		current_state = NVidiaSDIDataSourceImpl::RUNNING;

		
		try
		{
			bool hasframe = sdiin->has_frame();
			// note: has_frame() will not throw an exception in case setup is broken

			// make sure we try to capture a frame if the previous one was too long ago.
			// that will check for errors and throw an exception if necessary, which will then allow us to restart.
			if ((timeGetTime() - m_LastSuccessfulFrame) > 1000)
				hasframe = true;

			if (hasframe)
			{
				// note: capture() will block for a frame to arrive
				// that's why we have hasframe above
				video::FrameInfo fi = sdiin->capture();
				// in case of partial transfers we'll get the zero sequence number.
				if (fi.sequence_number != 0)
				{
					m_LastSuccessfulFrame = timeGetTime();

					// keep the most recent set of texture ids around
					// this is mainly for the preview window.
					// and for the pbo bits below.
					for (int i = 0; i < 4; ++i)
					{
						textureids[i] = sdiin->get_texture_id(i, -1);
					}

					igtl::TimeStamp::Pointer timeCreated = igtl::TimeStamp::New();
					timeCreated->GetTime();
					fi.id = timeCreated->GetTimeStampInNanoseconds();

					int newest_slot = sdiin->get_current_ringbuffer_slot();
					// whatever we had in this slot is now obsolete
					auto oldsni = slot2sn_map.find(newest_slot);
					if (oldsni != slot2sn_map.end())
					{
						auto oldsloti = sn2slot_map.find(oldsni->second);
						if (oldsloti != sn2slot_map.end())
						{
							sn2slot_map.erase(oldsloti);
						}
						slot2sn_map.erase(oldsni);
					}
					slot2sn_map[newest_slot] = fi;
					sn2slot_map[fi] = newest_slot;

					// queue a copy to pbo (that we can map later on).
					glBindBuffer(GL_PIXEL_PACK_BUFFER, m_ReadbackPBOs[newest_slot]);
					check_oglerror("Failed binding readback PBO");

					// each stream/channel has its own texture, but we copy all into one giant pbo.
					// the address is relative to the bound pbo (if there is a pbo bound, that is).
					ReadbackRGBA((char*)0, m_CaptureWidth * 4, m_CaptureWidth, m_CaptureHeight * streamcount, newest_slot);
					glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

					state_message = "Grabbing";
				}
				else
				{
					// in case of partial captures hasframe will be true but we wont get new sequence numbers.
					// this is effectively a brown-out. not sure how to handle it actually.
					// because we'd want the capture bits to recover with the original number of streams. unless of course
					// there was a genuine change of camera etc.
					if ((timeGetTime() - m_LastSuccessfulFrame) > 2000)
						throw "trigger handler below";
				}
			}
		}
		// capture() might throw if the capture setup has become invalid
		// e.g. a mode change or signal lost
		catch (...)
		{
			state_message = "Glitched out";
			// mark our source as failed! otherwise we run the risk of silently corrupting a
			// recording session where (due to flaky cable) one stream drops out, source recovers
			// but enums only one channel, and recording just continues.
			current_state = NVidiaSDIDataSourceImpl::FAILED;
			// dont re-setup too quickly
			SetInterval(500);
			// whatever sequence numbers we had in the ringbuffer are now obsolete.
			// if we dont explicitly delete these then QmitkIGINVidiaDataSource::GrabData() will get mightily confused.
			sn2slot_map.clear();
			slot2sn_map.clear();

			glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

			emit SignalFatalError(QString("SDI capture setup failed! Try to remove and add it again."));
			return;
		}
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::Reset()
	{
		QMutexLocker    l(&lock);

		// dont forget to remove any stale sequence numbers.
		sn2slot_map.clear();
		slot2sn_map.clear();

		state_message = "Initialising";

		current_state = PRE_INIT;
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::DecompressRGBA(unsigned int sequencenumber, IplImage** img, unsigned int* streamcountinimg)
	{
		// has to be called with lock held!

		assert(img != 0);
		assert(streamcountinimg != 0);
		// nothing to be done.
		if (*img == 0)
		{
			*streamcountinimg = 0;
			return;
		}

		// these are display dimenions (what we are interested in).bl
		std::pair<int, int> dims = decoder->get_dims();
		int w = dims.first;
		int h = dims.second;

		if (((*img)->width != w) ||
			((*img)->height != (h * streamcount)))
		{
			// decompressor will not crop its output if it doesnt fit, it will simply fail.
			*streamcountinimg = 0;
			return;
		}

		// mark layout as rgba instead of the opencv-default bgr.
		// (this is not used/supported by opencv itself.)
		std::memcpy(&((*img)->channelSeq[0]), "RGBA", 4);

		// during playback there are no proper sequence numbers. instead, we repurpose
		// these as a frame number (quite similar anyway).
		bool ok = true;
		for (int i = 0; i < streamcount; ++i)
		{
			// note: opencv's widthStep is in bytes.
			char* subimg = &((*img)->imageData[i * h * (*img)->widthStep]);
			try
			{
				ok &= decoder->decompress(sequencenumber + i, subimg, (*img)->widthStep * h, (*img)->widthStep);
				if (ok) {
					IplImage  subimgInput;
					cvInitImageHeader(&subimgInput, cvSize((*img)->widthStep, h), IPL_DEPTH_1U, 1);
					cvSetData(&subimgInput, (void*)subimg, (*img)->widthStep);
					cvFlip(&subimgInput, NULL);
				}				
			}
			catch (const std::exception& e)
			{
				std::cerr << "Caught exception while decompressing: " << e.what() << std::endl;
				ok = false;
				break;
			}
		}

		if (ok)
		{
			*streamcountinimg = streamcount;
		}
		else
		{
			*streamcountinimg = 0;
		}
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::DoGetRGBAImage(unsigned int sequencenumber, IplImage** img, unsigned int* streamcountinimg)
	{
		QMutexLocker    l(&lock);

		// redirect to decompression early.
		// all the other stuff below is for live video capture.
		if (current_state == PLAYBACK)
		{			
			assert(decoder);
			DecompressRGBA(sequencenumber, img, streamcountinimg);
			return;
		}

		// temp obj to search for the correct slot
		video::FrameInfo    fi = { 0 };
		fi.sequence_number = sequencenumber;

		auto sni = sn2slot_map.lower_bound(fi);
		// we need to check whether the requested sequence number is still valid.
		// there may have been a capture reset.
		if (sni == sn2slot_map.end())
		{
			*img = 0;
			*streamcountinimg = 0;
			return;
		}


		// make sure nobody messes around with contexts.
		// notice that we put this check here and not at the top so that the above check for
		// sequence number bails out if there are no captured frames.
		// this is necessary because otherwise there could be a race condition between this-sdi-thread
		// and gui-update where gui-update tries a GetRGBAImage() before sdi-thread has had a chance to 
		// check for InitVideo() with the corresponding ogl-make-current.
		//assert(QGLContext::currentContext() == oglwin->context());

		int w = sdiin->get_width();
		int h = sdiin->get_height();

		IplImage* frame = *img;
		// FIXME: passing in a null buffer is no longer supported!
		if (frame == 0)
		{
			frame = cvCreateImage(cvSize(w, h * streamcount), IPL_DEPTH_8U, 4);
		}
		// mark layout as rgba instead of the opencv-default bgr
		std::memcpy(&frame->channelSeq[0], "RGBA", 4);

		ReadbackViaPBO(frame->imageData, frame->widthStep, frame->width, frame->height, sni->second);

		*img = frame;
		*streamcountinimg = streamcount;
	}


	//-----------------------------------------------------------------------------
	int NVidiaSDIDataSourceImpl::GetRGBAImage(unsigned int sequencenumber, IplImage* targetbuffer)
	{
		IplImage*     img = targetbuffer;
		unsigned int  streamcount = 0;
		// this will block on the sdi thread. so no locking in this method!
		emit SignalGetRGBAImage(sequencenumber, &img, &streamcount);

		return streamcount;
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::DoCompressFrame(unsigned int sequencenumber, unsigned int* frameindex)
	{
		QMutexLocker    l(&lock);
		
		// it's unlikely that we'll ever catch an exception here.
		// all the other bits around the sdi data source would start failing earlier
		// if there's something wrong and we'd never even get here.
		// BUT: various libvideo functions *can* throw in error conditions. so better
		// prevent the exception from leaking into the qt message loop.
		try
		{
			if (encoder == 0)
			{
				// also keep sdi logs
				sdiin->flush_log();
				sdiin->set_log_filename(m_CompressionOutputFilename + ".sdicapture.log");

				// when we get a new compressor we want to start counting from zero again
				m_NumFramesCompressed = 0;

				encoder = new Encoder();
				if (!encoder->initialize_encoder(sdiin->get_width(), sdiin->get_height(), device_id, (int)(format.refreshrate * streamcount / 1000.0),
					m_CompressionOutputFilename)) {
					throw std::runtime_error("Could not create the encoder");					
				}			

				frame_data = std::vector<char>(sdiin->get_width() * sdiin->get_height() * 4);								
			}
			
			// find out which ringbuffer slot the request sequence number is in, if any
			video::FrameInfo  fi = { 0 };
			fi.sequence_number = sequencenumber;
			auto sloti = sn2slot_map.find(fi);
			if (sloti != sn2slot_map.end())
			{
				// sanity check
				assert(slot2sn_map.find(sloti->second)->second.sequence_number == sequencenumber);

				// compress each stream
				for (int i = 0; i < streamcount; ++i)
				{
					int tid = sdiin->get_texture_id(i, sloti->second);
					assert(tid != 0);
					
					int w = sdiin->get_width();
					int h = sdiin->get_height();						

					glBindTexture(GL_TEXTURE_2D, tid);
					glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, &frame_data[0]);					
					check_oglerror("Could not copy texture");
					encoder->encode_frame((unsigned char *)(&frame_data[0]));
					m_NumFramesCompressed++;
				}

				*frameindex = m_NumFramesCompressed;
			}
			else
			{
				// i want to know how often this happens, really...
				std::cerr << "Debug: sdi compressor: requested sn that is no longer available" << std::endl;
				*frameindex = 0;
			}
		}
		catch (...)
		{
			// die straight away

			// FIXME: cant remember why i'd pop the cuda context here (or after compressor constructor fail, where these lines used to be).
			//CUcontext ctx;
			//cuCtxPopCurrent(&ctx);
			*frameindex = 0;
			return;
		}

	}


	//-----------------------------------------------------------------------------
	unsigned int NVidiaSDIDataSourceImpl::CompressFrame(unsigned int sequencenumber)
	{
		unsigned int frameindex = 0;
		emit SignalCompress(sequencenumber, &frameindex);
		return frameindex;
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::DoStopCompression()
	{
		QMutexLocker    l(&lock);
		sdiin->flush_log();
		if (!encoder->flush_encoder()) {
			throw std::runtime_error("Could not flush encoder.");
		}

		DumpNALIndex();

		delete encoder;
		encoder = 0;
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::StopCompression()
	{
		emit SignalStopCompression();
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::SetPlayback(bool on, int expectedstreamcount)
	{
		QMutexLocker    l(&lock);

		if (on)
		{
			// recording should have stopped earlier!
			if (encoder != 0)
			{
				throw std::runtime_error("Compressor still running! Cannot playback and record at the same time.");
			}
			// decompressor should have been initialised via TryPlayback().
			if (decoder == 0)
			{
				throw std::runtime_error("No decompressor. Forgot to call TryPlayback()?");
			}
			if (expectedstreamcount <= 0)
			{
				throw std::runtime_error("Need correct stream count for playback");
			}
			if (expectedstreamcount > 4)
			{
				std::cerr << "WARNING: more than 4 streams during playback! That is likely an error: current hardware can capture max 4." << std::endl;
			}

			// may not be necessary but cleanup a bit anyway to avoid confusion later.
			sn2slot_map.clear();
			slot2sn_map.clear();

			// beware: do not stop this thread!
			// it still needs to do all event processing for GetRGBAImage() and friends.
			// but there's no need to run it at full speed.
			SetInterval(1000);
			// it will not do anything: wake up from timer sleep, and return immediately.
			current_state = PLAYBACK;

			// the compressor is bare-bones, it doesnt know anything about multiple streams.
			// it only compresses individual images we pipe in.
			// so the number of streams during playback has to come from an external source.
			streamcount = expectedstreamcount;

			std::pair<int, int> dims = decoder->get_dims();

			m_CaptureWidth = dims.first;
			m_CaptureHeight = dims.second;
		}
		else
		{
			current_state = HW_ENUM;
			// bumping this thread will wake it up from timer sleep.
			// it will then do a hardware check and set its own refreshrate to something suitable for the input video.
			emit SignalBump();

			// note: decompressor will not be killed during InitVideo()! it will just linger around.
			// there is not really a nice way of cleaning up with the current design...
			// there's a race condition: user calls TryPlayback() which inits the decompressor,
			// meanwhile OnTimeoutImpl will try to call InitVideo() which would kill the decompressor,
			// and then user calls here SetPlayback(true): exception.
		}
	}


	//-----------------------------------------------------------------------------
	bool NVidiaSDIDataSourceImpl::DumpNALIndex() const
	{
		QMutexLocker    l(&lock);

		if (encoder)
		{
			if (encoder->write_nal_to_file(GetCompressionOutputFilename() + ".nalindex")) {
				encoder->clear_nals();
				return true;
			}
		}		

		return true;
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::DoTryPlayback(const char* filename, bool* ok, const char** errormsg)
	{
		QMutexLocker    l(&lock);
		
		delete decoder;
		decoder = 0;

		try
		{	
			decoder = new Decoder(device_id);
			if (!decoder->initialize_decoder(std::string(filename))) {
				throw std::runtime_error(std::string(filename) + ": Could not initialize the decoder");
			}

			std::string indexFilePath = std::string(filename) + ".nalindex";
			std::ifstream  indexfile(indexFilePath);
			int count = 0;
			std::string current_line;
			size_t frame_number;
			uint_least64_t offset = 0;
			int type = 0;
			bool legacy = false;

			std::map<OldFrameTypes, NV_ENC_PIC_TYPE> FRAME_TYPE_MAPPER;
			FRAME_TYPE_MAPPER[OldFrameTypes::I] = NV_ENC_PIC_TYPE_IDR;
			FRAME_TYPE_MAPPER[OldFrameTypes::P] = NV_ENC_PIC_TYPE_P;
			FRAME_TYPE_MAPPER[OldFrameTypes::B] = NV_ENC_PIC_TYPE_B;

			NV_ENC_PIC_TYPE  npt = NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_UNKNOWN;
			OldFrameTypes oft;

			while (std::getline(indexfile, current_line)) {
				if (!current_line.empty()) {
					std::istringstream ss(current_line);
					ss >> frame_number;
					ss >> offset;
					ss >> type;

					if (count == 0) {
						legacy = (type == 1);
						decoder->set_legacy(legacy);
					}

					if (legacy) {
						oft = static_cast<OldFrameTypes>(type);
						decoder->update_index(frame_number, offset,
							FRAME_TYPE_MAPPER[oft]);
					}
					else {
						npt = static_cast<NV_ENC_PIC_TYPE>(type);
						decoder->update_index(frame_number, offset, npt);
					}
					++count;
				}
			}
		}
		catch (const std::exception& e)
		{
			*ok = false;
			std::cerr << "Decompression init failed: " << e.what() << std::endl;
			*errormsg = "Decompression init failed";
			return;
		}

		*errormsg = 0;
		*ok = true;
	}


	//-----------------------------------------------------------------------------
	void NVidiaSDIDataSourceImpl::TryPlayback(const std::string& filename)
	{
		QMutexLocker    l(&lock);

		const char*   fns = filename.c_str();
		bool          ok = false;
		const char*   err = 0;

		emit SignalTryPlayback(fns, &ok, &err);

		if (!ok)
		{
			assert(err != 0);
			throw std::runtime_error(err);
		}
	}


	//-----------------------------------------------------------------------------
	bool NVidiaSDIDataSourceImpl::SequenceNumberComparator::operator()(const video::FrameInfo& a, const video::FrameInfo& b) const
	{
		return a.sequence_number < b.sequence_number;
	}

} // end namespace
