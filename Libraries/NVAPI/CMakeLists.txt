#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)

if(NOT (CUDA_VERSION_MAJOR LESS 7))
	ADD_DEFINITIONS(-DUSE_NV_7)
	ADD_DEFINITIONS(-DNV12_2_ARGB_PTX_DIR="${CMAKE_SOURCE_DIR}//Libraries/NVAPI/nv7/ptx")
	ADD_DEFINITIONS(-DGLEW_STATIC)
	set(niftknvapi_SRCS
		glew/glew.c
		nv7/cudamodulemgr.cpp
		nv7/cudaprocessframe.cpp
		nv7/dynlink_cuda.cpp
		nv7/dynlink_nvcuvid.cpp
		nv7/textureencoder.cpp
		nv7/hwencoder.cpp
		nv7/encoder.cpp
		nv7/decoder.cpp
		video/device.cxx
		video/deviceimpl.cxx
		video/frame.cxx
		video/sdiinput.cxx
		video/sdioutput.cxx
	)

	include_directories(${CMAKE_SOURCE_DIR}/Libraries/NVAPI/video)
	include_directories(${CMAKE_SOURCE_DIR}/Libraries/NVAPI/nv7)
	include_directories(${CMAKE_SOURCE_DIR}/Libraries/NVAPI/glew)
	include_directories(${NVAPI_INCLUDE_DIR})
	include_directories(${CMAKE_BINARY_DIR})
	include_directories(${CUDA_TOOLKIT_INCLUDE})

    add_definitions(-DNIFTK_NVAPI_STATIC)
	CUDA_ADD_LIBRARY(niftknvapi STATIC ${niftknvapi_SRCS})

#	foreach(cuda_dll nvcuda)
#	  if(WIN32)
#	    set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll")
#      endif()
#	  find_library(cuda_dll_location
#	    NAMES ${cuda_dll}
#	  )
#	  install(PROGRAMS ${cuda_dll_location} DESTINATION bin)
#	endforeach()

	target_link_libraries(niftknvapi ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${NVAPI_LIBRARY}
		${OPENGL_gl_LIBRARY})
		#${CUDA_CUDA_LIBRARY}
		#${CUDA_CUDART_LIBRARY})

	if(MSVC AND NIFTK_DELAYLOAD_CUDA)
		# all of the nvidia-driver/cuda dependencies should be delay-loaded
		#  so this library would at least load on a machine that doesnt have nv hardware
		set_target_properties(niftknvapi PROPERTIES LINK_FLAGS 
		"/DELAYLOAD:nvcuda.dll /DELAYLOAD:${CUDA_CUDART_DLL_NAME}")
	endif()

else(NOT (CUDA_VERSION_MAJOR LESS 7))
	set(niftknvapi_SRCS
		niftkNVAPICommon.cxx
		glew/glew.c
		video/compress.cxx
		video/conversion.cxx
		video/decompress.cxx
		video/device.cxx
		video/deviceimpl.cxx
		video/frame.cxx
		video/sdiinput.cxx
		video/sdioutput.cxx
		video/rgba2nv12.cu
	)

	# with all the itk etc stuff, nvcc dies with "command line too long".
	# so instead drop off all the not needed stuff and start clean.
	# http://www.cmake.org/Bug/print_bug_page.php?bug_id=12198
	set_directory_properties( PROPERTIES INCLUDE_DIRECTORIES "" )

	include_directories(${CMAKE_SOURCE_DIR}/Libraries/NVAPI/video)
	include_directories(${CMAKE_SOURCE_DIR}/Libraries/NVAPI/glew)
	include_directories(${NVAPI_INCLUDE_DIR})
	include_directories(${CMAKE_BINARY_DIR})
	include_directories(${CUDA_TOOLKIT_INCLUDE})

	# add_library(niftknvapi ${niftknvapi_SRCS})
	CUDA_ADD_LIBRARY(niftknvapi ${niftknvapi_SRCS})

	if(BUILD_SHARED_LIBS)
		if(WIN32)
			add_definitions(-DNIFTKNVAPI_WINDOWS_EXPORT)
			add_definitions(-DLIBVIDEO_BUILDING_DLL)
		endif(WIN32)
	endif(BUILD_SHARED_LIBS)


	get_filename_component(cudaencoderpath ${CUDA_CUDART_LIBRARY} PATH)

	target_link_libraries(niftknvapi  
		${NVAPI_LIBRARY}
		${OPENGL_gl_LIBRARY}
		${CUDA_CUDA_LIBRARY}
		${CUDA_CUDART_LIBRARY}
		${cudaencoderpath}/nvcuvenc.lib
		${cudaencoderpath}/nvcuvid.lib
	)

	# unconditionally delay-load the ENcoder.
	# it has been dropped from newer driver versions, so might not be present on the user's machine.
	if(MSVC)
		set_target_properties(niftknvapi PROPERTIES LINK_FLAGS "/DELAYLOAD:nvcuvenc.dll")
	endif()

	if(MSVC AND NIFTK_DELAYLOAD_CUDA)
		# all of the nvidia-driver/cuda dependencies should be delay-loaded
		#  so this library would at least load on a machine that doesnt have nv hardware
		set_target_properties(niftknvapi PROPERTIES LINK_FLAGS 
		"/DELAYLOAD:nvcuvenc.dll /DELAYLOAD:nvcuvid.dll /DELAYLOAD:nvcuda.dll /DELAYLOAD:${CUDA_CUDART_DLL_NAME}")
	endif()

endif(NOT (CUDA_VERSION_MAJOR LESS 7)) 
 
