/*=============================================================================

  libvideo: a library for SDI video processing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#pragma once
#ifndef LIBVIDEO_DLLEXPORT_H_4669DD5453A942E5914ACAF8788444C9
#define LIBVIDEO_DLLEXPORT_H_4669DD5453A942E5914ACAF8788444C9


// for the moment, libvideo works on windows only
// but lets just guard for it anyway
#ifdef _MSC_VER

#ifdef LIBVIDEO_BUILDING_DLL
#define LIBVIDEO_DLL_EXPORTS    __declspec(dllexport)
#else
#define LIBVIDEO_DLL_EXPORTS    __declspec(dllimport)
#endif


// various bits rely on safely dll-exporting class members which may reference
//  crt components (that may not be explicitly declared to be exported)
// this checks that we are building against the dll version of the crt
#ifndef _DLL
#ifdef _MT
#error You are compiling against the static version of the CRT. This is not supported! Choose DLL instead!
#else
#pragma message("Warning: cannot tell which CRT version you are building with. Stuff might fail.")
#endif
#endif

#endif // _MSC_VER


#endif // LIBVIDEO_DLLEXPORT_H_4669DD5453A942E5914ACAF8788444C9
