// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#define NOMINMAX

#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <sstream>
#include <map>
#include <fstream>

#include <GL/glew.h>
#include <GL/wglew.h>

#include <nvapi.h>
#include <NVEncoderAPI.h>
#include <nvcuvid.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <windows.h>
