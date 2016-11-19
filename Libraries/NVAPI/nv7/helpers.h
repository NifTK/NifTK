#ifndef HELPERS_H__
#define HELPERS_H__

#include "macros.h"

inline void close_file(HANDLE hFileHandle)
{
    if (hFileHandle)
    {
#if defined (NV_WINDOWS)
        CloseHandle(hFileHandle);
#else
        fclose((FILE *)hFileHandle);
#endif
    }
}

#endif
