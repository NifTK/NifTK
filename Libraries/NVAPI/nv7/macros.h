#ifndef __MACROS_H__
#define __MACROS_H__

#define DISALLOW_COPY_AND_ASSIGNMENT(TypeName) \
  TypeName(const TypeName&) = delete;   \
  void operator=(const TypeName&) = delete
  
#if defined(_WIN32) || defined(_WIN16)
#   define NV_WINDOWS
#endif

#if (defined(__unix__) || defined(__unix) ) && !defined(nvmacosx) && !defined(vxworks) && !defined(__DJGPP__) && !defined(NV_UNIX) && !defined(__QNX__) && !defined(__QNXNTO__)
#   define NV_UNIX
typedef void* HANDLE;
typedef void* HINSTANCE;
#endif 

#endif


