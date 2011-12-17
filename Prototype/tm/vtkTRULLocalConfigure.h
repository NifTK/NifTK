/*=========================================================================
This source has no copyright.  It is intended to be copied by users
wishing to create their own VTK classes locally.
=========================================================================*/
#ifndef __Configure_h
#define __Configure_h

#if 1
# define vtkTRULLocal_SHARED
#endif

#if defined(_MSC_VER) && defined(vtkTRULLocal_SHARED)
# pragma warning ( disable : 4275 )
#endif

#if defined(_WIN32) && defined(vtkTRULLocal_SHARED)
# if defined(vtkTRULLocal_EXPORTS)
#  define VTK_vtkTRULLocal_EXPORT __declspec( dllexport ) 
# else
#  define VTK_vtkTRULLocal_EXPORT __declspec( dllimport ) 
# endif
#else
# define VTK_vtkTRULLocal_EXPORT
#endif

#endif // __vtkTRULLocalConfigure_h
