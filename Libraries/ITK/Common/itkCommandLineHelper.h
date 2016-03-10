/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkCommandLineHelper_h
#define itkCommandLineHelper_h

#include <NifTKConfigure.h>
#include <niftkITKWin32ExportHeader.h>

#include <itkImageIOBase.h>

namespace itk
{

  /**
   * Take a peek at an image to determine the number of dimensions (eg. 2D or 3D).
   * \param typename std::string filename the filename of the image
   * \return int the number of dimensions
   * \throws typename itk::ExceptionObject if it fails for any reason.
   */
  extern "C++" NIFTKITK_WINEXPORT ITK_EXPORT int PeekAtImageDimension(std::string filename);

  /**
   * Take a peek at an image to determine the component type (eg. float, short etc. )
   * \param typename std::string filename the filename of the image
   * \return IOComponentType
   * \throws typename itk::ExceptionObject if it fails for any reason.
   */
  extern "C++" NIFTKITK_WINEXPORT ITK_EXPORT ImageIOBase::IOComponentType PeekAtComponentType(std::string filename);

  /**
   * Take a peek at an image to determine the pixel type (scalar, vector etc).
   * \param typename std::string filename the filename of the image
   * \return IOPixelType
   * \throws typename itk::ExceptionObject if it fails for any reason.
   */
  extern "C++" NIFTKITK_WINEXPORT ITK_EXPORT ImageIOBase::IOPixelType PeekAtPixelType(std::string filename);

  /**
   * Take a peek at an image to determine the dimension of an image based
   * on the number of voxels (e.g. Nx,Ny,1 is a 2D image).
   * \param typename std::string filename the filename of the image
   * \return SizeType
   * \throws typename itk::ExceptionObject if it fails for any reason.
   */
  extern "C++" NIFTKITK_WINEXPORT ITK_EXPORT int PeekAtImageDimensionFromSizeInVoxels(std::string filename);

  /** Helper method called by above. */
  extern "C++" NIFTKITK_WINEXPORT ITK_EXPORT void InitialiseImageIO(std::string filename, ImageIOBase::Pointer& imageIO);

  /**
   * Prints out a reasonable message from an exception, so we can log it, but
   * ideally, you should probably rethrow it aswell.
   * \param typename itk::ExceptionObject err the exception
   * \return std::string a formatted exception string
   */
  extern "C++" NIFTKITK_WINEXPORT ITK_EXPORT std::string GetExceptionString(ExceptionObject& err);

} // end namespace

#endif
