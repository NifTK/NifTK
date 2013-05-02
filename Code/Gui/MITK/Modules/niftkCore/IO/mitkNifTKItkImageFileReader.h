/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef NifTKItkImageFileReader_h
#define NifTKItkImageFileReader_h

#include "niftkCoreExports.h"
#include "mitkCommon.h"
#include "mitkItkImageFileReader.h"
#include "itkImage.h"

namespace mitk {

/**
 * \class NifTKItkImageFileReader
 * \brief NifTK specific version of an ItkImageFileReader that uses ITK readers to read images, APART
 * from Analyze and Nifti, where we override the normal ITK classes and install NifTK specific ones.
 *
 * For Analyze images, we flip them to cope with DRC conventions.
 * For Nifti images, we include the sform if present.
 */
class /*NIFTKCORE_EXPORT*/ NifTKItkImageFileReader : public ItkImageFileReader
{
public:

    mitkClassMacro(NifTKItkImageFileReader, ItkImageFileReader);

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

protected:

    virtual void GenerateData();

    NifTKItkImageFileReader();
    ~NifTKItkImageFileReader();

private:

    template<unsigned int VImageDimension, typename TPixel>
    bool LoadImageUsingItk(
        mitk::Image::Pointer mitkImage,
        std::string fileName
        );

};

} // namespace mitk

#endif
