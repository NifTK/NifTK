/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKNIFTKITKIMAGEFILEREADER_H
#define MITKNIFTKITKIMAGEFILEREADER_H

#include "niftkMitkExtExports.h"
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
class /*NIFTKMITKEXT_EXPORT*/ NifTKItkImageFileReader : public ItkImageFileReader
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

#endif // MITKNIFTKITKIMAGEFILEREADER_H
