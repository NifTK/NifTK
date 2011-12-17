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

#ifndef MITKDRCITKIMAGEFILEREADER_H
#define MITKDRCITKIMAGEFILEREADER_H

#include "niftkMitkExtExports.h"
#include "mitkCommon.h"
#include "mitkItkImageFileReader.h"
#include "itkImage.h"

namespace mitk {

/**
 * \class DRCItkImageFileReader
 * \brief DRC specific version of file reader that uses ITK readers to read images, APART
 * from Analyze, where we override the normal pipeline to cope with DRC specific versions.
 */
class /*NIFTKMITKEXT_EXPORT*/ DRCItkImageFileReader : public ItkImageFileReader
{
public:

    mitkClassMacro(DRCItkImageFileReader, ItkImageFileReader);

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

protected:

    virtual void GenerateData();

    DRCItkImageFileReader();
    ~DRCItkImageFileReader();

private:

    template<unsigned int VImageDimension, typename TPixel>
    bool LoadImageUsingItk(
        mitk::Image::Pointer mitkImage,
        std::string fileName
        );

};

} // namespace mitk

#endif // MITKDRCITKIMAGEFILEREADER_H
