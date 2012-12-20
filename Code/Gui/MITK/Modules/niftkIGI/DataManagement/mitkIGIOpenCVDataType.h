/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKIGIOPENCVDATATYPE_H
#define MITKIGIOPENCVDATATYPE_H

#include "niftkIGIExports.h"
#include "mitkIGIDataType.h"
#include <cv.h>

namespace mitk
{

/**
 * \class IGIOpenCVDataType
 * \brief Class to represent video frame data from OpenCV, to integrate within the niftkIGI framework.
 */
class NIFTKIGI_EXPORT IGIOpenCVDataType : public IGIDataType
{
public:

  mitkClassMacro(IGIOpenCVDataType, IGIDataType);
  itkNewMacro(IGIOpenCVDataType);

  /**
   * \brief Used for loading in an image, see mitk::OpenCVVideoSource
   */
  void CloneImage(const IplImage *image);

  /**
   * \brief Returns the internal image, so do not modify it.
   */
  const IplImage* GetImage();

protected:

  IGIOpenCVDataType(); // Purposefully hidden.
  virtual ~IGIOpenCVDataType(); // Purposefully hidden.

  IGIOpenCVDataType(const IGIOpenCVDataType&); // Purposefully not implemented.
  IGIOpenCVDataType& operator=(const IGIOpenCVDataType&); // Purposefully not implemented.

private:

  IplImage *m_Image;

};

} // end namespace

#endif // MITKIGIDATATYPE_H
