/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKIGIOPENCVDATASOURCE_H
#define MITKIGIOPENCVDATASOURCE_H

#include "niftkIGIExports.h"
#include "mitkIGILocalDataSource.h"

namespace mitk
{

/**
 * \class IGIOpenCVDataSource
 * \brief Data source that provides access to a local video frame grabber using OpenCV
 */
class NIFTKIGI_EXPORT IGIOpenCVDataSource : public IGILocalDataSource
{

public:

  mitkClassMacro(IGIOpenCVDataSource, IGILocalDataSource);
  itkNewMacro(IGIOpenCVDataSource);

  /**
   * \brief Defined in base class, so we check that the data type is in fact
   * a mitk::IGIOpenCVDataType, returning true if it is and false otherwise.
   * \see mitk::IGIDataSource::CanHandleData()
   */
  virtual bool CanHandleData(mitk::IGIDataType* data) const;

protected:

  IGIOpenCVDataSource(); // Purposefully hidden.
  virtual ~IGIOpenCVDataSource(); // Purposefully hidden.

  IGIOpenCVDataSource(const IGIOpenCVDataSource&); // Purposefully not implemented.
  IGIOpenCVDataSource& operator=(const IGIOpenCVDataSource&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif

