/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkSurfaceReconstruction_h
#define mitkSurfaceReconstruction_h

#include "niftkIGIExports.h"
#include <mitkDataStorage.h>
#include <mitkImage.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

namespace mitk {

/**
 * \class SurfaceReconstruction
 * \brief Takes image data, and calculates a surface reconstruction, and write a point cloud to mitk::DataStorage.
 */
class NIFTKIGI_EXPORT SurfaceReconstruction : public itk::Object
{
public:

  mitkClassMacro(SurfaceReconstruction, itk::Object);
  itkNewMacro(SurfaceReconstruction);

  /**
   * \brief Write My Documentation
   */
  void Run(const mitk::DataStorage::Pointer dataStorage,
           const mitk::Image::Pointer image1,
           const mitk::Image::Pointer image2);

protected:

  SurfaceReconstruction(); // Purposefully hidden.
  virtual ~SurfaceReconstruction(); // Purposefully hidden.

  SurfaceReconstruction(const SurfaceReconstruction&); // Purposefully not implemented.
  SurfaceReconstruction& operator=(const SurfaceReconstruction&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
