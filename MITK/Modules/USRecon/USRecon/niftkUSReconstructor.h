/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkUSReconstructor_h
#define niftkUSReconstructor_h

#include "niftkUSReconExports.h"
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <mitkCommon.h>
#include <mitkImage.h>
#include <niftkCoordinateAxesData.h>
#include <memory>

namespace niftk
{

class USReconstructorPrivate;

/**
 * \class USReconstructor
 * \brief Does US reconstruction.
 *
 *
 * All errors must be thrown as a subclass of mitk::Exception.
 */
class NIFTKUSRECON_EXPORT USReconstructor : public itk::Object
{
public:

  mitkClassMacroItkParent(USReconstructor, itk::Object)
  itkNewMacro(USReconstructor)

  /**
   * \brief Main Reconstruction algorithm, which returns a new image.
   */
  mitk::Image::Pointer DoReconstruction();

  /**
   * \brief Clear data down (reset).
   */
  void ClearData();

  /**
   * \brief Accumulates an image and tracking transform.
   */
  void AddPair(mitk::Image::Pointer image, niftk::CoordinateAxesData::Pointer transform);

protected:

  /**
   * \brief Constructor
   * \param outputDirName output directory name
   */
  USReconstructor();          // Purposefully hidden.
  virtual ~USReconstructor(); // Purposefully hidden.

  USReconstructor(const USReconstructor&);            // Purposefully not implemented.
  USReconstructor& operator=(const USReconstructor&); // Purposefully not implemented.

private:

  std::unique_ptr<USReconstructorPrivate> m_Impl;

}; // end class

} // end namespace

#endif
