/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkPointsAndNormalsBasedRegistration_h
#define mitkPointsAndNormalsBasedRegistration_h

#include "niftkIGIExports.h"
#include <mitkDataStorage.h>
#include <vtkMatrix4x4.h>
#include <mitkPointSet.h>
#include <mitkDataNode.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

namespace mitk {

/**
 * \class PointsAndNormalsBasedRegistration
 * \brief Class to implement Liu, Fitzpatrick 2003 to register points with normals.
 *
 * See also <a href="http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=758228">this link</a>.
 */
class NIFTKIGI_EXPORT PointsAndNormalsBasedRegistration : public itk::Object
{
public:

  mitkClassMacro(PointsAndNormalsBasedRegistration, itk::Object);
  itkNewMacro(PointsAndNormalsBasedRegistration);

  /**
   * \brief Main method to calculate the point based registration.
   * \param[In] fixedPointSet a point set
   * \param[In] movingPointSet a point set
   * \param[In] fixedNormals a point set containing the corresponding normals for fixedPointSet.
   * \param[In] movingNormals a point set containing the corresponding normals for movingPointSet.
   * \param[In,Out] the transformation to transform the moving point set into the coordinate system of the fixed point set.
   * \return Returns the Fiducial Registration Error
   */
  double Update(const mitk::PointSet::Pointer fixedPointSet,
              const mitk::PointSet::Pointer movingPointSet,
              const mitk::PointSet::Pointer fixedNormals,
              const mitk::PointSet::Pointer movingNormals,
              vtkMatrix4x4& outputTransform) const;

protected:

  PointsAndNormalsBasedRegistration(); // Purposefully hidden.
  virtual ~PointsAndNormalsBasedRegistration(); // Purposefully hidden.

  PointsAndNormalsBasedRegistration(const PointsAndNormalsBasedRegistration&); // Purposefully not implemented.
  PointsAndNormalsBasedRegistration& operator=(const PointsAndNormalsBasedRegistration&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
