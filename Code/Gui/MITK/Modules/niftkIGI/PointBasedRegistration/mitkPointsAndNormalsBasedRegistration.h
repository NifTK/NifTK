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
   * \brief Stores the default value of UsePointIDToMatchPoints = true.
   */
  static const bool DEFAULT_USE_POINT_ID_TO_MATCH;

  /**
   * \brief Stores the default value of UseTwoPhase = true.
   */
  static const bool DEFAULT_USE_TWO_PHASE;

  /**
   * \brief Stores the default value of UseExhaustiveSearch = true.
   */
  static const bool DEFAULT_USE_EXHAUSTIVE_SEARCH;

  /**
   * \brief If true, will try to filter matching pairs of points using the mitk::PointSet PointID feature.
   */
  itkSetMacro(UsePointIDToMatchPoints, bool);
  itkGetMacro(UsePointIDToMatchPoints, bool);

  /**
   * \brief If true, do two phase registration, just point based, and then points and normals.
   */
  itkSetMacro(UseTwoPhase, bool);
  itkGetMacro(UseTwoPhase, bool);

  /**
   * \brief If true, will also do an exhaustive search for best fit of 2 or more points.
   */
  itkSetMacro(UseExhaustiveSearch, bool);
  itkGetMacro(UseExhaustiveSearch, bool);

  /**
   * \brief Main method to calculate the point based registration.
   * \param[In] fixedPointSet a point set
   * \param[In] movingPointSet a point set
   * \param[In] fixedNormals a point set containing the corresponding normals for fixedPointSet.
   * \param[In] movingNormals a point set containing the corresponding normals for movingPointSet.
   * \param[Out] outputTransform the transformation to transform the moving point set into the coordinate system of the fixed point set.
   * \param[Out] fiducialRegistrationError the Fiducial Registration Error
   * \return returns true if the registration was successful and false otherwise. By successful, we mean
   * the computation was deemed valid, and we are not saying whether the Fiducial Registration Error is "acceptable" or not.
   */
  bool Update(const mitk::PointSet::Pointer fixedPointSet,
              const mitk::PointSet::Pointer movingPointSet,
              const mitk::PointSet::Pointer fixedNormals,
              const mitk::PointSet::Pointer movingNormals,
              vtkMatrix4x4& outputTransform,
              double& fiducialRegistrationError) const;

protected:

  PointsAndNormalsBasedRegistration(); // Purposefully hidden.
  virtual ~PointsAndNormalsBasedRegistration(); // Purposefully hidden.

  PointsAndNormalsBasedRegistration(const PointsAndNormalsBasedRegistration&); // Purposefully not implemented.
  PointsAndNormalsBasedRegistration& operator=(const PointsAndNormalsBasedRegistration&); // Purposefully not implemented.

private:

  bool m_UsePointIDToMatchPoints;
  bool m_UseTwoPhase;
  bool m_UseExhaustiveSearch;

}; // end class

} // end namespace

#endif
