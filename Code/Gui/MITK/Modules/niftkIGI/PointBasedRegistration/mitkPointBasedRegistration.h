/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkPointBasedRegistration_h
#define mitkPointBasedRegistration_h

#include "niftkIGIExports.h"
#include <mitkDataStorage.h>
#include <vtkMatrix4x4.h>
#include <mitkPointSet.h>
#include <mitkDataNode.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

namespace mitk {

/**
 * \class PointBasedRegistration
 * \brief Class to implement point based registration of two point sets.
 *
 * This class is called from both PointRegView and TagTrackerView.
 * This is two different use-cases, and the usage is quite different.
 * The code is kept here in one class for convenience to the user.
 */
class NIFTKIGI_EXPORT PointBasedRegistration : public itk::Object
{
public:

  mitkClassMacro(PointBasedRegistration, itk::Object);
  itkNewMacro(PointBasedRegistration);

  /**
   * \brief Stores the default value of UseICPInitialisation = false.
   */
  static const bool DEFAULT_USE_ICP_INITIALISATION;

  /**
   * \brief Stores the default value of UsePointIDToMatchPoints = false.
   */
  static const bool DEFAULT_USE_POINT_ID_TO_MATCH;

  /**
   * \brief Stores the default value of UseSVDBasedMethod = true.
   */
  static const bool DEFAULT_USE_SVD_BASED_METHOD;

  /**
   * \brief If true, will try to filter matching pairs of points using the mitk::PointSet PointID feature.
   */
  itkSetMacro(UsePointIDToMatchPoints, bool);
  itkGetMacro(UsePointIDToMatchPoints, bool);

  /**
   * \brief If true, will use an SVD based match, so UseICPInitialisation is irrelevant, and if false will use MITK LandmarkTransformFilter.
   */
  itkSetMacro(UseSVDBasedMethod, bool);
  itkGetMacro(UseSVDBasedMethod, bool);

  /**
   * \brief If true, points are assumed to be unordered, and so an closest point search is used.
   * Not relevant if you are doing SVD.
   */
  itkSetMacro(UseICPInitialisation, bool);
  itkGetMacro(UseICPInitialisation, bool);

  /**
   * \brief Main method to calculate the point based registration.
   * \param[In] fixedPointSet a point set
   * \param[In] movingPointSet a point set
   * \param[Out] outputTransform the transformation to transform the moving point set into the coordinate system of the fixed point set.
   * \param[Out] fiducialRegistrationError the Fiducial Registration Error
   * \return returns true if the registration was successful and false otherwise. By successful, we mean
   * the computation was deemed valid, and we are not saying whether the Fiducial Registration Error is "acceptable" or not.
   */
  bool Update(const mitk::PointSet::Pointer fixedPointSet,
              const mitk::PointSet::Pointer movingPointSet,
              vtkMatrix4x4& outputTransform,
              double& fiducialRegistrationError) const;

protected:

  PointBasedRegistration(); // Purposefully hidden.
  virtual ~PointBasedRegistration(); // Purposefully hidden.

  PointBasedRegistration(const PointBasedRegistration&); // Purposefully not implemented.
  PointBasedRegistration& operator=(const PointBasedRegistration&); // Purposefully not implemented.

private:

  bool m_UseICPInitialisation;
  bool m_UsePointIDToMatchPoints;
  bool m_UseSVDBasedMethod;

}; // end class

} // end namespace

#endif
