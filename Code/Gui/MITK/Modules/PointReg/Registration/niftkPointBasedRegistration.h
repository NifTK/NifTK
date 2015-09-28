/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPointBasedRegistration_h
#define niftkPointBasedRegistration_h

#include <niftkPointRegExports.h>

#include <vtkMatrix4x4.h>
#include <mitkPointSet.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

namespace niftk {

namespace PointBasedRegistrationConstants
{
const bool DEFAULT_USE_ICP_INITIALISATION = false;
const bool DEFAULT_USE_POINT_ID_TO_MATCH = false;
const bool DEFAULT_STRIP_NAN_FROM_INPUT = true;
}

/**
* \class PointBasedRegistration
* \brief Class to implement point based registration of two point sets.
*
* This class is called from both PointRegView and TagTrackerView.
* This is two different use-cases, and the usage is quite different.
* The code is kept here in one class for convenience to the user.
*
* IMPORTANT: Must throw mitk::Exception or subclasses for all errors.
*/
class NIFTKPOINTREG_EXPORT PointBasedRegistration : public itk::Object
{
public:

  mitkClassMacroItkParent(PointBasedRegistration, itk::Object);
  itkNewMacro(PointBasedRegistration);

  /**
  * \brief If true, will try to filter matching pairs of points using the mitk::PointSet PointID feature.
  */
  itkSetMacro(UsePointIDToMatchPoints, bool);
  itkGetMacro(UsePointIDToMatchPoints, bool);

  /**
  * \brief If true, points are assumed to be unordered, and so a closest point search is used.
  */
  itkSetMacro(UseICPInitialisation, bool);
  itkGetMacro(UseICPInitialisation, bool);

  /**
  * \brief If true, both fixed and moving points are checked for NaN values prior to matching.
  */
  itkSetMacro(StripNaNFromInput, bool);
  itkGetMacro(StripNaNFromInput, bool);

  /**
  * @brief Main method to calculate the point based registration.
  * @param[In] fixedPoints a point set
  * @param[In] movingPoints a point set
  * @param[Out] outputMatrix the transformation to transform the moving
  * point set into the coordinate system of the fixed point set.
  * @return fiducial registration error
  */
  double Update(const mitk::PointSet::Pointer fixedPoints,
                const mitk::PointSet::Pointer movingPoints,
                vtkMatrix4x4& outputMatrix) const;

protected:

  PointBasedRegistration(); // Purposefully hidden.
  virtual ~PointBasedRegistration(); // Purposefully hidden.

  PointBasedRegistration(const PointBasedRegistration&); // Purposefully not implemented.
  PointBasedRegistration& operator=(const PointBasedRegistration&); // Purposefully not implemented.

private:

  bool m_UseICPInitialisation;
  bool m_UsePointIDToMatchPoints;
  bool m_StripNaNFromInput;

}; // end class

} // end namespace

#endif
