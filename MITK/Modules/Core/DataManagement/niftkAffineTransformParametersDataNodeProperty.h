/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkAffineTransformParametersDataNodeProperty_h
#define niftkAffineTransformParametersDataNodeProperty_h

#include "niftkCoreExports.h"

#include <algorithm>

#include <mitkBaseProperty.h>

namespace niftk
{

/**
 * \class AffineTransformParametersProperty
 * \brief MITK data-node property suitable for holding affine transform parameters.
 *
 * The parameters are defined to be an array of doubles in this order:
 * rx, ry, rz, tx, ty, tz, sx, sy, sz, k1, k2, k3 and cc (0 or 1)
 * to describe if we are rotating about centre or not.
 *
 * So, exactly 13 doubles long.
 */
class NIFTKCORE_EXPORT AffineTransformParametersDataNodeProperty : public mitk::BaseProperty
{

public:

  mitkClassMacro(AffineTransformParametersDataNodeProperty, mitk::BaseProperty)
  itkNewMacro(AffineTransformParametersDataNodeProperty)
  mitkNewMacro1Param(AffineTransformParametersDataNodeProperty, const std::vector<double>&)

  /**
   * \brief The ParametersType is defined to be an array of double, it should be exactly 13 doubles long.
   */
  typedef std::vector<double> ParametersType;

  /**
   * \brief Get the parameters from this property object.
   */
  const ParametersType& GetAffineTransformParameters() const;

  /**
   * \brief Set the parameters on this property object.
   */
  void SetAffineTransformParameters(const ParametersType& parameters);

  /**
   * \brief Defined in base class, returns the current value as a string for display in property view.
   */
  virtual std::string GetValueAsString() const override;

  /**
   * \brief Method to set these parameters back to identity.
   */
  virtual void Identity();

protected:

  virtual ~AffineTransformParametersDataNodeProperty();
  AffineTransformParametersDataNodeProperty();                                                       // Purposefully hidden.
  AffineTransformParametersDataNodeProperty(const AffineTransformParametersDataNodeProperty& other); // Purposefully hidden.
  AffineTransformParametersDataNodeProperty(const ParametersType& parameters);                       // Purposefully hidden.

  /**
   * \see mitk::BaseProperty::IsEqual()
   */
  virtual bool IsEqual(const mitk::BaseProperty& property) const override;

  /**
   * \see mitk::BaseProperty::Assign()
   */
  virtual bool Assign(const mitk::BaseProperty& ) override;

private:

  AffineTransformParametersDataNodeProperty& operator=(const AffineTransformParametersDataNodeProperty&); // Purposefully not implemented.
  itk::LightObject::Pointer InternalClone() const override;

  ParametersType m_Parameters;
};

}

#endif
