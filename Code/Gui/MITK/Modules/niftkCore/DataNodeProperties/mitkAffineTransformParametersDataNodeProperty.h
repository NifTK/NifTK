/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKAFFINETRANSFORMPARAMETERSDATANODEPROPERTY_H
#define MITKAFFINETRANSFORMPARAMETERSDATANODEPROPERTY_H

#include "niftkCoreExports.h"
#include "mitkBaseProperty.h"
#include <algorithm>

namespace mitk {

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
class NIFTKCORE_EXPORT AffineTransformParametersDataNodeProperty : public BaseProperty
{

public:

  mitkClassMacro(AffineTransformParametersDataNodeProperty, BaseProperty);
  itkNewMacro(AffineTransformParametersDataNodeProperty);
  mitkNewMacro1Param(AffineTransformParametersDataNodeProperty, const std::vector<double>&);

  /// \brief The ParametersType is defined to be an array of double, it should be exactly 13 doubles long.
  typedef std::vector<double> ParametersType;

  /// \brief Get the parameters from this property object.
  const ParametersType& GetAffineTransformParameters() const;

  /// \brief Set the parameters on this property object.
  void SetAffineTransformParameters(const ParametersType& parameters);

  /// \brief Defined in base class, returns the current value as a string for display in property view.
  virtual std::string GetValueAsString() const;

  /// \brief Method to set these parameters back to identity.
  virtual void Identity();

protected:

  AffineTransformParametersDataNodeProperty();                                  // Purposefully hidden.
  AffineTransformParametersDataNodeProperty(const ParametersType& parameters);  // Purposefully hidden.
  virtual ~AffineTransformParametersDataNodeProperty();

private:

  /*!
    Override this method in subclasses to implement a meaningful comparison. The property
    argument is guaranteed to be castable to the type of the implementing subclass.
  */
  virtual bool IsEqual(const BaseProperty& property) const;

  /*!
    Override this method in subclasses to implement a meaningful assignment. The property
    argument is guaranteed to be castable to the type of the implementing subclass.

    @warning This is not yet exception aware/safe and if this method returns false,
             this property's state might be undefined.

    @return True if the argument could be assigned to this property.
   */
  virtual bool Assign(const BaseProperty& );

  ParametersType m_Parameters;
};

} // namespace mitk

#endif /* MITKLEVELWINDOWPROPERTY_H_HEADER_INCLUDED_C10EEAA8 */
