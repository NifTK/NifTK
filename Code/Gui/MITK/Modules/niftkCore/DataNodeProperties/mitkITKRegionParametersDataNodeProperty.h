/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKITKREGIONPARAMETERSDATANODEPROPERTY_H
#define MITKITKREGIONPARAMETERSDATANODEPROPERTY_H

#include "niftkCoreExports.h"
#include "mitkBaseProperty.h"
#include <algorithm>

namespace mitk {

/**
 * \class ITKRegionParametersDataNodeProperty
 * \brief MITK data-node property suitable for holding an ITK Region, consisting of a "valid" flag,
 * plus 6 parameters containing Size and Index, as a simple vector of integers.
 */
class NIFTKCORE_EXPORT ITKRegionParametersDataNodeProperty : public BaseProperty
{

public:

  mitkClassMacro(ITKRegionParametersDataNodeProperty, BaseProperty);
  itkNewMacro(ITKRegionParametersDataNodeProperty);

  Pointer Clone() const;

  /**
   * \brief Parameters are 6 integers, corresponding to size[X,Y,Z], index[X,Y,Z].
   */
  typedef std::vector<int> ParametersType;

  /**
   * \brief Get the region parameters from this property object where size[X,Y,Z] = [0-2], and index[X,Y,Z] = [3-5].
   */
  const ParametersType& GetITKRegionParameters() const;

  /**
   * \brief Set the region parameters on this property object where size[X,Y,Z] = [0-2], and index[X,Y,Z] = [3-5].
   */
  void SetITKRegionParameters(const ParametersType& parameters);

  /**
   * \brief Returns true of the size of the volume is at least 1 voxel (eg. 1x1x1).
   */
  bool HasVolume() const;

  /**
   * \brief Method to set the size.
   */
  void SetSize(int x, int y, int z);

  /**
   * \brief Get the m_IsValid status flag.
   */
  bool IsValid() const;

  /**
   * \brief Set the isValid status flag.
   */
  void SetValid(bool valid);

  /**
   * \brief Defined in base class, returns the current value as a string for display in property view.
   */
  virtual std::string GetValueAsString() const;

  /**
   * \brief Method to set these parameters back to identity, which is [false, 0,0,0,0,0,0].
   */
  virtual void Identity();

protected:

  virtual ~ITKRegionParametersDataNodeProperty();
  ITKRegionParametersDataNodeProperty();                                                 // Purposefully hidden.
  ITKRegionParametersDataNodeProperty(const ITKRegionParametersDataNodeProperty& other); // Purposefully hidden.

  /**
   * \see mitk::BaseProperty::IsEqual()
   */
  virtual bool IsEqual(const BaseProperty& property) const;

  /**
   * \see mitk::BaseProperty::Assign()
   */
  virtual bool Assign(const BaseProperty& );

private:

  ITKRegionParametersDataNodeProperty& operator=(const ITKRegionParametersDataNodeProperty&); // Purposefully not implemented.
  itk::LightObject::Pointer InternalClone() const;

  ParametersType m_Parameters;
  bool           m_IsValid;
};

} // namespace mitk

#endif /* MITKITKREGIONPARAMETERSDATANODEPROPERTY_H */
