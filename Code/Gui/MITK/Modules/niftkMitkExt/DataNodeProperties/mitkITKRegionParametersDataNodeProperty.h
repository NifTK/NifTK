/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKITKREGIONPARAMETERSDATANODEPROPERTY_H
#define MITKITKREGIONPARAMETERSDATANODEPROPERTY_H

#include <algorithm>
#include "niftkMitkExtExports.h"
#include "mitkBaseProperty.h"

namespace mitk {

/**
 * \class ITKRegionParametersDataNodeProperty
 * \brief MITK data-node property suitable for holding an ITK Region, consisting of a "valid" flag,
 * plus 6 parameters containing Size and Index, as a simple vector of integers.
 */
class NIFTKMITKEXT_EXPORT ITKRegionParametersDataNodeProperty : public BaseProperty
{

public:

  mitkClassMacro(ITKRegionParametersDataNodeProperty, BaseProperty);
  itkNewMacro(ITKRegionParametersDataNodeProperty);
  virtual ~ITKRegionParametersDataNodeProperty();

  /// \brief Parameters are 6 integers, corresponding to size[X,Y,Z], index[X,Y,Z].
  typedef std::vector<int> ParametersType;

  /// \brief Get the region parameters from this property object where size[X,Y,Z] = [0-2], and index[X,Y,Z] = [3-5].
  const ParametersType& GetITKRegionParameters() const;

  /// \brief Set the region parameters on this property object where size[X,Y,Z] = [0-2], and index[X,Y,Z] = [3-5].
  void SetITKRegionParameters(const ParametersType& parameters);

  /// \brief Returns true of the size of the volume is at least 1 voxel (eg. 1x1x1).
  bool HasVolume() const;

  /// \brief Method to set the size
  void SetSize(int x, int y, int z);

  /// \brief Get the m_IsValid status flag.
  bool IsValid() const;

  /// \brief Set the isValid status flag.
  void SetValid(bool valid);

  /// \brief Defined in base class, returns the current value as a string for display in property view.
  virtual std::string GetValueAsString() const;

  /// \brief Method to set these parameters back to identity, which is [false, 0,0,0,0,0,0].
  virtual void Identity();

protected:

  ITKRegionParametersDataNodeProperty();                                 // Purposefully hidden.
  ITKRegionParametersDataNodeProperty(const ParametersType& parameters); // Purposefully hidden.

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

private:

  ParametersType m_Parameters;
  bool           m_IsValid;
};

} // namespace mitk

#endif /* MITKLEVELWINDOWPROPERTY_H_HEADER_INCLUDED_C10EEAA8 */
