/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCUDAImageProperty_h
#define niftkCUDAImageProperty_h

#include "niftkCUDAExports.h"
#include <CUDAImage/niftkLightweightCUDAImage.h>
#include <mitkBaseProperty.h>

namespace niftk
{

/**
* A wrapper for LightweightCUDAImage that can be attached as a property to
* an existing data object or node.
*/
class NIFTKCUDA_EXPORT CUDAImageProperty : public mitk::BaseProperty
{
public:
  mitkClassMacro(CUDAImageProperty, mitk::BaseProperty)
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)


  LightweightCUDAImage Get() const;
  void Set(LightweightCUDAImage lwci);


  /** @name As required by mitk::BaseProperty. */
  //@{
public:
  using BaseProperty::operator=;

private:
  CUDAImageProperty& operator=(const CUDAImageProperty&);

  virtual bool IsEqual(const mitk::BaseProperty& property) const;
  virtual bool Assign(const mitk::BaseProperty& property);
  //@}


private:
  LightweightCUDAImage      m_LWCI;
};

} // end namespace

#endif
