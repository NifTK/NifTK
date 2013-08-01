/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkAffineTransformDataNodeProperty_h
#define mitkAffineTransformDataNodeProperty_h

#include "niftkCoreExports.h"
#include <mitkBaseProperty.h>
#include <mitkDataNode.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <algorithm>

namespace mitk {

/**
 * \class AffineTransformDataNodeProperty
 * \brief MITK data-node property suitable for holding affine transforms
 */
class NIFTKCORE_EXPORT AffineTransformDataNodeProperty : public mitk::BaseProperty {

public:

  mitkClassMacro(AffineTransformDataNodeProperty, mitk::BaseProperty);
  itkNewMacro(Self);

  Pointer Clone() const;

  /**
   * \return R/W access to transform.
   */
  vtkMatrix4x4& GetTransform(void) {
    return *msp_Transform;
  }

  /**
   * \return R/O access to transform.
   */
  const vtkMatrix4x4& GetTransform(void) const {
    return *msp_Transform;
  }

  /**
   * \brief Sets transform
   *
   *
   * The transform is copied to memory managed by the property object.
   */
  void SetTransform(const vtkMatrix4x4 &transform) {
    if (msp_Transform.GetPointer() == NULL)
      msp_Transform = vtkSmartPointer<vtkMatrix4x4>::New();
    msp_Transform->DeepCopy(&transform.Element[0][0]);
  }

  /**
   * \brief Defined in base class, returns the current value as a string for display in property view.
   */
  virtual std::string GetValueAsString() const;

  /**
   * \brief Method to set these parameters back to identity.
   */
  virtual void Identity() {
    msp_Transform->Identity();
  }

  /**
   * \brief Reads an affine transform from an MITK data node.
   *
   *
   * If there is no affine transform set for the given node, the identity transform is returned.
   */
  static vtkSmartPointer<vtkMatrix4x4> LoadTransformFromNode(const std::string propertyName, const mitk::DataNode &node);

  /**
   * \brief Writes an affine transform to a node (as a 4x4 VTK Matrix)
   */
  static void StoreTransformInNode(const std::string propertyName, const vtkMatrix4x4 &transform, mitk::DataNode &r_node);

protected:

  virtual ~AffineTransformDataNodeProperty(void) {}
  AffineTransformDataNodeProperty(void);                                         // Purposefully hidden.
  AffineTransformDataNodeProperty(const AffineTransformDataNodeProperty& other); // Purposefully hidden.

private:

  AffineTransformDataNodeProperty& operator=(const AffineTransformDataNodeProperty&); // Purposefully not implemented.
  itk::LightObject::Pointer InternalClone() const;

  /**
   * \see mitk::BaseProperty::IsEqual()
   */
  virtual bool IsEqual(const BaseProperty& property) const;

  /**
   * \see mitk::BaseProperty::Assign()
   */
  virtual bool Assign(const BaseProperty& );

  vtkSmartPointer<vtkMatrix4x4> msp_Transform;
};

} // end namespace

#endif
