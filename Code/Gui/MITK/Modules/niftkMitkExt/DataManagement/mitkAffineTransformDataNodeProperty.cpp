/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-31 06:45:24 +0000 (Mon, 31 Oct 2011) $
 Revision          : $Rev: 7634 $
 Last modified by  : $Author: mjc $

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkAffineTransformDataNodeProperty.h"

namespace mitk
{

const std::string AffineTransformDataNodeProperty::PropertyKey = "niftk.affinetransform";

vtkSmartPointer<vtkMatrix4x4> AffineTransformDataNodeProperty::LoadTransformFromNode(const std::string propertyName, const mitk::DataNode &node) {
  vtkSmartPointer<vtkMatrix4x4> sp_transform;
  AffineTransformDataNodeProperty *p_property;

  sp_transform = vtkSmartPointer<vtkMatrix4x4>::New();
  if ((p_property = dynamic_cast<AffineTransformDataNodeProperty*>(node.GetProperty(propertyName.c_str())))) {
    sp_transform->DeepCopy(&p_property->GetTransform().Element[0][0]);
  } else {
    sp_transform->Identity();
  }

  return sp_transform;
}

void AffineTransformDataNodeProperty::StoreTransformInNode(const std::string propertyName, const vtkMatrix4x4 &transform, mitk::DataNode &r_node) {
  AffineTransformDataNodeProperty::Pointer sp_property;

  sp_property = AffineTransformDataNodeProperty::New();
  sp_property->SetTransform(transform);
  r_node.AddProperty(PropertyKey.c_str(), sp_property, NULL, true);
}

std::string AffineTransformDataNodeProperty::GetValueAsString() const
{
  std::stringstream myStr;
  myStr <<   "[" << msp_Transform->GetElement(0, 0) \
        << ", " << msp_Transform->GetElement(0, 1) \
        << ", " << msp_Transform->GetElement(0, 2) \
        << ", " << msp_Transform->GetElement(0, 3)<< "]" \
        << ", [" << msp_Transform->GetElement(1, 0) \
        << ", " << msp_Transform->GetElement(1, 1) \
        << ", " << msp_Transform->GetElement(1, 2) \
        << ", " << msp_Transform->GetElement(1, 3) << "]" \
        << ", [" << msp_Transform->GetElement(2, 0) \
        << ", " << msp_Transform->GetElement(2, 1) \
        << ", " << msp_Transform->GetElement(2, 2) \
        << ", " << msp_Transform->GetElement(2, 3) << "]" \
        << ", [" << msp_Transform->GetElement(3, 0) \
        << ", " << msp_Transform->GetElement(3, 1) \
        << ", " << msp_Transform->GetElement(3, 2)  \
        << ", " << msp_Transform->GetElement(3, 3) << "]" \
        ;
  return myStr.str();
}

bool AffineTransformDataNodeProperty::IsEqual(const BaseProperty& property) const
{
  const Self *pc_affineProp = dynamic_cast<const Self*>(&property);

  if(pc_affineProp==NULL) return false;

  return (std::equal(&pc_affineProp->GetTransform().Element[0][0], &pc_affineProp->GetTransform().Element[0][0] + 16, &GetTransform().Element[0][0]));
}

bool AffineTransformDataNodeProperty::Assign(const BaseProperty& property)
{
  const Self *pc_affineProp = dynamic_cast<const Self*>(&property);

  if(pc_affineProp==NULL) return false;

  msp_Transform->DeepCopy(&pc_affineProp->GetTransform().Element[0][0]);

  return true;
}

} //end namespace
