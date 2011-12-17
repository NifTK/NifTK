/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 20:57:34 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7341 $
 Last modified by  : $Author: ad $
 
 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef _itkMultiResolutionDeformableImageRegistrationMethod_txx
#define _itkMultiResolutionDeformableImageRegistrationMethod_txx

#include "itkMultiResolutionDeformableImageRegistrationMethod.h"
#include "itkDeformableTransform.h"

#include "itkLogHelper.h"

namespace itk
{
/*
 * Constructor
 */
template <typename TInputImageType, class TScalarType, unsigned int NDimensions, class TDeformationScalar, class TPyramidFilter>
MultiResolutionDeformableImageRegistrationMethod<TInputImageType, TScalarType, NDimensions, TDeformationScalar, TPyramidFilter>
::MultiResolutionDeformableImageRegistrationMethod()
{
  m_JacobianImageFileName = std::string("");
  m_JacobianImageFileExtension = std::string("");
  m_WriteJacobianImageAtEachLevel = false;
  m_VectorImageFileName = std::string("");
  m_VectorImageFileExtension = std::string("");
  m_WriteVectorImageAtEachLevel = false;
  m_ParameterFileName = ""; 
  m_WriteParametersAtEachLevel = false; 
  
  niftkitkDebugMacro(<<"MultiResolutionDeformableImageRegistrationMethod():Constructed with" \
      << " m_JacobianImageFileName=" << m_JacobianImageFileName \
      << ", m_JacobianImageFileExtension=" << m_JacobianImageFileExtension \
      << ", m_WriteJacobianImageAtEachLevel=" << m_WriteJacobianImageAtEachLevel \
      << ", m_VectorImageFileName=" << m_VectorImageFileName \
      << ", m_VectorImageFileExtension=" << m_VectorImageFileExtension \
      << ", m_WriteVectorImageAtEachLevel=" << m_WriteVectorImageAtEachLevel);
}

template <typename TInputImageType, class TScalarType, unsigned int NDimensions, class TDeformationScalar, class TPyramidFilter>
void
MultiResolutionDeformableImageRegistrationMethod<TInputImageType, TScalarType, NDimensions, TDeformationScalar, TPyramidFilter>
::WriteJacobianImage(std::string filename)
{
  // We need the re-interpret cast as DeformableTransform has different template parameters to base class Transform
  DeformableTransformPointer transform = dynamic_cast<DeformableTransformPointer>(this->GetSingleResMethod()->GetTransform());
  if (transform == NULL)
    {
      itkExceptionMacro(<<"Transform is null, the transform should be a subclass of itkDeformableTransform.");
    }
  else
    {
      transform->WriteJacobianImage(filename);  
    }
}

template <typename TInputImageType, class TScalarType, unsigned int NDimensions, class TDeformationScalar, class TPyramidFilter>
void
MultiResolutionDeformableImageRegistrationMethod<TInputImageType, TScalarType, NDimensions, TDeformationScalar, TPyramidFilter>
::WriteVectorImage(std::string filename)
{
  // We need the re-interpret cast as DeformableTransform has different template parameters to base class Transform
  DeformableTransformPointer transform = dynamic_cast<DeformableTransformPointer>(this->GetSingleResMethod()->GetTransform());
  if (transform == NULL)
    {
      itkExceptionMacro(<<"Transform is null, the transform should be a subclass of itkDeformableTransform.");
    }
  else
    {
      transform->WriteVectorImage(filename);  
    }
}

template <typename TInputImageType, class TScalarType, unsigned int NDimensions, class TDeformationScalar, class TPyramidFilter>
void
MultiResolutionDeformableImageRegistrationMethod<TInputImageType, TScalarType, NDimensions, TDeformationScalar, TPyramidFilter>
::WriteParameters(std::string filename)
{
  // We need the re-interpret cast as DeformableTransform has different template parameters to base class Transform
  DeformableTransformPointer transform = dynamic_cast<DeformableTransformPointer>(this->GetSingleResMethod()->GetTransform());
  if (transform == NULL)
    {
      itkExceptionMacro(<<"Transform is null, the transform should be a subclass of itkDeformableTransform.");
    }
  else
    {
      transform->WriteParameters(filename);  
    }
}

template <typename TInputImageType, class TScalarType, unsigned int NDimensions, class TDeformationScalar, class TPyramidFilter>
void
MultiResolutionDeformableImageRegistrationMethod<TInputImageType, TScalarType, NDimensions, TDeformationScalar, TPyramidFilter>
::AfterSingleResolutionRegistration()
{
  niftkitkDebugMacro(<<"AfterSingleResolutionRegistration():Started");
  
  Superclass::AfterSingleResolutionRegistration();
  
  if (m_WriteJacobianImageAtEachLevel)
    {
      niftkitkDebugMacro(<<"AfterSingleResolutionRegistration():m_WriteJacobianImageAtEachLevel must be true");
      this->WriteJacobianImageForLevel();  
    }
  else if (m_JacobianImageFileName.length() > 0 && m_JacobianImageFileExtension.length() > 0)
    {
      niftkitkDebugMacro(<<"AfterSingleResolutionRegistration():Writing single jacobian image");
      this->WriteJacobianImage();
    }

  if (m_WriteVectorImageAtEachLevel)
    {
      niftkitkDebugMacro(<<"AfterSingleResolutionRegistration():m_WriteVectorImageAtEachLevel must be true");
      this->WriteVectorImageForLevel();  
    }
  else if (m_VectorImageFileName.length() > 0 && m_VectorImageFileExtension.length() > 0)
    {
      niftkitkDebugMacro(<<"AfterSingleResolutionRegistration():Writing single vector image");
      this->WriteVectorImage();
    }

  if (m_WriteParametersAtEachLevel)
    {
      niftkitkDebugMacro(<<"AfterSingleResolutionRegistration():m_WriteParametersAtEachLevel must be true");
      this->WriteParametersForLevel();  
    }
  else if (m_ParameterFileName.length() > 0 && m_ParameterFileExt.length() > 0)
    {
      niftkitkDebugMacro(<<"AfterSingleResolutionRegistration():Writing single parameter file");
      this->WriteParameters();
    }

  niftkitkDebugMacro(<<"AfterSingleResolutionRegistration():Finished");
}

} // end namespace

#endif
