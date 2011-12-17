/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkMultiResolutionDeformableImageRegistrationMethod_h
#define __itkMultiResolutionDeformableImageRegistrationMethod_h

#include "itkMultiResolutionImageRegistrationWrapper.h"
#include "ConversionUtils.h"

namespace itk
{

/** 
 * \class MultiResolutionDeformableImageRegistrationMethod
 * \brief Extends MultiResolutionImageRegistrationWrapper to provide various common methods for FFD and Fluid,
 * such as saving Jacobian images.
 * 
 * \sa MultiResolutionImageRegistrationWrapper
 */
template <typename TInputImageType, class TScalarType, unsigned int NDimensions, class TDeformationScalar, class TPyramidFilter = itk::RecursiveMultiResolutionPyramidImageFilter< TInputImageType, TInputImageType > >
class ITK_EXPORT MultiResolutionDeformableImageRegistrationMethod 
  : public MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter> 
{
public:
  /** Standard class typedefs. */
  typedef MultiResolutionDeformableImageRegistrationMethod          Self;
  typedef MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter>  Superclass;
  typedef SmartPointer<Self>                                        Pointer;
  typedef SmartPointer<const Self>                                  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiResolutionDeformableImageRegistrationMethod, MultiResolutionImageRegistrationWrapper);

  /** Typedefs. */
  typedef TInputImageType InputImageType;
  typedef DeformableTransform<TInputImageType,TScalarType,  NDimensions, TDeformationScalar> DeformableTransformType;
  typedef DeformableTransformType*                                       DeformableTransformPointer;
  
  /** Set the filename without extension. No default. */
  itkSetMacro(JacobianImageFileName, std::string);
  itkGetMacro(JacobianImageFileName, std::string);

  /** Set the jacobian file extension. No default. */
  itkSetMacro(JacobianImageFileExtension, std::string);
  itkGetMacro(JacobianImageFileExtension, std::string);

  /** 
   * If true, we take the JacobianImageFileName and JacobianImageFileExtension
   * and write an image  JacobianImageFileName.<current level>.JacobianImageFileExtension
   */
  itkSetMacro(WriteJacobianImageAtEachLevel, bool);
  itkGetMacro(WriteJacobianImageAtEachLevel, bool);

  /** Set the filename without extension. No default. */
  itkSetMacro(VectorImageFileName, std::string);
  itkGetMacro(VectorImageFileName, std::string);

  /** Set the jacobian file extension. No default. */
  itkSetMacro(VectorImageFileExtension, std::string);
  itkGetMacro(VectorImageFileExtension, std::string);

  /** 
   * If true, we take the VectorImageFileName and JacobianFileExtension
   * and write an image  VectorImageFileName.<current level>.VectorImageFileExtension
   */
  itkSetMacro(WriteVectorImageAtEachLevel, bool);
  itkGetMacro(WriteVectorImageAtEachLevel, bool);

  /** Set the filename without extension. No default. */
  itkSetMacro(ParameterFileName, std::string);
  itkGetMacro(ParameterFileName, std::string);

  /** Set the jacobian file extension. No default. */
  itkSetMacro(ParameterFileExt, std::string);
  itkGetMacro(ParameterFileExt, std::string);

  /** 
   * If true, we take the JacobianFileExtension and ParameterFileExt
   * and write an image  ParameterFileName.<current level>.ParameterFileExt
   */
  itkSetMacro(WriteParametersAtEachLevel, bool);
  itkGetMacro(WriteParametersAtEachLevel, bool);

  /** Writes a jacobian image for the current level, using the current filename and extension. */
  virtual void WriteJacobianImageForLevel() 
    { 
      this->WriteJacobianImage(this->m_JacobianImageFileName + "." + niftk::ConvertToString((int)this->m_CurrentLevel) + "." + this->m_JacobianImageFileExtension);
    }
  
  /** Writes the jacobian image using the current filename and extension. */
  virtual void WriteJacobianImage()
    {
      this->WriteJacobianImage(this->m_JacobianImageFileName + "." + this->m_JacobianImageFileExtension);
    }

  /** Writes a vector image for the current level, using the current filename and extension. */
  virtual void WriteVectorImageForLevel() 
    { 
      this->WriteVectorImage(this->m_VectorImageFileName + "." + niftk::ConvertToString((int)this->m_CurrentLevel) + "." + this->m_VectorImageFileExtension);
    }
  
  /** Writes the vector image using the current filename and extension. */
  virtual void WriteVectorImage()
    {
      this->WriteVectorImage(this->m_VectorImageFileName + "." + this->m_VectorImageFileExtension);
    }

  /** Writes parameters out for the current level, using the current filename and extension. */
  virtual void WriteParametersForLevel() 
    { 
      this->WriteParameters(this->m_ParameterFileName + "." + niftk::ConvertToString((int)this->m_CurrentLevel) + "." + this->m_ParameterFileExt);
    }
  
  /** Writes the vector image using the current filename and extension. */
  virtual void WriteParameters()
    {
      this->WriteParameters(this->m_ParameterFileName + "." + this->m_ParameterFileExt);
    }

protected:
  MultiResolutionDeformableImageRegistrationMethod();
  virtual ~MultiResolutionDeformableImageRegistrationMethod() {};

  /** Actually writes the jacobian image (also called 'dv' image in Midas land). */
  virtual void WriteJacobianImage(std::string filename);

  /** Actually writes the vector image (also called 'stretch' file in Midas land). */
  virtual void WriteVectorImage(std::string filename);

  /** Actually writes the parameters. */
  virtual void WriteParameters(std::string filename);

  /** Used to trigger dumping the jacobian image after each resolution level. */
  virtual void AfterSingleResolutionRegistration();
  
private:
  MultiResolutionDeformableImageRegistrationMethod(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** Set the jacobian image file name, otherwise, it defaults to "jacobian". */
  std::string m_JacobianImageFileName;
  
  /** Set the jacobian image extension, otherwise, it defaults to "nii" */
  std::string m_JacobianImageFileExtension;

  /** So we can have jacobian image, at each level. */
  bool m_WriteJacobianImageAtEachLevel;
  
  /** Set the vector image file name, otherwise, it defaults to "vector". */
  std::string m_VectorImageFileName;
  
  /** Set the vector image extension, otherwise, it defaults to "vtk" */
  std::string m_VectorImageFileExtension;

  /** So we can have vector image, at each level. */
  bool m_WriteVectorImageAtEachLevel;
  
  /** So we can dump the parameters at each level. */
  std::string m_ParameterFileName;
  
  /** So we can dump the parameters at each level. */
  std::string m_ParameterFileExt;
  
  /** So we can dump the parameters at each level. */
  bool m_WriteParametersAtEachLevel;
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiResolutionDeformableImageRegistrationMethod.txx"
#endif

#endif



