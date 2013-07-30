/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkDeformableransform_txx
#define __itkDeformableransform_txx

#include "itkDeformableTransform.h"
#include <itkDisplacementFieldJacobianDeterminantFilter.h>
#include <itkImageFileWriter.h>
#include <itkCastImageFilter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkVectorLinearInterpolateImageFunction.h>
#include <itkImageDuplicator.h>
#include <fstream>
#include <niftkConversionUtils.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkBSplineInterpolateImageFunction.h>

#include <itkLogHelper.h>

namespace itk
{
// Constructor with default arguments
template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::DeformableTransform():Superclass(1)
{
  this->m_DeformationField = DeformationFieldType::New();
  this->m_JacobianFilter = JacobianDeterminantFilterType::New();
  this->m_JacobianFilter->SetInput(this->m_DeformationField);
  this->m_JacobianFilter->SetUseImageSpacingOff();
  this->m_Parameters.SetSize(1);
  this->m_Parameters.Fill(0);
  this->m_InverseVoxelTolerance = 0.0001;
  this->m_InverseIterationTolerance = 0.01;
  this->m_MaxNumberOfInverseIterations = 200;
  this->m_InverseSearchRadius = 2.0;   
  
  niftkitkDebugMacro(<< "DeformableTransform():Constructed with parameters:" << this->m_Parameters);
}

// Destructor
template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::~DeformableTransform()
{
  return;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::SetIdentity()
{
  // Set all parameters to zero.
  this->m_Parameters.Fill(0);
  
  // Set deformation field to zero.
  DeformationFieldPixelType fieldValue;
  fieldValue.Fill(0);
  this->m_DeformationField->FillBuffer(fieldValue);
  
  // Useful to know when this is called.
  niftkitkDebugMacro(<< "SetIdentity():Reset parameters to zero, and deformation field to zero");
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
bool
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::IsIdentity()
{
  unsigned long int i, size;
  size = this->m_Parameters.GetSize();
  
  for (i = 0; i < size; i++)
    {
      if(this->m_Parameters.GetElement(i) != 0)
        {
          return false;
        }
    }
  return true;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void 
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::Initialize(FixedImagePointer image)
{

  // Copy image dimensions.
  DeformationFieldSpacingType spacing = image->GetSpacing();
  DeformationFieldDirectionType direction = image->GetDirection();
  DeformationFieldOriginType origin = image->GetOrigin();
  DeformationFieldSizeType size = image->GetLargestPossibleRegion().GetSize();
  DeformationFieldIndexType index = image->GetLargestPossibleRegion().GetIndex();
  DeformationFieldRegionType region;
  region.SetSize(size);
  region.SetIndex(index);

  // And set them on our deformation field.
  this->m_DeformationField->SetRegions(region);
  this->m_DeformationField->SetOrigin(origin);
  this->m_DeformationField->SetDirection(direction);
  this->m_DeformationField->SetSpacing(spacing);
  this->m_DeformationField->Allocate();
  this->m_DeformationField->Update();

  // Set deformation field to zero.
  DeformationFieldPixelType fieldValue;
  fieldValue.Fill(0);
  this->m_DeformationField->FillBuffer(fieldValue);

  // Jacobian filter always connected to deformation field, after first call to Initialize.
  this->m_JacobianFilter->SetInput(this->m_DeformationField);
  
  niftkitkDebugMacro(<< "Initialize():Set deformation field to size=" << size << ", spacing=" << spacing << ", and origin:" << origin);
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::InitialiseGlobalTransformation()
{
  ImageRegionIteratorWithIndex<DeformationFieldType> iterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion());

  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    DeformationFieldIndexType index = iterator.GetIndex();
    OutputPointType currentPoint;
    OutputPointType transformedPoint;
    OutputPointType deltaPoint;
    // Work out the deformation caused by the global affine transformation.
    this->m_DeformationField->TransformIndexToPhysicalPoint(index, currentPoint);
    transformedPoint = m_GlobalTransform->TransformPoint(currentPoint);
    for (unsigned int i = 0; i < NDimensions; i++)
      deltaPoint[i] = transformedPoint[i]-currentPoint[i];

    // Convert it back to voxel units.
    ContinuousIndex< double, NDimensions > imageDeformation;
    this->m_DeformationField->TransformPhysicalPointToContinuousIndex(deltaPoint, imageDeformation);

    // Set it.
    DeformationFieldPixelType value;
    for (unsigned int i = 0; i < NDimensions; i++)
        value[i] = imageDeformation[i];
     iterator.Set(value);
  }
  // Update the parameters.
  SetParametersFromField(this->m_DeformationField);
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
typename DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>::OutputPointType
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::TransformPoint(const InputPointType  &point ) const
{
  OutputPointType result;
  DeformationFieldIndexType index;
  DeformationFieldPixelType pixel;
  ContinuousIndex< double, NDimensions > imageDeformation;
  OutputPointType physicalDeformation; 
  DeformationFieldOriginType origin = this->m_DeformationField->GetOrigin();

  if (m_DeformationField->TransformPhysicalPointToIndex(point, index))
    {
      pixel = m_DeformationField->GetPixel(index);

      // Transform the deformation from image space to physical/world space. 
      for (unsigned int i = 0; i < NDimensions; i++)
      {
        imageDeformation[i] = pixel[i];
      }
      this->m_DeformationField->TransformContinuousIndexToPhysicalPoint(imageDeformation, physicalDeformation);
      for (unsigned int i = 0; i < NDimensions; i++)
      {
        result[i] = point[i] + (physicalDeformation[i]-origin[i]);
      }
    }

  return result;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::PrintSelf(std::ostream &os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  if (! this->m_GlobalTransform.IsNull()) 
    {
      os << indent << "Global transform: " << std::endl;
      this->m_GlobalTransform.GetPointer()->Print(os, indent.GetNextIndent());
    }
  else
    {
      os << indent << "GlobalTransform: NULL" << std::endl;
    }

  if (! this->m_DeformationField.IsNull()) 
    {
      os << indent << "Deformation field: " << std::endl;
      this->m_DeformationField.GetPointer()->Print(os, indent.GetNextIndent());
    }
  else
    {
      os << indent << "DeformationField: NULL" << std::endl;
    }

  if (! this->m_JacobianFilter.IsNull()) 
    {
      os << indent << "JacobianFilter: " << std::endl;
      this->m_JacobianFilter.GetPointer()->Print(os, indent.GetNextIndent());
    }
  else
    {
      os << indent << "JacobianFilter: NULL" << std::endl;
    }

  os << indent << "Parameter size: " << this->m_Parameters.GetSize() << std::endl;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
unsigned long int
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::GetNumberOfParametersImpliedByImage(const VectorFieldImagePointer image)
{
  VectorFieldSizeType size = image->GetLargestPossibleRegion().GetSize();
  unsigned long int totalParameters = NDimensions;
  for (unsigned int i = 0; i < NDimensions; i++)
    {
      totalParameters *= size[i];  
    }

  niftkitkDebugMacro(<< "GetNumberOfParametersImpliedByImage():image of size=" << size << ", implies:" << totalParameters << " parameters");
  return totalParameters;
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::ResizeParametersArray(const VectorFieldImagePointer image)
{
  unsigned long int totalParameters = GetNumberOfParametersImpliedByImage(image);
  
  this->m_Parameters.SetSize(totalParameters);
  
  niftkitkDebugMacro(<< "ResizeParametersArray():Using image:" << image.GetPointer() << ", set internal parameters array to size:" << this->m_Parameters.GetSize());
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::MarshallParametersToImage(VectorFieldImagePointer image)
{
  niftkitkDebugMacro(<< "MarshallParametersToImage():Starting");
  
  unsigned long int expectedParameters = this->GetNumberOfParametersImpliedByImage(image);
  
  if (expectedParameters != this->m_Parameters.GetSize())
    {
      itkExceptionMacro(<<"Parameters array is the wrong length, it is:" <<  this->m_Parameters.GetSize()
        << ", it should be:" +  expectedParameters);      
    }  

  VectorFieldIteratorType iterator(image, image->GetLargestPossibleRegion());
  VectorFieldPixelType value;

  unsigned long int parameterIndex = 0;
  unsigned int dimensionIndex = 0;
      
  for(iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
    { 
      for (dimensionIndex = 0; dimensionIndex < NDimensions; dimensionIndex++)
        {
          value[dimensionIndex] =  this->m_Parameters.GetElement(parameterIndex++);
        }
      iterator.Set(value);
    }
  
  niftkitkDebugMacro(<< "MarshallParametersToImage():Finished");
  
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::SetParametersFromField(const VectorFieldImagePointer& image, bool force)
{
  niftkitkDebugMacro(<< "SetParametersFromField():Starting, force:" << force);
  
  unsigned long int imageParameters = this->GetNumberOfParametersImpliedByImage(image);
  unsigned long int parametersArraySize = this->m_Parameters.GetSize();
  
  if (  imageParameters != this->m_Parameters.GetSize() && !force)
    {
      itkExceptionMacro(<<"Image implies:" << imageParameters << " parameters but parameters array only has:" << parametersArraySize << " elements"); 
    }
    
  if ( imageParameters != this->m_Parameters.GetSize() )
    {
      niftkitkDebugMacro(<< "SetParametersFromField():Resizing parameters to:" << imageParameters);
      this->m_Parameters.SetSize(imageParameters);
    }
  
  VectorFieldConstIteratorType iterator(image, image->GetLargestPossibleRegion());
  VectorFieldPixelType value;

  unsigned long int parameterIndex = 0;
  unsigned int dimensionIndex = 0;
        
  for(iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
    { 
      value = iterator.Get();
      for (dimensionIndex = 0; dimensionIndex < NDimensions; dimensionIndex++)
        {          
          this->m_Parameters.SetElement(parameterIndex++, value[dimensionIndex]);
        }
    }
  
  niftkitkDebugMacro(<< "SetParametersFromField():Finished");
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::ForceJacobianUpdate()
{
  // Forcing an update here. Might be able to remove it later.
  this->m_JacobianFilter->SetInput(this->m_DeformationField);
  this->m_JacobianFilter->Modified();
  this->m_JacobianFilter->UpdateLargestPossibleRegion();

  /*
  niftkitkDebugMacro(<< "ForceJacobianUpdate():Im object:" << this \
      << ", deformation field size:" << this->m_DeformationField->GetLargestPossibleRegion().GetSize() \
      << ", spacing:" << this->m_DeformationField->GetSpacing() \
      << ", jacobian output size is:" << m_JacobianFilter->GetOutput()->GetLargestPossibleRegion().GetSize() \
      << ", spacing is:" << m_JacobianFilter->GetOutput()->GetSpacing());
    */  
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
TScalarType
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::ComputeMinJacobian()
{

  TScalarType minJacobian;
  this->ForceJacobianUpdate();
  
  typedef ImageRegionConstIterator< typename JacobianDeterminantFilterType::OutputImageType > IteratorType;
  IteratorType iterator(m_JacobianFilter->GetOutput(), GetValidJacobianRegion());

  minJacobian = std::numeric_limits<TScalarType>::max();

  iterator.GoToBegin();
  while(!iterator.IsAtEnd())
    {
      if (iterator.Get() < minJacobian)
        {
          minJacobian = iterator.Get();
        }
      ++iterator;  
    }
  niftkitkDebugMacro(<< "ComputeMinJacobian():Minimum Jacobian is:" << minJacobian);

  return minJacobian;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
TScalarType
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::ComputeMaxJacobian()
{

  TScalarType maxJacobian;
  this->ForceJacobianUpdate();

  typedef ImageRegionConstIterator< typename JacobianDeterminantFilterType::OutputImageType > IteratorType;
  IteratorType iterator(m_JacobianFilter->GetOutput(), GetValidJacobianRegion());

  maxJacobian = std::numeric_limits<TScalarType>::min();

  iterator.GoToBegin();
  while(!iterator.IsAtEnd())
    {
      if (iterator.Get() > maxJacobian)
        {
          maxJacobian = iterator.Get();
        }
      ++iterator;
    }
  niftkitkDebugMacro(<< "ComputeMaxJacobian():Maximum Jacobian is:" << maxJacobian);

  return maxJacobian;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
TScalarType 
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::GetSumLogJacobianDeterminant()
{
  double sum = 0;
  double value;
  unsigned long int counter = 0;
  
  this->ForceJacobianUpdate();
  
  typedef ImageRegionConstIterator< typename JacobianDeterminantFilterType::OutputImageType > IteratorType;
  IteratorType iterator(m_JacobianFilter->GetOutput(), GetValidJacobianRegion());

  iterator.GoToBegin();
  while(!iterator.IsAtEnd())
    {
      value = iterator.Get();
      if (value > 0)
        {
          sum += value;
          counter++;
        }
      ++iterator;
    }
  sum /= (double) counter;
  
  niftkitkDebugMacro(<< "GetSumLogJacobianDeterminant():Sum is:" << sum);

  return sum;
  
}    
    
template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::WriteJacobianImage(std::string filename)
{
  niftkitkDebugMacro(<< "WriteJacobianImage():Started, filename=" << filename);
  
  this->ForceJacobianUpdate();
  
  // And write it.
  typedef float OutputPixelType;
  typedef Image<OutputPixelType, NDimensions>                 OutputImageType;  
  typedef Image<TScalarType, NDimensions>                     JacobianImageType;
  typedef CastImageFilter<JacobianImageType, OutputImageType> CastFilterType;
  typedef ImageFileWriter<OutputImageType>                    WriterType;
  
  typename CastFilterType::Pointer caster = CastFilterType::New();
  typename WriterType::Pointer writer = WriterType::New();
  
  caster->SetInput(m_JacobianFilter->GetOutput());
  
  writer->SetFileName(filename);
  writer->SetInput(caster->GetOutput());
  writer->Update();
  
  niftkitkDebugMacro(<< "WriteJacobianImage():Finished, filename=" << filename);
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::WriteMidasStrImage(std::string filename, int origin[NDimensions], typename TFixedImage::RegionType paddedDesiredRegion, const typename JacobianDeterminantFilterType::OutputImageType* jacobianImage)
{
  niftkitkDebugMacro(<< "WriteMidasStrImage():Started, filename=" << filename);
  
  this->ForceJacobianUpdate();
  
  if (jacobianImage == NULL)
  {
    jacobianImage = m_JacobianFilter->GetOutput(); 
  }
  
  ImageRegionConstIteratorWithIndex< typename JacobianDeterminantFilterType::OutputImageType > jacobianIterator(jacobianImage, paddedDesiredRegion);
  typename TFixedImage::SizeType size = paddedDesiredRegion.GetSize(); 
  typename std::ofstream outputStream(filename.c_str()); 
  niftkitkDebugMacro(<< "size=" << size);
  
  // Image size. 
  for (unsigned i = 0; i < NDimensions; i++)
  {
    int temp = size[i]; 
    outputStream.write(reinterpret_cast<char*>(&temp), sizeof(temp)); 
  }
  // Jacobian values. 
  for (jacobianIterator.GoToBegin(); !jacobianIterator.IsAtEnd(); ++jacobianIterator)
  {
    float temp = jacobianIterator.Get(); 
    outputStream.write(reinterpret_cast<char*>(&temp), sizeof(temp)); 
  }
  // Image origin. 
  for (unsigned i = 0; i < NDimensions; i++)
  {
    int temp = origin[i]; 
    outputStream.write(reinterpret_cast<char*>(&temp), sizeof(temp)); 
  }
  
  outputStream.close(); 
  niftkitkDebugMacro(<< "WriteMidasStrImage():Finished, filename=" << filename);
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::ReadMidasStrImage(std::string filename, int origin[NDimensions], typename TFixedImage::RegionType paddedDesiredRegion, typename JacobianDeterminantFilterType::OutputImageType* jacobianImage)
{
  niftkitkDebugMacro(<< "ReadMidasStrImage():Started, filename=" << filename);
  
  this->ForceJacobianUpdate();
  
  if (jacobianImage == NULL)
  {
    jacobianImage = m_JacobianFilter->GetOutput(); 
  }
  
  ImageRegionIteratorWithIndex< typename JacobianDeterminantFilterType::OutputImageType > jacobianIterator(jacobianImage, paddedDesiredRegion);
  typename TFixedImage::SizeType size = paddedDesiredRegion.GetSize(); 
  typename std::ifstream inputStream(filename.c_str()); 
  niftkitkDebugMacro(<< "size=" << size);
  
  // Image size. 
  for (unsigned i = 0; i < NDimensions; i++)
  {
    int temp = size[i]; 
    inputStream.read(reinterpret_cast<char*>(&temp), sizeof(temp)); 
  }
  // Jacobian values. 
  for (jacobianIterator.GoToBegin(); !jacobianIterator.IsAtEnd(); ++jacobianIterator)
  {
    float temp = 0.; 
    inputStream.read(reinterpret_cast<char*>(&temp), sizeof(temp)); 
    jacobianIterator.Set(temp); 
  }
  // Image origin. 
  for (unsigned i = 0; i < NDimensions; i++)
  {
    int temp = origin[i]; 
    inputStream.read(reinterpret_cast<char*>(&temp), sizeof(temp)); 
  }
  
  inputStream.close(); 
  niftkitkDebugMacro(<< "ReadMidasStrImage():Finished, filename=" << filename);
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::WriteMidasVecImage(std::string filename, int origin[NDimensions], typename TFixedImage::RegionType paddedDesiredRegion)
{
  niftkitkDebugMacro(<< "WriteMidasVecImage():Started, filename=" << filename);
  
  ImageRegionConstIteratorWithIndex< DeformationFieldType > fieldIterator(this->m_DeformationField, paddedDesiredRegion);
  typename TFixedImage::SizeType size = paddedDesiredRegion.GetSize(); 
  typename std::ofstream outputStream(filename.c_str()); 
  niftkitkDebugMacro(<< "size=" << size);
  
  // Image size. 
  for (unsigned i = 0; i < NDimensions; i++)
  {
    int temp = size[i]; 
    outputStream.write(reinterpret_cast<char*>(&temp), sizeof(temp)); 
  }
  // Jacobian values. 
  for (fieldIterator.GoToBegin(); !fieldIterator.IsAtEnd(); ++fieldIterator)
  {
    float deformation[NDimensions];
    for (unsigned i = 0; i < NDimensions; i++)
      deformation[i] = -fieldIterator.Get()[i]; 
    outputStream.write(reinterpret_cast<char*>(&deformation), sizeof(deformation[0])*NDimensions); 
  }
  // Image origin. 
  for (unsigned i = 0; i < NDimensions; i++)
  {
    int temp = origin[i]; 
    outputStream.write(reinterpret_cast<char*>(&temp), sizeof(temp)); 
  }
  
  outputStream.close(); 
  niftkitkDebugMacro(<< "WriteMidasVecImage():Finished, filename=" << filename);
}




template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::WriteVectorImage(std::string filename)
{
  niftkitkDebugMacro(<< "WriteVectorImage():Started, filename=" << filename);
  
  typedef float OutputVectorDataType;
  typedef Vector<OutputVectorDataType, NDimensions> OutputVectorPixelType;
  typedef Image<OutputVectorPixelType, NDimensions> OutputVectorImageType;
  typedef CastImageFilter<VectorFieldImageType, OutputVectorImageType> CastFilterType;
  typedef ImageFileWriter<OutputVectorImageType> WriterType;
  
  typename CastFilterType::Pointer caster = CastFilterType::New();
  typename WriterType::Pointer writer = WriterType::New();

  caster->SetInput(this->m_DeformationField);
  writer->SetFileName(filename);
  writer->SetInput(caster->GetOutput());
  writer->Update();
  
  niftkitkDebugMacro(<< "WriteVectorImage():Finished, filename=" << filename);
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
TScalarType
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::ComputeMaxDeformation()
{
  typedef ImageRegionConstIterator<DeformationFieldType> IteratorType;
  typedef typename DeformationFieldType::PixelType PixelType;

  PixelType value;
  IteratorType iterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion());

  double maxDeformation = 0;
  double tmp = 0;
  
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    value = iterator.Get();
    tmp = 0;
    for (unsigned int i = 0; i < NDimensions; i++)
      {
        tmp += (value[i] * value[i]);
      }
    if (sqrt(tmp) > maxDeformation)
      {
        maxDeformation = sqrt(tmp);
      }
  }

  niftkitkDebugMacro(<< "ComputeMaxDeformation():Maximum Deformation is:" << maxDeformation);
  return maxDeformation;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
TScalarType
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::ComputeMinDeformation()
{
  typedef ImageRegionConstIterator<DeformationFieldType> IteratorType;
  typedef typename DeformationFieldType::PixelType PixelType;

  PixelType value;
  IteratorType iterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion());

  double minDeformation = std::numeric_limits<TScalarType>::max();
  double tmp = 0;
  
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    value = iterator.Get();
    tmp = 0;
    
    for (unsigned int i = 0; i < NDimensions; i++)
      {
        tmp += (value[i] * value[i]);
      }
    if (sqrt(tmp) < minDeformation)
      {
        minDeformation = sqrt(tmp);
      }
  }

  niftkitkDebugMacro(<< "ComputeMinDeformation():Minimum Deformation is:" << minDeformation);
  return minDeformation;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
bool
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::GetInverse(Self* inverse) const
{
  typedef Point< TDeformationScalar, NDimensions > DeformationOutputPointType; 
  DeformationOutputPointType zeroPoint; 
  for (unsigned int dimensionIndex = 0; dimensionIndex < NDimensions; dimensionIndex++)
    zeroPoint[dimensionIndex] = 0.0;  
  double inversePhysicalSearchDistance = 0.0; 
  typename DeformationFieldType::SpacingType spacing; 
  typename DeformationFieldType::RegionType sourceRegion = inverse->m_DeformationField->GetLargestPossibleRegion(); 
  niftkitkDebugMacro(<< "sourceRegion:" << sourceRegion);
        
  spacing = inverse->m_DeformationField->GetSpacing(); 
  for (unsigned int i = 0; i < NDimensions; i++)
    inversePhysicalSearchDistance += this->m_InverseSearchRadius*spacing[i]; 
  inversePhysicalSearchDistance = sqrt(inversePhysicalSearchDistance); 
  
  // Store the minimum distance. 
  typename Image< float, NDimensions >::Pointer minDistanceMap = Image< float, NDimensions >::New(); 
  // Store the weightings in the Scattered data interpolation. 
  typename Image< float, NDimensions >::Pointer weightingMap = Image< float, NDimensions >::New(); 
  
  minDistanceMap->SetRegions(sourceRegion); 
  minDistanceMap->Allocate(); 
  minDistanceMap->FillBuffer(inversePhysicalSearchDistance+1.0); 
  weightingMap->SetRegions(sourceRegion); 
  weightingMap->Allocate(); 
  weightingMap->FillBuffer(0.0); 
  
  unsigned int numberOfNeighbourhoodVoxels = static_cast<unsigned int>(pow((2.0*m_InverseSearchRadius)+1, (int) NDimensions)); 
  niftkitkDebugMacro(<< "numberOfNeighbourhoodVoxels:" << numberOfNeighbourhoodVoxels);
  ImageRegionConstIteratorWithIndex<DeformationFieldType> forwardTransformationIterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion());  
  
  for (forwardTransformationIterator.GoToBegin(); !forwardTransformationIterator.IsAtEnd(); ++forwardTransformationIterator)
  {
    // Physhical position at the target image. 
    DeformationOutputPointType targetPhysicalPoint; 
    this->m_DeformationField->TransformIndexToPhysicalPoint(forwardTransformationIterator.GetIndex(), targetPhysicalPoint); 
    // Physical position at the source image and forward interpolation. 
    DeformationOutputPointType sourcePhysicalPoint; 
    DeformationOutputPointType forwardDeformation; 
    ContinuousIndex< TDeformationScalar, NDimensions > imageDeformation(forwardTransformationIterator.Get().GetDataPointer());
    this->m_DeformationField->TransformContinuousIndexToPhysicalPoint(imageDeformation, forwardDeformation);
    for (unsigned int i = 0; i < NDimensions; i++)
      sourcePhysicalPoint[i] = targetPhysicalPoint[i] + forwardDeformation[i]; 
    
    // Image position at the source image. 
    ContinuousIndex<TDeformationScalar, NDimensions> sourceIndex;
    inverse->m_DeformationField->TransformPhysicalPointToContinuousIndex(sourcePhysicalPoint, sourceIndex); 
    
    typedef NeighborhoodIterator< DeformationFieldType > NeighborhoodIteratorType;
    typename NeighborhoodIteratorType::RadiusType radius;
    radius.Fill((int)this->m_InverseSearchRadius); 
    typename NeighborhoodIteratorType::IndexType roundedSourceIndex; 
    typename NeighborhoodIteratorType::IndexType startIndex; 
    for (unsigned int i = 0; i < NDimensions; i++)
    {
      roundedSourceIndex[i] = (long int)niftk::Round(sourceIndex[i]); 
      startIndex[i] = (long int)(roundedSourceIndex[i]-this->m_InverseSearchRadius); 
    }
    NeighborhoodIteratorType inverseTransformIterator(radius, inverse->m_DeformationField, sourceRegion); 
    inverseTransformIterator.SetLocation(startIndex); 
    
    // Distribute the weight to the surrounding voxels in the source image. 
    for (unsigned int i = 0; i < numberOfNeighbourhoodVoxels; i++)
    {
      typename NeighborhoodIteratorType::IndexType currentSourceIndex = inverseTransformIterator.GetIndex(i); 
      
      if (!sourceRegion.IsInside(currentSourceIndex))
      {
        continue; 
      }
      DeformationOutputPointType currentSourcePhysicalPoint; 
      inverse->m_DeformationField->TransformIndexToPhysicalPoint(currentSourceIndex, currentSourcePhysicalPoint); 
      double distance = 0.0; 
      
      for (unsigned int dimensionIndex = 0; dimensionIndex < NDimensions; dimensionIndex++)
      {
        double tempDistance = currentSourcePhysicalPoint[dimensionIndex]-sourcePhysicalPoint[dimensionIndex]; 
        distance += tempDistance*tempDistance; 
      }
      distance = sqrt(distance); 
      double minDistance = minDistanceMap->GetPixel(currentSourceIndex); 
      // 1. Just set the target as the the inverse if it hits the source voxel, or...
      if (distance < this->m_InverseVoxelTolerance || minDistance < this->m_InverseVoxelTolerance)
      {
        if (distance < minDistance)
        {
          minDistanceMap->SetPixel(currentSourceIndex, distance); 
          inverse->m_DeformationField->SetPixel(currentSourceIndex, (zeroPoint-forwardDeformation).GetDataPointer()); 
        }
      }
      // 2. Update the weighting. 
      else
      {
        double weighting = (1.0/distance-1.0/inversePhysicalSearchDistance); 
        weighting = weighting*weighting; 
              
        weightingMap->SetPixel(currentSourceIndex, weighting+weightingMap->GetPixel(currentSourceIndex)); 
        DeformationOutputPointType weightedPoint; 
        for (unsigned int dimensionIndex = 0; dimensionIndex < NDimensions; dimensionIndex++)
        {
          weightedPoint[dimensionIndex] = -forwardDeformation[dimensionIndex]*weighting + inverse->m_DeformationField->GetPixel(currentSourceIndex)[dimensionIndex]; 
        }
        inverse->m_DeformationField->SetPixel(currentSourceIndex, weightedPoint.GetDataPointer()); 
      }
    }
  }
  
  ImageRegionIteratorWithIndex<DeformationFieldType> inverseTransformationIterator(inverse->m_DeformationField, sourceRegion); 
  typedef VectorLinearInterpolateImageFunction< DeformationFieldType, TDeformationScalar > DeformationFieldInterpolatorType; 
  typename DeformationFieldInterpolatorType::Pointer deformationFieldInterpolator = DeformationFieldInterpolatorType::New(); 
  
  deformationFieldInterpolator->SetInputImage(this->m_DeformationField); 
  
  for (inverseTransformationIterator.GoToBegin(); !inverseTransformationIterator.IsAtEnd(); ++inverseTransformationIterator)
  {
    typename ImageRegionIteratorWithIndex<DeformationFieldType>::IndexType inverseIndex = inverseTransformationIterator.GetIndex(); 
    DeformationOutputPointType sourcePhysicalPoint; 
    inverse->m_DeformationField->TransformIndexToPhysicalPoint(inverseIndex, sourcePhysicalPoint); 
    DeformationOutputPointType backwordDeformation = inverseTransformationIterator.Get().GetDataPointer(); 
    
    if (minDistanceMap->GetPixel(inverseIndex) > this->m_InverseVoxelTolerance)
    {
      // Divide by the weighting. 
      for (unsigned i = 0; i < NDimensions; i++)
      {
        if (weightingMap->GetPixel(inverseIndex) > 0.00001)
          backwordDeformation[i] = backwordDeformation[i]/weightingMap->GetPixel(inverseIndex); 
        else
          backwordDeformation[i] = 0.0; 
      }
      
      // Iteratively improve the inverse. 
      double minResidual = std::numeric_limits<double>::max(); 
      unsigned int numberOfIterations = 0; 
      while (numberOfIterations < m_MaxNumberOfInverseIterations && minResidual > this->m_InverseIterationTolerance)
      {
        DeformationOutputPointType newTargetPhysicalPoint; 
        for (unsigned i = 0; i < NDimensions; i++)
          newTargetPhysicalPoint[i] = sourcePhysicalPoint[i] + backwordDeformation[i]; 
        
        // Trilinear interpolation. 
        DeformationOutputPointType newSourcePhysicalPoint; 
        DeformationOutputPointType trilinearForwardDeformation; 
        
        ContinuousIndex< TDeformationScalar, NDimensions > newTargetImagePoint;
        this->m_DeformationField->TransformPhysicalPointToContinuousIndex(newTargetPhysicalPoint, newTargetImagePoint); 
        ContinuousIndex< double, NDimensions > trilinearDoubleTypeImageDeformation;
        if (this->m_DeformationField->GetLargestPossibleRegion().IsInside(newTargetImagePoint))
        {
          trilinearDoubleTypeImageDeformation = deformationFieldInterpolator->Evaluate(newTargetPhysicalPoint).GetDataPointer();
        }
        else
        {
          for (unsigned i = 0; i < NDimensions; i++)
            trilinearDoubleTypeImageDeformation[i] = 0; 
        }
        
        ContinuousIndex< TDeformationScalar, NDimensions > trilinearImageDeformation;
        for (unsigned int i = 0; i < NDimensions; i++)
          trilinearImageDeformation[i] = static_cast<TDeformationScalar>(trilinearDoubleTypeImageDeformation[i]); 
        this->m_DeformationField->TransformContinuousIndexToPhysicalPoint(trilinearImageDeformation, trilinearForwardDeformation);
        for (unsigned int i = 0; i < NDimensions; i++)
          newSourcePhysicalPoint[i] = newTargetPhysicalPoint[i] + trilinearForwardDeformation[i]; 
        
        DeformationOutputPointType residual = (newSourcePhysicalPoint - sourcePhysicalPoint).GetDataPointer(); 
        for (unsigned i = 0; i < NDimensions; i++)
          backwordDeformation[i] = backwordDeformation[i] - residual[i]/2.0; 
        numberOfIterations++; 
        minResidual = residual[0]; 
        for (unsigned int i = 1; i < NDimensions; i++)
        {
          if (residual[i] < minResidual)
            minResidual = residual[i]; 
        }
      }
    }
    
    // Convert back to deformation field. 
    ContinuousIndex<TDeformationScalar, NDimensions> inverseImageDeformation;
    
    inverse->m_DeformationField->TransformPhysicalPointToContinuousIndex(backwordDeformation, inverseImageDeformation); 
    inverseTransformationIterator.Set(inverseImageDeformation.GetDataPointer()); 
  }

  return true; 
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::ConcatenateAfterGivenTransform(Self* givenTransform)
{
  niftkitkDebugMacro(<< "ConcatenateAfterGivenTransform start...");
  niftkitkDebugMacro(<< "givenTransform region=" << givenTransform->m_DeformationField->GetLargestPossibleRegion());
  niftkitkDebugMacro(<< "this region=" << this->m_DeformationField->GetLargestPossibleRegion());
  if (givenTransform->m_DeformationField->GetLargestPossibleRegion().GetSize() != this->m_DeformationField->GetLargestPossibleRegion().GetSize())
    itkExceptionMacro("Given transform is not the same size as this transform."); 
  typedef Point< TDeformationScalar, NDimensions > DeformationOutputPointType; 
  
  // Make a copy for the interpolation. 
  typedef ImageDuplicator< DeformationFieldType > ImageDuplicatorType; 
  niftkitkDebugMacro(<< "Creating ImageDuplicatorType...");
  typename ImageDuplicatorType::Pointer deformationCopier = ImageDuplicatorType::New(); 
  deformationCopier->SetInputImage(this->m_DeformationField); 
  deformationCopier->Update(); 
  niftkitkDebugMacro(<< "Creating ImageDuplicatorType...done");
  
  typedef VectorLinearInterpolateImageFunction< DeformationFieldType, TDeformationScalar > DeformationFieldInterpolatorType; 
  typename DeformationFieldInterpolatorType::Pointer deformationFieldInterpolator = DeformationFieldInterpolatorType::New(); 
  deformationFieldInterpolator->SetInputImage(deformationCopier->GetOutput());
  
  ImageRegionConstIterator<DeformationFieldType> forwardTransformationIterator(givenTransform->m_DeformationField, givenTransform->m_DeformationField->GetLargestPossibleRegion());  
  ImageRegionIterator<DeformationFieldType> thisTransformationIterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion());  
  
  for (forwardTransformationIterator.GoToBegin(), thisTransformationIterator.GoToBegin(); 
       !forwardTransformationIterator.IsAtEnd(); 
       ++forwardTransformationIterator, ++thisTransformationIterator)
  {
    // Physical position after applying the given transform. 
    DeformationOutputPointType physicalPoint; 
    DeformationOutputPointType forwardDeformation; 
    ContinuousIndex< TDeformationScalar, NDimensions > imageDeformation(forwardTransformationIterator.Get().GetDataPointer());
    givenTransform->m_DeformationField->TransformContinuousIndexToPhysicalPoint(imageDeformation, forwardDeformation);
    givenTransform->m_DeformationField->TransformIndexToPhysicalPoint(forwardTransformationIterator.GetIndex(), physicalPoint); 
        for (unsigned int i = 0; i < NDimensions; i++)
      physicalPoint[i] = physicalPoint[i] + forwardDeformation[i]; 
    
    // Physical deformation at the calculate physical position. 
    //niftkitkDebugMacro(<< "Interpolating defromation..." << physicalPoint << "," << deformationFieldInterpolator);
    ContinuousIndex< TDeformationScalar, NDimensions > deformationIndex;
    this->m_DeformationField->TransformPhysicalPointToContinuousIndex(physicalPoint, deformationIndex); 
    ContinuousIndex< double, NDimensions > trilinearDoubleTypeImageDeformation;
    if (this->m_DeformationField->GetLargestPossibleRegion().IsInside(deformationIndex))
    {
      //niftkitkDebugMacro(<< "Inside...");
      trilinearDoubleTypeImageDeformation = deformationFieldInterpolator->EvaluateAtContinuousIndex(deformationIndex).GetDataPointer();
    }
    else
    {
      //niftkitkDebugMacro(<< "Outside...");
      for (unsigned int i = 0; i < NDimensions; i++)
        trilinearDoubleTypeImageDeformation[i] = 0.0; 
    }
    
    //niftkitkDebugMacro(<< "Interpolating defromation...done");
    ContinuousIndex< TDeformationScalar, NDimensions > trilinearImageDeformation;
    DeformationOutputPointType trilinearForwardDeformation; 
    for (unsigned int i = 0; i < NDimensions; i++)
      trilinearImageDeformation[i] = static_cast<TDeformationScalar>(trilinearDoubleTypeImageDeformation[i]); 
    this->m_DeformationField->TransformContinuousIndexToPhysicalPoint(trilinearImageDeformation, trilinearForwardDeformation);
    
    // Add up the deformation and save it. 
    ContinuousIndex<TDeformationScalar, NDimensions> totalImageDeformation;
    DeformationOutputPointType totalPhysicalDeformation; 
    for (unsigned int i = 0; i < NDimensions; i++)
      totalPhysicalDeformation[i] = forwardDeformation[i] + trilinearForwardDeformation[i]; 
    this->m_DeformationField->TransformPhysicalPointToContinuousIndex(totalPhysicalDeformation, totalImageDeformation); 
    thisTransformationIterator.Set(totalImageDeformation.GetDataPointer()); 
  }  
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
const typename DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>::ParametersType& 
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::GetFixedParameters(void) const 
{
  int index = 0; 
  
  this->m_FixedParameters.SetSize(NDimensions*4+NDimensions*NDimensions); 
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    this->m_FixedParameters.SetElement(index, this->m_DeformationField->GetSpacing()[i]);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    for (unsigned int j = 0; j < NDimensions; j++)
    {
      this->m_FixedParameters.SetElement(index, this->m_DeformationField->GetDirection()[i][j]);
      index++; 
    }
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    this->m_FixedParameters.SetElement(index, this->m_DeformationField->GetOrigin()[i]);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    this->m_FixedParameters.SetElement(index, this->m_DeformationField->GetLargestPossibleRegion().GetSize()[i]);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    this->m_FixedParameters.SetElement(index, this->m_DeformationField->GetLargestPossibleRegion().GetIndex()[i]);
    index++; 
  }
  return this->m_FixedParameters; 
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void 
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::SetFixedParameters(const ParametersType& parameters)
{
  typename FixedImageType::Pointer tempFixedImage = FixedImageType::New(); 
  DeformationFieldSpacingType spacing;
  DeformationFieldDirectionType direction;
  DeformationFieldOriginType origin;
  DeformationFieldSizeType regionSize;
  DeformationFieldIndexType regionIndex;
  DeformationFieldRegionType region;
  int index = 0; 
  
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    spacing[i] = parameters.GetElement(index);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    for (unsigned int j = 0; j < NDimensions; j++)
    {
      direction[i][j] = parameters.GetElement(index);
      index++; 
    }
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    origin[i] = parameters.GetElement(index);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    regionSize[i] = (unsigned long int)parameters.GetElement(index);
    index++; 
  }
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    regionIndex[i] = (long int)parameters.GetElement(index);
    index++; 
  }
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  
  tempFixedImage->SetSpacing(spacing);
  tempFixedImage->SetDirection(direction);
  tempFixedImage->SetOrigin(origin);
  tempFixedImage->SetRegions(region); 
  
  this->Initialize(tempFixedImage.GetPointer()); 
}


template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void 
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::ExtractComponents()
{
  // Copy image dimensions.
  DeformationFieldSpacingType spacing = this->m_DeformationField->GetSpacing();
  DeformationFieldDirectionType direction = this->m_DeformationField->GetDirection();
  DeformationFieldOriginType origin = this->m_DeformationField->GetOrigin();
  DeformationFieldSizeType size = this->m_DeformationField->GetLargestPossibleRegion().GetSize();
  DeformationFieldIndexType index = this->m_DeformationField->GetLargestPossibleRegion().GetIndex();
  DeformationFieldRegionType region;
  region.SetSize(size);
  region.SetIndex(index);
  
  ImageRegionConstIteratorWithIndex<DeformationFieldType> vectorIterator(this->m_DeformationField, this->m_DeformationField->GetLargestPossibleRegion()); 
  
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    this->m_DeformationFieldComponent[i] = DeformationFieldComponentImageType::New(); 
    this->m_DeformationFieldComponent[i]->SetRegions(region);
    this->m_DeformationFieldComponent[i]->SetOrigin(origin);
    this->m_DeformationFieldComponent[i]->SetDirection(direction);
    this->m_DeformationFieldComponent[i]->SetSpacing(spacing);
    this->m_DeformationFieldComponent[i]->Allocate();
    this->m_DeformationFieldComponent[i]->Update();
    
    ImageRegionIterator<DeformationFieldComponentImageType> iterator(this->m_DeformationFieldComponent[i], this->m_DeformationFieldComponent[i]->GetLargestPossibleRegion()); 
    
    for (iterator.GoToBegin(), vectorIterator.GoToBegin(); 
         !iterator.IsAtEnd(); 
         ++iterator, ++vectorIterator)
    {
      iterator.Set(vectorIterator.Get()[i]); 
    }
    
    
  }
}

#if 1

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void 
DeformableTransform<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::InvertUsingIterativeFixedPoint(typename Self::Pointer invertedTransform, int maxIterations, int maxOuterIterations, double tol)
{
  std::cout << "InvertUsingIterativeFixedPoint: start" << std::endl; 
  // const double tol = 0.001;
  // const int maxIterations = 30; 
  // const int maxOuterIterations = 5; 
  ImageRegionIteratorWithIndex<DeformationFieldType> vectorIterator(invertedTransform->m_DeformationField, invertedTransform->m_DeformationField->GetLargestPossibleRegion()); 
  
  typename DeformationFieldType::PointType origin = this->m_DeformationField->GetOrigin();
  DeformationFieldPixelType zero; 
  zero.Fill(0.); 
  invertedTransform->m_DeformationField->FillBuffer(zero); 
  this->ExtractComponents(); 
  // Setup the interpolation for the forward deformation field. Prefer the more accureate bspline interpolation. 
  //typedef BSplineInterpolateImageFunction<DeformationFieldComponentImageType, double> InterpolatorType; 
  typedef LinearInterpolateImageFunction<DeformationFieldComponentImageType, double> InterpolatorType; 
  typename InterpolatorType::Pointer interpolator[NDimensions]; 
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    interpolator[i] = InterpolatorType::New();
    interpolator[i]->SetInputImage(this->m_DeformationFieldComponent[i]); 
  }
  
  for (vectorIterator.GoToBegin(); 
       !vectorIterator.IsAtEnd(); 
       ++vectorIterator)
  {
    DeformationFieldIndexType index = vectorIterator.GetIndex();
    OutputPointType currentPoint;
    OutputPointType transformedPoint;
    
    invertedTransform->m_DeformationField->TransformIndexToPhysicalPoint(index, currentPoint);
    double diff = 1.; 
    
    // Loop until converge. 
    // Based on "A simple fixed-point apporach to invert a deformation field", Chen et al., Med. Phys. 2008, but modified to use 
    // a step size to control the magnitude of update in each step. (becoming like a gradient descent)
    DeformationFieldPixelType previousInvertedValue; 
    DeformationFieldPixelType invertedValue; 
    invertedValue.Fill(0.); 
    previousInvertedValue.Fill(0.);
    int count = 0; 
    int outerCount = 0; 
    double stepSize = 1.; 
    double previousDiff = 0.; 
    double finalDiff = 0.; 
    previousInvertedValue = vectorIterator.Get(); 
    while (diff > tol && outerCount < maxOuterIterations)
    {
      while (diff > tol && count < maxIterations && fabs(previousDiff-diff) > 1.e-12)
      {
        // Transform current point using the inverse. 
        transformedPoint = invertedTransform->TransformPoint(currentPoint); 
        for (unsigned i = 0; i < NDimensions; i++)
        {
          if (interpolator[i]->IsInsideBuffer(transformedPoint))
          {
            // Calculate the forward deformation at the transformed point, which should equal to the -ve of the inverse deformation. 
            // v_n = -u(x + v_(n-1)(x))
            invertedValue[i] = -interpolator[i]->Evaluate(transformedPoint);  
          }
          else
          {
            invertedValue[i] = 0.; 
          }
        }
        previousDiff = diff; 
        diff = (previousInvertedValue-invertedValue).GetNorm(); 
        double factor = std::min<double>(stepSize, stepSize/diff); 
        count++; 
        // Small modification to improve convergence by controlling the amount of update. When stepSize=1, this is the original fixed point iteration. 
        invertedValue = previousInvertedValue + (invertedValue-previousInvertedValue)*factor; 
        vectorIterator.Set(invertedValue); 
        previousInvertedValue = invertedValue; 
      }
      
      finalDiff = diff; 
      if (diff > tol)
      {
        stepSize *= 0.618033989; 
        previousDiff = 0.; 
        diff = 1.; 
      }
      outerCount++; 
      count = 0; 
    }
    if (diff > tol)
    {
      std::cout << "index=" << index << ",count=" << count << ",finalDiff=" << finalDiff << ",invertedValue=" << invertedValue << std::endl;  
    }
#if 0
    // Inverse by using iterative refinement. Iteratively adjust the inverse through a number of step size. 
    // Not using it. 
    if (diff > tol)
    {
      std::cout << "Switch to iterative refinement" << std::endl; 
      std::cout << "index=" << index << ",diff=" << diff << ",count=" << count << ",invertedValue=" << invertedValue << std::endl; 
      invertedValue.Fill(0.); 
      vectorIterator.Set(invertedValue); 
      DeformationFieldPixelType step; 
      step.Fill(0.1); 
      double bestDiff = 1.e9; 
      DeformationFieldPixelType bestInvertedValue; 
      count = 0; 
      while (step.GetNorm() > 0.0001 && count < maxIterations && diff > tol)
      {
        bool isImproved = false; 
        
        for (unsigned j = 0; j < NDimensions; j++)
        {
          invertedValue[j] += step[j]; 
          vectorIterator.Set(invertedValue); 
          // Apply backward transform to current point. 
          transformedPoint = invertedTransform->TransformPoint(currentPoint); 
          // Find out the forward defromation using interpolation. 
          DeformationFieldPixelType forwardValue; 
          for (unsigned i = 0; i < NDimensions; i++)
          {
            if (interpolator[i]->IsInsideBuffer(transformedPoint))
            {
              forwardValue[i] = interpolator[i]->Evaluate(transformedPoint);  
            }
          }
          // Apply forward transform to the backward-transformed point. 
          OutputPointType physicalDeformation; 
          OutputPointType currentPointAfterForwardBackwardTransform;
          ContinuousIndex< double, NDimensions > forwardImageDeformation;
          for (unsigned i = 0; i < NDimensions; i++)
          {
            forwardImageDeformation[i] = forwardValue[i]; 
          }
          this->m_DeformationField->TransformContinuousIndexToPhysicalPoint(forwardImageDeformation, physicalDeformation); 
          for (unsigned i = 0; i < NDimensions; i++)
          {
            currentPointAfterForwardBackwardTransform[i] = transformedPoint[i] - (physicalDeformation[i]-origin[i]);
          }
          // And hope to get back the current point. 
          diff = (currentPoint-currentPointAfterForwardBackwardTransform).GetNorm(); 
          if (diff < bestDiff)
          {
            bestDiff = diff; 
            bestInvertedValue = invertedValue; 
            isImproved = true; 
          }
          
          invertedValue[j] -= 2.*step[j]; 
          vectorIterator.Set(invertedValue); 
          // Apply backward transform to current point. 
          transformedPoint = invertedTransform->TransformPoint(currentPoint); 
          // Find out the forward defromation using interpolation. 
          for (unsigned i = 0; i < NDimensions; i++)
          {
            if (interpolator[i]->IsInsideBuffer(transformedPoint))
            {
              forwardValue[i] = interpolator[i]->Evaluate(transformedPoint);  
            }
          }
          // Apply forward transform to the backward-transformed point. 
          for (unsigned i = 0; i < NDimensions; i++)
          {
            forwardImageDeformation[i] = forwardValue[i]; 
          }
          this->m_DeformationField->TransformContinuousIndexToPhysicalPoint(forwardImageDeformation, physicalDeformation); 
          for (unsigned i = 0; i < NDimensions; i++)
          {
            currentPointAfterForwardBackwardTransform[i] = transformedPoint[i] - (physicalDeformation[i]-origin[i]);
          }
          // And hope to get back the current point. 
          diff = (currentPoint-currentPointAfterForwardBackwardTransform).GetNorm(); 
          if (diff < bestDiff)
          {
            bestDiff = diff; 
            bestInvertedValue = invertedValue; 
            isImproved = true; 
          }
          // Debug. 
          if (index[0] == 128 && index[1] == 40)
          {
            std::cout << "index=" << index << ",count=" << count << ",bestDiff=" << bestDiff << ",bestInvertedValue=" << bestInvertedValue << ",step=" << step << ",forwardValue=" << forwardValue << std::endl;  
          }
          invertedValue = bestInvertedValue; 
        }
        
        if (!isImproved)
        {
          step *= 0.5; 
        }
        else
        {
          step.Fill(0.1); 
        }
        count++; 
        
      }
      vectorIterator.Set(bestInvertedValue); 
      if (bestDiff > 0.1)
      {
        std::cout << "index=" << index << ",bestDiff=" << bestDiff << ",count=" << count << ",bestInvertedValue=" << bestInvertedValue << std::endl << std::endl;  
      }
    }
#endif                       
  }
  std::cout << "InvertUsingIterativeFixedPoint: end" << std::endl; 
}
#endif


} // namespace

#endif // __itkDeformableTransform_txx

