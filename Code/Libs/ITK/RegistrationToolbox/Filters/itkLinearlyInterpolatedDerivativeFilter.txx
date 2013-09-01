/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkLinearlyInterpolatedDerivativeFilter_txx
#define __itkLinearlyInterpolatedDerivativeFilter_txx

#include "itkLinearlyInterpolatedDerivativeFilter.h"
#include <itkObjectFactory.h>
#include <itkIdentityTransform.h>
#include <itkProgressReporter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkPoint.h>

#include <itkLogHelper.h>

namespace itk
{
/**
 * Initialize new instance
 */
template <typename TFixedImage, typename TMovingImage, class TScalar, class TDeformationScalar>
LinearlyInterpolatedDerivativeFilter<TFixedImage,TMovingImage,TScalar, TDeformationScalar>
::LinearlyInterpolatedDerivativeFilter()
{
  m_DefaultPixelValue.Fill(0);
  m_MovingImageLowerPixelValue = std::numeric_limits<MovingImagePixelType>::min();
  m_MovingImageUpperPixelValue = std::numeric_limits<MovingImagePixelType>::max();
  m_Transform = IdentityTransform<TScalar, Dimension>::New();
}

/**
 * Print out a description of self
 *
 * \todo Add details about this class
 */
template <typename TFixedImage, typename TMovingImage, class TScalar, class TDeformationScalar>
void 
LinearlyInterpolatedDerivativeFilter<TFixedImage,TMovingImage,TScalar, TDeformationScalar>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "DefaultPixelValue: " << m_DefaultPixelValue << std::endl;
  os << indent << "Transform: " << m_Transform.GetPointer() << std::endl;
  return;
}

template< class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar > 
void 
LinearlyInterpolatedDerivativeFilter< TFixedImage, TMovingImage, TScalarType, TDeformationScalar >
::SetFixedImage(const FixedImageType *image)
{
  this->ProcessObject::SetNthInput(0, const_cast< FixedImageType* >(image));
  this->Modified();
  
  niftkitkDebugMacro(<<"SetFixedImage():Set input[0] to address:" << image);
}

template< class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar> 
void 
LinearlyInterpolatedDerivativeFilter< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::SetMovingImage(const MovingImageType *image)
{
  this->ProcessObject::SetNthInput(1, const_cast< MovingImageType* >(image));
  this->Modified();
  
  niftkitkDebugMacro(<<"SetMovingImage():Set input[1] to address:" << image);
}

/**
 * Put any single threaded initialization here.
 */
template <typename TFixedImage, typename TMovingImage, class TScalar, class TDeformationScalar>
void 
LinearlyInterpolatedDerivativeFilter<TFixedImage,TMovingImage,TScalar, TDeformationScalar>
::BeforeThreadedGenerateData()
{
  if( !m_Transform )
    {
      itkExceptionMacro(<< "Transform not set");
    }
}

/**
 * Put any single threaded clean-up here.
 */
template <typename TFixedImage, typename TMovingImage, class TScalar, class TDeformationScalar>
void 
LinearlyInterpolatedDerivativeFilter<TFixedImage,TMovingImage,TScalar, TDeformationScalar>
::AfterThreadedGenerateData()
{
}

/** 
 * Inform pipeline of necessary input image region
 *
 * Determining the actual input region is non-trivial, especially
 * when we cannot assume anything about the transform being used.
 * So we do the easy thing and request the entire input image.
 */
template <typename TFixedImage, typename TMovingImage, class TScalar, class TDeformationScalar>
void 
LinearlyInterpolatedDerivativeFilter<TFixedImage,TMovingImage,TScalar, TDeformationScalar>
::GenerateInputRequestedRegion()
{
  // call the superclass's implementation of this method
  Superclass::GenerateInputRequestedRegion();

  if ( !this->GetInput(0))
    {
      itkExceptionMacro(<< "Fixed image is not set");
    }

  if ( !this->GetInput(1))
    {
      itkExceptionMacro(<< "Moving image is not set");
    }

  // Request the entire input image
  FixedImagePointer  fixedPtr  = 
    const_cast< TFixedImage *>( this->GetInput(0) );
  FixedImageRegionType fixedRegion;
  fixedRegion = fixedPtr->GetLargestPossibleRegion();
  fixedPtr->SetRequestedRegion(fixedRegion);

  // Request the entire input image
  MovingImagePointer  movingPtr  = 
    const_cast< TMovingImage *>( this->GetInput(1) );
  MovingImageRegionType movingRegion;
  movingRegion = movingPtr->GetLargestPossibleRegion();
  movingPtr->SetRequestedRegion(movingRegion);

  return;
}

/** 
 * Inform pipeline of required output region
 */
template <typename TFixedImage, typename TMovingImage, class TScalar, class TDeformationScalar>
void 
LinearlyInterpolatedDerivativeFilter<TFixedImage,TMovingImage,TScalar, TDeformationScalar>
::GenerateOutputInformation()
{
  // call the superclass' implementation of this method
  Superclass::GenerateOutputInformation();

  // get pointers to the input and output
  OutputImagePointer outputPtr = this->GetOutput();
  if ( !outputPtr )
    {
      itkExceptionMacro(<< "Output image is not set");
    }

  FixedImagePointer  fixedPtr  = 
    const_cast< TFixedImage *>( this->GetInput(0) );

  FixedImageRegionType    fixedImageRegion    = fixedPtr->GetLargestPossibleRegion();
  FixedImageIndexType     fixedImageIndex     = fixedImageRegion.GetIndex();
  FixedImageSizeType      fixedImageSize      = fixedImageRegion.GetSize();
  FixedImageSpacingType   fixedImageSpacing   = fixedPtr->GetSpacing();
  FixedImageOriginType    fixedImageOrigin    = fixedPtr->GetOrigin();
  FixedImageDirectionType fixedImageDirection = fixedPtr->GetDirection();
  
  OutputImageRegionType outputRegion;
  OutputImageIndexType  outputIndex = fixedImageIndex;
  OutputImageSizeType   outputSize  = fixedImageSize;
  outputRegion.SetSize(outputSize);
  outputRegion.SetIndex(outputIndex);
  OutputImageSpacingType outputSpacing = fixedImageSpacing;
  OutputImageOriginType  outputOrigin  = fixedImageOrigin;
  OutputImageDirectionType outputDirection = fixedImageDirection;
  
  outputPtr->SetRegions( outputRegion );
  outputPtr->SetSpacing( outputSpacing );
  outputPtr->SetOrigin( outputOrigin );
  outputPtr->SetDirection( outputDirection);

  return;
}

template <typename TFixedImage, typename TMovingImage, class TScalar, class TDeformationScalar>
void 
LinearlyInterpolatedDerivativeFilter<TFixedImage,TMovingImage,TScalar, TDeformationScalar>
::ThreadedGenerateData(
  const OutputImageRegionType& outputRegionForThread,
  ThreadIdType threadId)
{
  // Get the input pointers
  MovingImagePointer inputPtr = const_cast< TMovingImage *>( this->GetInput(1) );
  MovingImageSizeType inputSize = inputPtr->GetLargestPossibleRegion().GetSize();
  
  // Get the output pointers
  OutputImagePointer outputPtr = this->GetOutput();
  OutputPixelType outputPixel;
  
  // Create an iterator that will walk the output region for this thread.
  typedef ImageRegionIteratorWithIndex<OutputImageType> OutputIterator;
  OutputIterator outIt(outputPtr, outputRegionForThread);

  // Define a few indices that will be used to translate
  // from an output pixel to an input pixel
  typedef Point<TScalar, Dimension> PointType;
  
  PointType inputPoint;  
  PointType outputPoint;

  typedef ContinuousIndex<TScalar, Dimension> ContinuousIndexType;
  ContinuousIndexType inputIndex;

  // Support for progress methods/callbacks
  ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());
        
  // Walk the output region
  outIt.GoToBegin();

  // This fix works for images up to approximately 2^25 pixels in
  // any dimension.  If the image is larger than this, this constant
  // needs to be made lower.
  double precisionConstant = 1<<(NumericTraits<double>::digits>>1);
  
  MovingImageIndexType index;
  
  short voxel[Dimension];
  TScalar relative[Dimension];
  TScalar xBasis[2];
  TScalar yBasis[2];
  TScalar zBasis[2];
  
  TScalar deriv[2];
  deriv[0] = -1;
  deriv[1] = 1;
  
  while ( !outIt.IsAtEnd() )
    {
      // Determine the index of the current output pixel
      outputPtr->TransformIndexToPhysicalPoint( outIt.GetIndex(), outputPoint );

      // Compute corresponding input pixel position
      inputPoint = m_Transform->TransformPoint(outputPoint);
      inputPtr->TransformPhysicalPointToContinuousIndex(inputPoint, inputIndex);

      // The inputIndex is precise to many decimal points, but this precision
      // involves some error in the last bits.  
      // Sometimes, when an index should be inside of the image, the
      // index will be slightly
      // greater than the largest index in the image, like 255.00000000002
      // for a image of size 256.  This can cause an empty row to show up
      // at the bottom of the image.
      // Therefore, the following routine uses a
      // precisionConstant that specifies the number of relevant bits,
      // and the value is truncated to this precision.
      for (unsigned int i=0; i < Dimension; ++i)
        {
          double roundedInputIndex = vcl_floor(inputIndex[i]);
          double inputIndexFrac = inputIndex[i] - roundedInputIndex;
          double newInputIndexFrac = vcl_floor(precisionConstant * inputIndexFrac)/precisionConstant;
          inputIndex[i] = roundedInputIndex + newInputIndexFrac;
        }
    
      bool isInsideBuffer = true;
      for (unsigned int i = 0; i < Dimension; i++)
        {
          if (inputIndex[i] < 0 || inputIndex[i] >= inputSize[i])
            {
              isInsideBuffer = false;
              break;
            }
        }
      
      if( isInsideBuffer)
        {
          for (unsigned int i = 0; i < Dimension; i++)
            {
              voxel[i] = (short)vcl_floor(inputIndex[i]); 
              relative[i] = inputIndex[i] - voxel[i];
              if (relative[i] < 0.0) relative[i] = 0.0; // rounding error correction
            }
        
          /** Don't like this, but I'm doing a separate 2D and 3D version of Marc Modat's code. */
          if (Dimension == 3)
            {
            
              xBasis[0]=1.0-relative[0];
              xBasis[1]=relative[0];
              yBasis[0]=1.0-relative[1];
              yBasis[1]=relative[1];
              zBasis[0]=1.0-relative[2];
              zBasis[1]=relative[2];

              TScalar derivValueX=(TScalar)0.0;
              TScalar derivValueY=(TScalar)0.0;
              TScalar derivValueZ=(TScalar)0.0;
              bool zero=false;
            
              for(short c=0; c<2; c++){
              
                short Z = voxel[2] + c;
              
                if(-1 < Z && Z < (short)inputSize[2]){
                
                  TScalar yTempNewValue=(TScalar)0.0;
                  TScalar yTempDerivValueX=(TScalar)0.0;
                  TScalar yTempDerivValueY=(TScalar)0.0;
                  TScalar yTempDerivValueZ=(TScalar)0.0;

                  for(short b=0; b<2; b++){
                  
                    short Y = voxel[1] + b;
                                      
                    if(-1 < Y && Y < (short)inputSize[1]){
                    
                      TScalar xTempNewValue=(TScalar)0.0;
                      TScalar xTempDerivValueX=(TScalar)0.0;
                    
                      for(short a=0; a<2; a++){
                      
                        short X = voxel[0] + a;
                      
                        if(-1 < X && X < (short)inputSize[0]){
                        
                          index[0] = X;
                          index[1] = Y;
                          index[2] = Z;
                        
                          TScalar intensity = (TScalar)inputPtr->GetPixel(index);

                          if(intensity <= m_MovingImageLowerPixelValue || intensity > m_MovingImageUpperPixelValue) 
                            {
                              zero=true;
                            }

                          xTempNewValue +=  (TScalar)(intensity * xBasis[a]);
                          xTempDerivValueX +=  (TScalar)(intensity * deriv[a]);
                        
                        }
                        else zero=true;
                      }
                      yTempNewValue += (TScalar)(xTempNewValue * yBasis[b]);
                      yTempDerivValueX += (TScalar)(xTempDerivValueX * yBasis[b]);
                      yTempDerivValueY += (TScalar)(xTempNewValue * deriv[b]);
                      yTempDerivValueZ += (TScalar)(xTempNewValue * yBasis[b]);
                    }
                    else zero=true;
                  }
                  derivValueX += (TScalar)(yTempDerivValueX * zBasis[c]);
                  derivValueY += (TScalar)(yTempDerivValueY * zBasis[c]);
                  derivValueZ += (TScalar)(yTempDerivValueZ * deriv[c]);
                }
                else zero=true;
              }
              if (zero == true)
                {
                  outputPixel = m_DefaultPixelValue;
                }
              else
                {
                  outputPixel[0] = derivValueX;
                  outputPixel[1] = derivValueY;
                  outputPixel[2] = derivValueZ;
                }
              
            }
          else if (Dimension == 2)
            {

              xBasis[0]=1.0-relative[0];
              xBasis[1]=relative[0];
              yBasis[0]=1.0-relative[1];
              yBasis[1]=relative[1];

              bool zero=false;
              TScalar derivValueX=(TScalar)0.0;
              TScalar derivValueY=(TScalar)0.0;

              for(short b=0; b<2; b++){
                  
                short Y = voxel[1] + b;
                  
                if(-1 < Y && Y < (short)inputSize[1]){
                    
                  TScalar xTempNewValue=(TScalar)0.0;
                  TScalar xTempDerivValueX=(TScalar)0.0;
                    
                  for(short a=0; a<2; a++){
                      
                    short X = voxel[0] + a;
                      
                    if(-1 < X && X < (short)inputSize[0]){
                        
                      index[0] = X;
                      index[1] = Y;
                      TScalar intensity = (TScalar)inputPtr->GetPixel(index);
                        
                      if(intensity < m_MovingImageLowerPixelValue || intensity > m_MovingImageUpperPixelValue) 
                        {
                          zero=true;
                        }
                        
                      xTempNewValue +=  (TScalar)(intensity * xBasis[a]);
                      xTempDerivValueX +=  (TScalar)(intensity * deriv[a]);
                        
                    }
                    else zero=true;
                  }
                  derivValueX += (TScalar)(xTempDerivValueX * yBasis[b]);
                  derivValueY += (TScalar)(xTempNewValue * deriv[b]);
   
                }
                else zero=true;
              }
              
              if (zero == true)
                {
                  outputPixel = m_DefaultPixelValue;
                }
              else
                {
                  outputPixel[0] = derivValueX;
                  outputPixel[1] = derivValueY;
                }
            }
          else
            {
              itkExceptionMacro(<< "This filter is only suitable for 2D or 3D images");  
            }
          outIt.Set( outputPixel );
        }
      else
        {
          outIt.Set( m_DefaultPixelValue ); // default background value
        }

      progress.CompletedPixel();
      ++outIt;
    }
  return;
}

/** 
 * Verify if any of the components has been modified.
 */
template <typename TFixedImage, typename TMovingImage, class TScalar, class TDeformationScalar>
unsigned long 
LinearlyInterpolatedDerivativeFilter<TFixedImage,TMovingImage,TScalar, TDeformationScalar>
::GetMTime( void ) const
{
  unsigned long latestTime = Object::GetMTime(); 

  if( m_Transform )
    {
    if( latestTime < m_Transform->GetMTime() )
      {
      latestTime = m_Transform->GetMTime();
      }
    }

  return latestTime;
}

} // end namespace itk

#endif

