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
#ifndef __itkRegistrationForceFilter_txx
#define __itkRegistrationForceFilter_txx

#include "itkRegistrationForceFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkCastImageFilter.h"

#include "itkLogHelper.h"
#include "itkUCLMacro.h"

namespace itk {

template< class TFixedImage, class TMovingImage, class TScalarType >
RegistrationForceFilter< TFixedImage, TMovingImage, TScalarType  >
::RegistrationForceFilter()
{
  // At least 2 inputs are necessary for a vector image.
  this->SetNumberOfRequiredInputs( 2 ); 
  m_ScaleToSizeOfVoxelAxis = false;
  m_FixedLowerPixelValue = std::numeric_limits<InputImagePixelType>::min();
  m_FixedUpperPixelValue = std::numeric_limits<InputImagePixelType>::max();
  m_MovingLowerPixelValue = std::numeric_limits<InputImagePixelType>::min();
  m_MovingUpperPixelValue = std::numeric_limits<InputImagePixelType>::max();
  this->m_IsSymmetric = false; 
  
  this->m_FixedImageMask = NULL; 
  niftkitkDebugMacro(<<"RegistrationForceFilter():Constructed m_ScaleToSizeOfVoxelAxis=" << m_ScaleToSizeOfVoxelAxis \
      << ", m_FixedLowerPixelValue=" << m_FixedLowerPixelValue \
      << ", m_FixedUpperPixelValue=" << m_FixedUpperPixelValue \
      << ", m_MovingLowerPixelValue=" << m_MovingLowerPixelValue \
      << ", m_MovingUpperPixelValue=" << m_MovingUpperPixelValue \
      );
}

template< class TFixedImage, class TMovingImage, class TScalarType > 
void 
RegistrationForceFilter< TFixedImage, TMovingImage, TScalarType  >
::SetNthInput(unsigned int idx, const InputImageType *image)
{
  this->ProcessObject::SetNthInput(idx, const_cast< InputImageType* >(image));
  this->Modified();
  
  niftkitkDebugMacro(<<"SetNthInput():Set input[" << idx << "] to address:" << image);
}

template< class TFixedImage, class TMovingImage, class TScalarType > 
void 
RegistrationForceFilter< TFixedImage, TMovingImage, TScalarType  >
::BeforeThreadedGenerateData()
{

  // Check we have a histogram based similarity measure.
  if (!this->m_Metric.GetPointer())
    {
      niftkitkExceptionMacro(<< "RegistrationForceFilter must have a HistogramSimilarityMeasure.");
    }
    
  // Check to verify all inputs are specified and have the same metadata, spacing etc...
  
  const unsigned int numberOfInputs = this->GetNumberOfInputs();
  
  // We should have at least two inputs.
  if (numberOfInputs < 2)
    {
	  niftkitkExceptionMacro(<< "RegistrationForceFilter should only at least two inputs");
    }

  RegionType region;
  for (unsigned int i=0; i<numberOfInputs; i++)
    {
      // Check each input is set.
      InputImageType *input = static_cast< InputImageType * >(this->ProcessObject::GetInput(i));
      if (!input)
        {
    	  niftkitkExceptionMacro(<< "Input " << i << " not set!");
        }
        
      // Check they are the same size.
      if (i==0)
        {
          region = input->GetLargestPossibleRegion();
        }
      else if (input->GetLargestPossibleRegion() != region) 
        {
    	  niftkitkExceptionMacro(<< "All Inputs must have the same dimensions.");
        }
    }
}

template< class TFixedImage, class TMovingImage, class TScalarType >
void
RegistrationForceFilter< TFixedImage, TMovingImage, TScalarType  >
::AfterThreadedGenerateData()
{
 //  #ifdef(DEBUG)
     niftkitkDebugMacro(<<"AfterThreadedGenerateData():Checking force statistics");
 //  #endif

      float magnitude;
      OutputPixelType value;
      
      OutputPixelType min;
      OutputPixelType max;
      OutputPixelType minAbsolute;
      OutputPixelType maxAbsolute;
      OutputPixelType mean;
      OutputDataType minMagnitude;
      OutputDataType maxMagnitude;
      OutputDataType meanMagnitude;

      unsigned int i = 0;
      unsigned long int samples = 0;
      unsigned long int counter = 0;
      
      minMagnitude = std::numeric_limits<TScalarType>::max();
      maxMagnitude = std::numeric_limits<TScalarType>::min();
      min.Fill(std::numeric_limits<TScalarType>::max());
      minAbsolute.Fill(std::numeric_limits<TScalarType>::max());
      max.Fill(std::numeric_limits<TScalarType>::min());
      maxAbsolute.Fill(std::numeric_limits<TScalarType>::min());
      mean.Fill(0);
      meanMagnitude = 0;
      
      ImageRegionConstIteratorWithIndex<OutputImageType> iterator(this->GetOutput(), this->GetOutput()->GetLargestPossibleRegion());
      
      samples = 0;
      iterator.GoToBegin();
      while (!iterator.IsAtEnd())
        {
          value = iterator.Get();
          magnitude = 0;
          for (i = 0; i < Dimension; i++)
            {
              magnitude += (value[i]*value[i]);
            }
          magnitude = sqrt(magnitude);
          
          if (magnitude > 0)
            {
              if (magnitude < minMagnitude)
                {
                  minMagnitude = magnitude;
                }
              if (magnitude > maxMagnitude)
                {
                  maxMagnitude = magnitude;
                }
              for (i = 0; i < Dimension; i++)
                {
                  if (value[i] < min[i])
                    {
                      min[i] = value[i];
                    }
                  if (value[i] > max[i])
                    {
                      max[i] = value[i];
                    }
                  if (fabs(value[i]) < minAbsolute[i])
                    {
                      minAbsolute[i] = fabs(value[i]);   
                    }
                  if (fabs(value[i]) > maxAbsolute[i])
                    {
                      maxAbsolute[i] = fabs(value[i]);   
                    }
                  mean[i] += value[i];
                }
              meanMagnitude += magnitude;
              samples++;
/*              
              std::cout << "counter=" << counter << ", value=" << value << std::endl;
*/              
            }
          ++iterator;
          counter++;
        }
      for (i = 0; i < Dimension; i++)
        {
          mean[i] /= (TScalarType) samples;
        }
      meanMagnitude /= (TScalarType) samples;
      
      niftkitkDebugMacro(<<"AfterThreadedGenerateData():Checking force statistics, minMagnitude=" << minMagnitude \
        << ", maxMagnitude=" << maxMagnitude \
        << ", meanMagnitude=" << meanMagnitude \
        << ", min=" << min \
        << ", max=" << max \
        << ", mean=" << mean \
        << ", minAbsolute=" << minAbsolute \
        << ", maxAbsolute=" << maxAbsolute \
        << ", samples=" << samples);
        
}

template< class TFixedImage, class TMovingImage, class TScalarType > 
void
RegistrationForceFilter< TFixedImage, TMovingImage, TScalarType  >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  
  if (this->m_Metric.GetPointer() != 0)
    {
      os << indent << "Metric:" << std::endl << this->m_Metric << std::endl;
    }
}

template< class TFixedImage, class TMovingImage, class TScalarType > 
void
RegistrationForceFilter< TFixedImage, TMovingImage, TScalarType  >
::WriteForceImage(std::string filename)
{
  niftkitkDebugMacro(<<"WriteForceImage():Writing to:" << filename);
  
  typedef float OutputVectorDataType;
  typedef Vector<OutputVectorDataType, TFixedImage::ImageDimension> OutputVectorPixelType;
  typedef Image<OutputVectorPixelType, TFixedImage::ImageDimension> OutputVectorImageType;
  typedef CastImageFilter<OutputImageType, OutputVectorImageType> CastFilterType;
  typedef ImageFileWriter<OutputVectorImageType> WriterType;
  
  typename CastFilterType::Pointer caster = CastFilterType::New();
  typename WriterType::Pointer writer = WriterType::New();

  caster->SetInput(this->GetOutput());
  writer->SetFileName(filename);
  writer->SetInput(caster->GetOutput());
  writer->Update();
  
  niftkitkDebugMacro(<<"WriteForceImage():Writing to:" << filename << "....DONE");
}

} // end namespace itk

#endif
