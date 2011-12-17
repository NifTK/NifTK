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
#ifndef __itkExtendedBrainMaskWithSmoothDropOffCompositeFilter_h
#define __itkExtendedBrainMaskWithSmoothDropOffCompositeFilter_h


#include "itkImageToImageFilter.h"
#include "itkInjectSourceImageGreaterThanZeroIntoTargetImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryCrossStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"

namespace itk 
{

/**
 * \class ExtendedBrainMaskWithSmoothDropOffCompositeFilter
 * \brief Takes a mask, dilates outwards, blurs with a gaussian, then substitutes in
 * the dilated mask, so you end up with a bigger mask than you started, with gaussian 
 * blurred edges (ie. smooth drop off to zero).
 * 
 * This filter is designed for Nicky Hobbs to create a brain mask for her fluid registration 
 * experiments. As per email from Jo Barnes, you have 4 parameters,
 * <pre>
 * 
 * 1.) Threshold using value T
 * 2.) First stage of dilation = X voxels
 * 3.) Second stage of dilation = Y voxels
 * 4.) Gaussian FWHM = Z millimetres
 * 
 * </pre>
 * and the process is:
 * <pre>
 * 
 * 1.) Threshold using value T so that if value < T, output = 0 else value = 1.
 * 2.) Dilate mask by X voxels in all directions, let's call the result mask A (binary);
 * 3.) Dilate mask A further by Y voxels in all directions, let's call the result mask B (binary);
 * 4.) Applied to mask B gaussian smoothing with a Zmm FWHM kernel, to generate an approximation 
 * to a smooth signal drop-off, let's call the result mask C (reals);
 * 5.) Replaced the inner portion of mask C with mask A, i.e. to enforce the preservation of the 
 * original signal intensity values in a neighbourhood of 8 voxels around the original brain mask, 
 * followed by a smooth drop-off around that, let's call the result mask D (reals);
 * 6.) The output is mask D.
 * 
 * </pre>
 * 
 * This class can be used as an example of a Composite Filter.
 */
template< class TImageType>
class ITK_EXPORT ExtendedBrainMaskWithSmoothDropOffCompositeFilter : public ImageToImageFilter<TImageType, TImageType>
{
  
public:

  // Standard ITK style typedefs.
  typedef ExtendedBrainMaskWithSmoothDropOffCompositeFilter Self;
  typedef ImageToImageFilter<TImageType, TImageType>        Superclass;
  typedef SmartPointer<Self>                                Pointer;
  typedef SmartPointer<const Self>                          ConstPointer;

  /** Method for creation through object factory */
  itkNewMacro(Self);

  /** Run-time type information */
  itkTypeMacro(ExtendedBrainMaskWithSmoothDropOffCompositeFilter, ImageToImageFilter);

  // typedefs for internal filters.
  typedef TImageType                                                                         ImageType;
  typedef typename ImageType::Pointer                                                        ImagePointer;
  typedef BinaryThresholdImageFilter<TImageType, TImageType>                                 BinaryThresholdFilterType;
  typedef typename BinaryThresholdFilterType::Pointer                                        BinaryThresholdFilterPointer;
  typedef BinaryCrossStructuringElement<typename TImageType::PixelType, 
                                        TImageType::ImageDimension>                          StructuringElementType;
  typedef BinaryDilateImageFilter<TImageType, TImageType, StructuringElementType>            BinaryDilateFilterType;
  typedef typename  BinaryDilateFilterType::Pointer                                          BinaryDilateFilterPointer;
  typedef DiscreteGaussianImageFilter<TImageType, TImageType>                                GaussianFilterType;
  typedef typename GaussianFilterType::Pointer                                               GaussianFilterPointer;
  typedef InjectSourceImageGreaterThanZeroIntoTargetImageFilter<TImageType, 
                                                                TImageType, 
                                                                TImageType>                  InjectSourceImageGreaterThanZeroIntoTargetImageFilterType;
  typedef typename InjectSourceImageGreaterThanZeroIntoTargetImageFilterType::Pointer        InjectSourceImageGreaterThanZeroIntoTargetImageFilterPointer;
  
  /** Display */
  void PrintSelf( std::ostream& os, itk::Indent indent ) const;

  /** Set/Get our parameters. */
  itkGetMacro( InitialThreshold, float);
  itkSetMacro( InitialThreshold, float);
  itkGetMacro( FirstNumberOfDilations, unsigned int);
  itkSetMacro( FirstNumberOfDilations, unsigned int);
  itkGetMacro( SecondNumberOfDilations, unsigned int);
  itkSetMacro( SecondNumberOfDilations, unsigned int);
  itkGetMacro( GaussianFWHM, float);
  itkSetMacro( GaussianFWHM, float);

protected:

  ExtendedBrainMaskWithSmoothDropOffCompositeFilter();
  void GenerateData();

private:

  ExtendedBrainMaskWithSmoothDropOffCompositeFilter(Self&);   // intentionally not implemented
  void operator=(const Self&);                                // intentionally not implemented

  float        m_InitialThreshold;
  unsigned int m_FirstNumberOfDilations;
  unsigned int m_SecondNumberOfDilations;
  float        m_GaussianFWHM; 
  
  /** These are the filters we are wrapping up. */
  BinaryThresholdFilterPointer m_ThresholdFilter;
  BinaryDilateFilterPointer m_FirstDilateFilter;
  BinaryDilateFilterPointer m_SecondDilateFilter;
  GaussianFilterPointer m_GaussianFilter;
  InjectSourceImageGreaterThanZeroIntoTargetImageFilterPointer m_InjectorFilter;
}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkExtendedBrainMaskWithSmoothDropOffCompositeFilter.txx"
#endif

#endif
