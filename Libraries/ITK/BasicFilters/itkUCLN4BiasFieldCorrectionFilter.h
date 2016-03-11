/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkUCLN4BiasFieldCorrectionFilter_h
#define itkUCLN4BiasFieldCorrectionFilter_h

#include <itkImageToImageFilter.h>
#include <itkMacro.h>
#include <itkArray.h>


namespace itk {
  
/** \class UCLN4BiasFieldCorrectionFilter
 *
 * \brief N4 bias field correction algorithm contributed to ITK by
 * Nicholas J. Tustison and James C. Gee. \n\nThis program runs the
 * ITK N4BiasFieldCorrectionImageFilter on an image to correct
 * nonuniformity commonly associated with MR images. The algorithm
 * assumes a simple parametric model (Gaussian) for the bias field and
 * does not require tissue class segmentation. References: J.G. Sled,
 * A.P. Zijdenbos and A.C. Evans. "A Nonparametric Method for
 * Automatic Correction of Intensity Nonuniformity in Data" IEEE
 * Transactions on Medical Imaging, Vol 17, No 1. Feb
 * 1998. N.J. Tustison, B.B. Avants, P.A. Cook, Y. Zheng, A. Egan,
 * P.A. Yushkevich, and J.C. Gee. "N4ITK: Improved N3 Bias Correction"
 * IEEE Transactions on Medical Imaging, 29(6):1310-1320, June 2010.
 *
 */

template < class TInputImage, class TOutputImage >
class ITK_EXPORT UCLN4BiasFieldCorrectionFilter : 
    public ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef UCLN4BiasFieldCorrectionFilter                   Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(UCLN4BiasFieldCorrectionFilter, ImageToImageFilter);

  /** ImageDimension enumeration. */
  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int, TOutputImage::ImageDimension);

  /** Inherit types from Superclass. */
  typedef typename Superclass::InputImageConstPointer InputImageConstPointer;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::InputImageType         InputImageType;

  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::OutputImageType        OutputImageType;

  typedef unsigned char MaskPixelType;
  typedef typename itk::Image<MaskPixelType, ImageDimension> MaskImageType;
  typedef typename MaskImageType::Pointer                    MaskImagePointer;


#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(SameDimensionCheck,
    (Concept::SameDimension<ImageDimension, OutputImageDimension>));
  itkConceptMacro(OutputHasNumericTraitsCheck,
    (Concept::HasNumericTraits<typename TOutputImage::PixelType>));
  /** End concept checking */
#endif

  itkSetMacro( Subsampling, float );
  itkSetMacro( SplineOrder, float );
  itkSetMacro( NumberOfHistogramBins, float );
  itkSetMacro( WeinerFilterNoise, float );
  itkSetMacro( BiasFieldFullWidthAtHalfMaximum, float );
  itkSetMacro( MaximumNumberOfIterations, float );
  itkSetMacro( ConvergenceThreshold, float );
  itkSetMacro( NumberOfFittingLevels, float );
  itkSetMacro( NumberOfControlPoints, float );

  /// Set an optional mask image
  itkSetObjectMacro( Mask, MaskImageType );
  /// Get the computed mask image
  itkGetObjectMacro( Mask, MaskImageType );

  /// Get the computed bias field
  itkGetObjectMacro( BiasField, InputImageType );

protected:
  UCLN4BiasFieldCorrectionFilter();
  ~UCLN4BiasFieldCorrectionFilter() {};

  float m_Subsampling;
  float m_SplineOrder;
  float m_NumberOfHistogramBins;
  float m_WeinerFilterNoise;
  float m_BiasFieldFullWidthAtHalfMaximum;
  float m_MaximumNumberOfIterations;
  float m_ConvergenceThreshold;
  float m_NumberOfFittingLevels;
  float m_NumberOfControlPoints;

  MaskImagePointer m_Mask;
  InputImagePointer m_BiasField;

  void PrintSelf(std::ostream&os, Indent indent) const;

  /** Generate the output data. */
  void GenerateData();


private:
  UCLN4BiasFieldCorrectionFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};


} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkUCLN4BiasFieldCorrectionFilter.txx"
#endif

#endif
