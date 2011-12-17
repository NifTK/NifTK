/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-10-08 16:21:33 +0100 (Fri, 08 Oct 2010) $
 Revision          : $Revision: 4004 $
 Last modified by  : $Author: jhh $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkHessianToRaundahlStripinessMeasureImageFilter_h
#define __itkHessianToRaundahlStripinessMeasureImageFilter_h

#include "itkSymmetricSecondRankTensor.h"
#include "itkSymmetricEigenAnalysisImageFilter.h"

namespace itk
{
/** \class HessianToRaundahlStripinessMeasureImageFilter
 * \brief A filter to enhance stripey features as per Raundahl et al
 * "Automated Effect-Specific Mammographic Pattern Measures", TMI 27:8 2008
 *
 * \sa MultiScaleHessianBasedMeasureImageFilter 
 * \sa Hessian3DToVesselnessMeasureImageFilter
 * \sa HessianSmoothedRecursiveGaussianImageFilter 
 * \sa SymmetricEigenAnalysisImageFilter
 * \sa SymmetricSecondRankTensor
 * 
 * \ingroup IntensityImageFilters TensorObjects
 *
 */
  
template < typename TInputImage, typename TOutputImage > 
class ITK_EXPORT HessianToRaundahlStripinessMeasureImageFilter : public
ImageToImageFilter< TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef HessianToRaundahlStripinessMeasureImageFilter Self;

  typedef ImageToImageFilter< TInputImage, TOutputImage >  Superclass;

  typedef SmartPointer< Self >                  Pointer;
  typedef SmartPointer< const Self >            ConstPointer;
  
  typedef typename Superclass::InputImageType   InputImageType;
  typedef typename Superclass::OutputImageType  OutputImageType;
  typedef typename InputImageType::PixelType    InputPixelType;
  typedef typename OutputImageType::PixelType   OutputPixelType;
  
  /** Image dimension */
  itkStaticConstMacro(ImageDimension, unsigned int, ::itk::GetImageDimension<InputImageType>::ImageDimension);

  typedef double                                                    EigenValueType;
  typedef itk::FixedArray< EigenValueType, itkGetStaticConstMacro( ImageDimension ) > EigenValueArrayType;
  typedef itk::Image< EigenValueArrayType, itkGetStaticConstMacro( ImageDimension ) > EigenValueImageType;

  typedef SymmetricEigenAnalysisImageFilter< 
    InputImageType, EigenValueImageType >                           EigenAnalysisFilterType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Toggle scaling the objectness measure with the magnitude of the largest absolute eigenvalue */ 
  itkSetMacro(ScaleStripinessMeasure,bool);
  itkGetConstMacro(ScaleStripinessMeasure,bool);
  itkBooleanMacro(ScaleStripinessMeasure);

  /** Enhance bright structures on a dark background if true, the opposite if false. */
  itkSetMacro(BrightObject,bool);
  itkGetConstMacro(BrightObject,bool);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(DoubleConvertibleToOutputCheck,(Concept::Convertible<double, OutputPixelType>));
  /** End concept checking */
#endif
  
protected:
  HessianToRaundahlStripinessMeasureImageFilter();
  ~HessianToRaundahlStripinessMeasureImageFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;
  
  /** Generate Data */
  void GenerateData(void);

private:
  HessianToRaundahlStripinessMeasureImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typename EigenAnalysisFilterType::Pointer m_SymmetricEigenValueFilter;

  bool                   m_BrightObject;
  bool                   m_ScaleStripinessMeasure;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkHessianToRaundahlStripinessMeasureImageFilter.txx"
#endif
  
#endif
