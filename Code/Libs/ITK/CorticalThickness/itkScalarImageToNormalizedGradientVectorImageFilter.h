/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkScalarImageToNormalizedGradientVectorImageFilter_h
#define __itkScalarImageToNormalizedGradientVectorImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkVector.h>
#include <itkImage.h>
#include <itkGradientImageFilter.h>
#include <itkGradientRecursiveGaussianImageFilter.h>
#include <itkCovariantVector.h>
#include "itkNormaliseVectorFilter.h"
#include <itkGaussianSmoothVectorFieldFilter.h>

namespace itk {

/** 
 * \class ScalarImageToNormalizedGradientVectorImageFilter
 * \brief This class takes scalar image as input, and outputs a vector field of image gradient.
 *
 * This class implements step (8) in in Jones et al. Human Brain Mapping
 * 11:12-32 (2000). The input should be the output of itkLaplacianSolverImageFilter,
 * but the filter will work for any scalar image. We simply compute the 
 * derivative using 2 point central differences, and normalize it to unit length.
 * The output vector length takes into account anisotropic voxels, so the
 * output vectors are in mm space, and hence are 1mm long.
 * 
 * Additionally, we added a member variable UseMillimetreScaling, so that 
 * if true, the size in millimetre of the voxel is taken into account. Defaults to on.
 * 
 * Additionally, we added a member variable, DivideByTwo, so that if 
 * true, the difference of the pixel + and pixel - is divided by two. Defaults to on.
 * 
 * Additionally, we added member variable Normalize, which is a boolean
 * to turn normalization to unit length on/off. Defaults to on.
 * 
 * Additionally, we added a pad value. If the current pixel, or +/- 1 pixel
 * in any dimension is == pad value, then the gradient is ignored.
 * 
 * As of 25th Nov 2009, we have wrapped the classes GradientImageFilter
 * and GradientRecursiveGaussianImageFilter.
 * 
 * As of 1st Dec 2009, I have also wrapped smoothing of the vector field.
 * You can supply a sigma (default 0 - i.e. no smoothing) and the resultant
 * vectors will be smoothed. Also, if normalisation is on, the the smoothed
 * vectors come out, normalised to unit length.
 * 
 * \sa LaplacianSolverImageFilter
 * \sa RelaxStreamlinesFilter
 * \sa IntegrateStreamlinesFilter
 * \sa LagrangianInitializedRelaxStreamlinesFilter
 */

template< class TInputImage, typename TScalarType >
class ITK_EXPORT ScalarImageToNormalizedGradientVectorImageFilter :
  public ImageToImageFilter<TInputImage, 
                            Image<  
                              Vector< TScalarType, ::itk::GetImageDimension<TInputImage>::ImageDimension>, 
                              ::itk::GetImageDimension<TInputImage>::ImageDimension> >
{
public:

  typedef TScalarType                                                                   VectorDataType;
  /** Standard "Self" typedef. */
  typedef ScalarImageToNormalizedGradientVectorImageFilter                              Self;
  typedef ImageToImageFilter<TInputImage, 
                             Image<  
                               Vector< VectorDataType, 
                                 ::itk::GetImageDimension<TInputImage>::ImageDimension>, 
                               ::itk::GetImageDimension<TInputImage>::ImageDimension> > Superclass;
  typedef SmartPointer<Self>                                                            Pointer;
  typedef SmartPointer<const Self>                                                      ConstPointer;
  
  /** To define what derivative type we will try. */
  typedef enum {CENTRAL_DIFFERENCES, DERIVATIVE_OPERATOR, DERIVATIVE_OF_GAUSSIAN} DerivativeType;
  
  /** Method for creation through the object factory.  */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(ScalarImageToNormalizedGradientVectorImageFilter, ImageToImageFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TInputImage::ImageDimension);

  /** Standard typedefs. */
  typedef Vector< VectorDataType, itkGetStaticConstMacro(Dimension) >                 OutputPixelType;
  typedef CovariantVector< VectorDataType, itkGetStaticConstMacro(Dimension) >        CovariantVectorType;
  typedef Image< OutputPixelType, itkGetStaticConstMacro(Dimension) >                 OutputImageType;
  typedef Image< CovariantVectorType, itkGetStaticConstMacro(Dimension) >             CovariantVectorImageType;
  typedef typename CovariantVectorImageType::Pointer                                  CovariantVectorImagePointer;
  typedef typename Superclass::InputImageType                                         InputImageType;
  typedef typename InputImageType::PixelType                                          InputPixelType;
  typedef typename InputImageType::RegionType                                         InputRegionType;
  typedef GradientImageFilter<TInputImage, VectorDataType, VectorDataType>            GradientImageFilterType;
  typedef typename GradientImageFilterType::Pointer                                   GradientImageFilterPointer;
  typedef GradientRecursiveGaussianImageFilter<TInputImage, CovariantVectorImageType> GradientRecursiveGaussianImageFilterType;
  typedef typename GradientRecursiveGaussianImageFilterType::Pointer                  GradientRecursiveGaussianImageFilterPointer;
  typedef GaussianSmoothVectorFieldFilter<VectorDataType, itkGetStaticConstMacro(Dimension), itkGetStaticConstMacro(Dimension)> GaussianSmoothFilterType;
  typedef typename GaussianSmoothFilterType::Pointer                                          GaussianSmoothFilterPointer;
  typedef NormaliseVectorFilter<VectorDataType, itkGetStaticConstMacro(Dimension) >           NormaliseFilterType;
  typedef typename NormaliseFilterType::Pointer                                               NormaliseFilterPointer;
  
  /** Sets the scalar image, at input 0. */
  void SetScalarImage(const InputImageType *image) {this->SetNthInput(0, const_cast<InputImageType *>(image)); }

  /** Scale vector by voxel size. Default on. */  
  itkSetMacro(UseMillimetreScaling, bool);
  itkGetMacro(UseMillimetreScaling, bool);

  /** Scale vector by dividing by two. Default on. */
  itkSetMacro(DivideByTwo, bool);
  itkGetMacro(DivideByTwo, bool);

  /** Turn normalization to unit length on/off. Default on. */
  itkSetMacro(Normalize, bool);
  itkGetMacro(Normalize, bool);

  /** 
   * Set a pad value, whereby we skip gradient evaluation. 
   * Default -1, which you should probably set to at least zero. 
   * I left it at -1 because various cortical thickness test 
   * images had lots of zeros in which are important. 
   */
  itkSetMacro(PadValue, InputPixelType);
  itkGetMacro(PadValue, InputPixelType);

  /**
   * Set/Get the derivative type, as this class wraps itkGradientImageFilter and itkGradientRecursiveGaussianImageFilter.
   */
  itkSetMacro(DerivativeType, DerivativeType);
  itkGetMacro(DerivativeType, DerivativeType);
  
  /** Set/Get the sigma for Gaussian smoothing. Default 0, i.e. off. */
  itkSetMacro(Sigma, double);
  itkGetMacro(Sigma, double);
  
protected:
  
  ScalarImageToNormalizedGradientVectorImageFilter();
  ~ScalarImageToNormalizedGradientVectorImageFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  // The main method to implement in derived classes.
  virtual void GenerateData();
    
private:
  
  /**
   * Prohibited copy and assignment. 
   */
  ScalarImageToNormalizedGradientVectorImageFilter(const Self&); 
  void operator=(const Self&); 

  /** Turn on/off millimetre scaling. */
  bool m_UseMillimetreScaling;

  /** Turn on/off division by two. */
  bool m_DivideByTwo;
  
  /** Turn on/off normalization. */
  bool m_Normalize;
  
  /** Set the sigma for smoothing. Default 0. */
  double m_Sigma;
  
  /** Set a pad value. */
  InputPixelType m_PadValue;
  
  /** Flag for derivative type. */
  DerivativeType m_DerivativeType;
  
  /** Gradient image filter that we can delegate to. */
  GradientImageFilterPointer m_GradientImageFilter;
  
  /** A gradient recursive filter that we can delegate to. */
  GradientRecursiveGaussianImageFilterPointer m_GradientRecursiveGaussianImageFilter;
  
  /** A filter to do some smoothing. */
  GaussianSmoothFilterPointer m_GaussianSmoothFilter;
  
  /** A filter to normalise vectors. */
  NormaliseFilterPointer m_NormalizeFilter;
  
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkScalarImageToNormalizedGradientVectorImageFilter.txx"
#endif

#endif
