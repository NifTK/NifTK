/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-05 13:51:34 +0100 (Mon, 05 Sep 2011) $
 Revision          : $Revision: 7237 $
 Last modified by  : $Author: jhh $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkLewisGriffinRecursiveGaussianImageFilter_h
#define __itkLewisGriffinRecursiveGaussianImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkNumericTraits.h"

namespace itk
{
  
/** \class LewisGriffinRecursiveGaussianImageFilter
 * \brief Lewis Griffin's implementation of a recursive Gaussian
 * filter which does not produce the oscillations that
 * itk::RecursiveSeparableImageFilter does.
 */
template <typename TInputImage, typename TOutputImage=TInputImage>
class ITK_EXPORT LewisGriffinRecursiveGaussianImageFilter :
    public ImageToImageFilter<TInputImage,TOutputImage> 
{
public:
  /** Standard class typedefs. */
  typedef LewisGriffinRecursiveGaussianImageFilter       Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>   Superclass;
  typedef SmartPointer<Self>                             Pointer;
  typedef SmartPointer<const Self>                       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Type macro that defines a name for this class. */
  itkTypeMacro( LewisGriffinRecursiveGaussianImageFilter, ImageToImageFilter );

  /** Smart pointer typedef support.  */
  typedef typename TInputImage::Pointer       InputImagePointer;
  typedef typename TInputImage::ConstPointer  InputImageConstPointer;

  /** Real type to be used in internal computations. RealType in general is
   * templated over the pixel type. (For example for vector or tensor pixels,
   * RealType is a vector or a tensor of doubles.) ScalarRealType is a type 
   * meant for scalars.
   */
  typedef typename TInputImage::PixelType                        InputPixelType;
  typedef typename NumericTraits<InputPixelType>::RealType       RealType;
  typedef typename NumericTraits<InputPixelType>::ScalarRealType ScalarRealType;

  typedef typename TOutputImage::RegionType                      OutputImageRegionType;

  /** Type of the input image */
  typedef TInputImage      InputImageType;

  /** Type of the output image */
  typedef TOutputImage      OutputImageType;

  /** Get the direction in which the filter is to be applied. */   
  itkGetConstMacro(Direction, unsigned int);

  /** Set the direction in which the filter is to be applied. */   
  itkSetMacro(Direction, unsigned int);

  /** Set Input Image. */
  void SetInputImage( const TInputImage * );
    
  /** Get Input Image. */
  const TInputImage * GetInputImage( void );

  /** Set/Get the Sigma, measured in world coordinates, of the Gaussian
   * kernel.  The default is 1.0.  */   
  itkGetConstMacro( Sigma, ScalarRealType );
  itkSetMacro( Sigma, ScalarRealType );

  /** Enum type that indicates if the filter applies the equivalent operation
      of convolving with a gaussian, first derivative of a gaussian or the 
      second derivative of a gaussian.  */
  typedef  enum { ZeroOrder, FirstOrder, SecondOrder } OrderEnumType;
 
  /** Set/Get the Order of the Gaussian to convolve with. 
      \li ZeroOrder is equivalent to convolving with a Gaussian.  This
      is the default.
      \li FirstOrder is equivalent to convolving with the first derivative of a Gaussian.
      \li SecondOrder is equivalent to convolving with the second derivative of a Gaussian.
    */
  itkSetMacro( Order, OrderEnumType );
  itkGetConstMacro( Order, OrderEnumType );

  /** Explicitly set a zeroth order derivative. */
  void SetZeroOrder();

  /** Explicitly set a first order derivative. */
  void SetFirstOrder();

  /** Explicitly set a second order derivative. */
  void SetSecondOrder();

protected:
  LewisGriffinRecursiveGaussianImageFilter();
  virtual ~LewisGriffinRecursiveGaussianImageFilter();

  LewisGriffinRecursiveGaussianImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  void PrintSelf(std::ostream& os, Indent indent) const;

  /** GenerateData (apply) the filter. */   
  void BeforeThreadedGenerateData();
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, int threadId );

  int SplitRequestedRegion(int i, int num, OutputImageRegionType& splitRegion);

  /** LewisGriffinRecursiveGaussianImageFilter needs all of the input only in the
   *  "Direction" dimension. Therefore we enlarge the output's
   *  RequestedRegion to this. Then the superclass's
   *  GenerateInputRequestedRegion method will copy the output region
   *  to the input.
   *
   * \sa ImageToImageFilter::GenerateInputRequestedRegion() 
   */
  void EnlargeOutputRequestedRegion(DataObject *output);

  /** Set up the coefficients of the filter to approximate a specific kernel.
   * Typically it can be used to approximate a Gaussian or one of its
   * derivatives. Parameter is the spacing along the dimension to
   * filter. */
  virtual void SetUp(ScalarRealType spacing);

  /** Apply the Recursive Filter to an array of data.  This method is called
   * for each line of the volume.  */
  void FilterDataArray(RealType *outs, RealType *data, int ln);

  /// Zero order Gaussian
  RealType GaussianZeroOrder(RealType x, RealType sigma);
  /// First order Gaussian
  RealType GaussianFirstOrder(RealType x, RealType sigma);
  /// Second order Gaussian
  RealType GaussianSecondOrder(RealType x, RealType sigma);

  /** Direction in which the filter is to be applied
   * this should be in the range [0,ImageDimension-1]. */ 
  unsigned int m_Direction;

  /** Sigma of the gaussian kernel. */   
  ScalarRealType m_Sigma;

  /// Gaussian kernel order
  OrderEnumType m_Order;

  /// The Gaussian kernel
  RealType *m_Kernel;

  /// The size of the kernel
  unsigned int m_KernelSize;
};


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLewisGriffinRecursiveGaussianImageFilter.txx"
#endif


#endif
