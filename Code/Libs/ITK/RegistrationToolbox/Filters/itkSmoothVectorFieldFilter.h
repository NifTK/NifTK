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
#ifndef __itkSmoothVectorFieldFilter_h
#define __itkSmoothVectorFieldFilter_h
#include "itkImage.h"
#include "itkVector.h"
#include "itkInPlaceImageFilter.h"
#include "itkNeighborhoodOperator.h"
#include "itkVectorNeighborhoodOperatorImageFilter.h"

namespace itk {
/** 
 * \class SmoothVectorFieldFilter
 * \brief Abstract base class that takes a vector field as input and smoothes it.
 *
 * As of 16/01/2010, I have extended this to cope with time varying vector fields.
 * The previous version of this class, only did 2D or 3D. This will do N/M-dimensional.
 * However, be aware that for example, you can have N=2 dimensional vectors, and
 * M=3 dimensional images. So, if N != M, then do you really want to try smoothing?
 * It might be better to set the sigma to zero if you don't wan't smoothing in all dimensions.
 *
 * As of 16/01/2010, also note, that we now extend InPlaceImageFilter.
 * This is because with time varying velocity fields, on decent size images,
 * you need a lot of memory. So, rather than have an input/output buffer,
 * we use one buffer. But this filter defaults to OFF to minimise impact
 * on other programs that are not using time varying velocity fields.
 * Obviously, you need some memory to perform the smoothing, so each
 * subclass should clear references as soon as its done.
 *
 */
template <
	class TScalarType,                      // Data type for the vectors
	unsigned int NumberImageDimensions = 3, // Number of image dimensions
	unsigned int NumberVectorDimensions = NumberImageDimensions>   // Number of dimensions in the vectors
class ITK_EXPORT SmoothVectorFieldFilter :
  public InPlaceImageFilter<Image< Vector<TScalarType, NumberVectorDimensions>,  NumberImageDimensions>, // Input image
                            Image< Vector<TScalarType, NumberVectorDimensions>,  NumberImageDimensions>  // Output image
                           >
{
public:

  /** Standard "Self" typedef. */
  typedef SmoothVectorFieldFilter                             Self;
  typedef ImageToImageFilter<
                             Image<
                               Vector<
                                 TScalarType,
                                 NumberVectorDimensions>,
                               NumberImageDimensions>,
                             Image<
                               Vector<
                                 TScalarType,
                                 NumberVectorDimensions>,
                             NumberImageDimensions>
                            >                                 Superclass;
  typedef SmartPointer<Self>                                  Pointer;
  typedef SmartPointer<const Self>                            ConstPointer;
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(SmoothVectorFieldFilter, ImageToImageFilter);

  /** Get the number of image dimensions we are working in. */
  itkStaticConstMacro(ImageDimensions, unsigned int, NumberImageDimensions);

  /** Get the number of vector dimensions we are working in. */
  itkStaticConstMacro(VectorDimensions, unsigned int, NumberVectorDimensions);

  /** Standard typedefs. */
  typedef TScalarType                                                                 OutputDataType;
  typedef Vector< OutputDataType, NumberVectorDimensions >                            OutputPixelType;
  typedef Image< OutputPixelType, NumberImageDimensions >                             OutputImageType;
  typedef typename OutputImageType::Pointer                                           OutputImagePointer;
  typedef typename Superclass::InputImageType                                         InputImageType;
  typedef typename InputImageType::Pointer                                            InputImagePointer;
  typedef typename Superclass::InputImageRegionType                                   RegionType;
  typedef NeighborhoodOperator<TScalarType,  NumberImageDimensions>                   NeighborhoodOperatorType;
  typedef VectorNeighborhoodOperatorImageFilter<InputImageType, OutputImageType>      SmootherFilterType;
  typedef typename SmootherFilterType::Pointer                                        SmootherFilterPointer;

  /** Writes image. Just calls ITK writers, so if they don't support it, expect an Exception. */
  void WriteVectorImage(std::string filename);

protected:

  SmoothVectorFieldFilter();
  ~SmoothVectorFieldFilter() {};
  
  /** Make sure we request the right size (all of the image). */
  virtual void GenerateInputRequestedRegion() throw(InvalidRequestedRegionError);

  /** Tell the pipeline the correct size (all of the image). */
  virtual void EnlargeOutputRequestedRegion(DataObject *output);

  /**
   * The main filter method. Note, single threaded. This is OK, as
   * this filter is a composite filter, combining filters internally,
   * which are multi-threaded. This implements TemplateMethod [2],
   * running seperable filters in each direction, x, y, z, t (etc.)
   * only calling the subclass for GetOperator.
   */
  virtual void GenerateData();

  /** Get the operator from the subclass. The subclass creates it new, and this class deletes it, once used. */
  virtual NeighborhoodOperatorType* CreateOperator(int dimension) = 0;

private:
  
  /**
   * Prohibited copy and assignment. 
   */
  SmoothVectorFieldFilter(const Self&); 
  void operator=(const Self&); 

};

} // end namespace.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSmoothVectorFieldFilter.txx"
#endif

#endif
