/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkImageAndArray_h
#define __itkImageAndArray_h

#include "itkImage.h"
#include "itkArray.h"


namespace itk
{
/** \class Image
 *  \brief Simultaneous itk::Image and itk::Array (and hence vnl_vector) class.
 *
 * \sa Image
 * \sa Array
 *
 * */
template <class TPixel, unsigned int VImageDimension=2>
class ITK_EXPORT ImageAndArray : public Image<TPixel, VImageDimension>, public Array<TPixel>
{
public:
  /** Standard class typedefs */
  typedef ImageAndArray                    Self;
  typedef Image<TPixel, VImageDimension>   Superclass;
  typedef SmartPointer<Self>           Pointer;
  typedef SmartPointer<const Self>     ConstPointer;
  typedef WeakPointer<const Self>      ConstWeakPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageAndArray, Image);

  /** Pixel typedef support. Used to declare pixel type in filters
   * or other operations. */
  typedef TPixel PixelType;

  /** Typedef alias for PixelType */
  typedef TPixel ValueType;

  /** Internal Pixel representation. Used to maintain a uniform API
   * with Image Adaptors and allow to keep a particular internal
   * representation of data while showing a different external
   * representation. */
  typedef TPixel InternalPixelType;

  typedef PixelType IOPixelType;

  /** Accessor type that convert data between internal and external
   *  representations.  */
  typedef DefaultPixelAccessor< PixelType >    AccessorType;
  typedef DefaultPixelAccessorFunctor< Self >  AccessorFunctorType;

  /** Typedef for the functor used to access a neighborhood of pixel
   * pointers. */
  typedef NeighborhoodAccessorFunctor< Self >  NeighborhoodAccessorFunctorType;

  /** Dimension of the image.  This constant is used by functions that are
   * templated over image type (as opposed to being templated over pixel type
   * and dimension) when they need compile time access to the dimension of
   * the image. */
  itkStaticConstMacro(ImageDimension, unsigned int, VImageDimension);

  /** Container used to store pixels in the image. */
  typedef ImportImageContainer<unsigned long, PixelType> PixelContainer;

  /** Index typedef support. An index is used to access pixel values. */
  typedef typename Superclass::IndexType       IndexType;
  typedef typename Superclass::IndexValueType  IndexValueType;

  /** Offset typedef support. An offset is used to access pixel values. */
  typedef typename Superclass::OffsetType OffsetType;

  /** Size typedef support. A size is used to define region bounds. */
  typedef typename Superclass::SizeType       SizeType;
  typedef typename Superclass::SizeValueType  SizeValueType;

  /** Direction typedef support. A matrix of direction cosines. */
  typedef typename Superclass::DirectionType  DirectionType;

  /** Region typedef support. A region is used to specify a subset of an image. */
  typedef typename Superclass::RegionType  RegionType;

  /** Spacing typedef support.  Spacing holds the size of a pixel.  The
   * spacing is the geometric distance between image samples. */
  typedef typename Superclass::SpacingType SpacingType;

  /** Origin typedef support.  The origin is the geometric coordinates
   * of the index (0,0). */
  typedef typename Superclass::PointType PointType;

  /** A pointer to the pixel container. */
  typedef typename PixelContainer::Pointer        PixelContainerPointer;
  typedef typename PixelContainer::ConstPointer   PixelContainerConstPointer;

  /** Offset typedef (relative position between indices) */
  typedef typename Superclass::OffsetValueType OffsetValueType;

  /** Restore the data object to its initial state. This means releasing
   * memory. */
  virtual void Initialize();

  /** Restore the data object to its initial state. This means releasing
   * memory. */
  virtual void SynchronizeArray();

  /** \brief Access a pixel. This version can be an lvalue.
   *
   * For efficiency, this function does not check that the
   * image has actually been allocated yet. */
  TPixel & operator[](const IndexType &index)
  { return this->Image<TPixel, VImageDimension>::GetPixel(index); }

  /** \brief Access a pixel. This version can only be an rvalue.
   *
   * For efficiency, this function does not check that the
   * image has actually been allocated yet. */
  const TPixel& operator[](const IndexType &index) const
     { return this->Image<TPixel, VImageDimension>::GetPixel(index); }


protected:
  ImageAndArray();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~ImageAndArray() {};


private:
  ImageAndArray(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk

// Define instantiation macro for this template.
#define ITK_TEMPLATE_ImageAndArray(_, EXPORT, x, y) namespace itk { \
  _(2(class EXPORT ImageAndArray< ITK_TEMPLATE_2 x >)) \
  namespace Templates { typedef ImageAndArray< ITK_TEMPLATE_2 x > ImageAndArray##y; } \
  }

#if ITK_TEMPLATE_EXPLICIT
# include "Templates/itkImageAndArray+-.h"
#endif

#if ITK_TEMPLATE_TXX
# include "itkImageAndArray.txx"
#endif

#endif
