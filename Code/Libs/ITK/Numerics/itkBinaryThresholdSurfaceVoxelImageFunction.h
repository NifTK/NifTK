/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkBinaryThresholdSurfaceVoxelImageFunction.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkBinaryThresholdSurfaceVoxelImageFunction_h
#define __itkBinaryThresholdSurfaceVoxelImageFunction_h

#include "itkImageFunction.h"

namespace itk
{

/** \class BinaryThresholdSurfaceVoxelImageFunction
 * \brief Returns true is the value of an image lies within a range 
 *        of thresholds
 * This ImageFunction returns true (or false) if the pixel value lies
 * within (outside) a lower and upper threshold value. The threshold
 * range can be set with the ThresholdBelow, ThresholdBetween or
 * ThresholdAbove methods.  The input image is set via method
 * SetInputImage().
 *
 * Methods Evaluate, EvaluateAtIndex and EvaluateAtContinuousIndex
 * respectively evaluate the function at an geometric point, image index
 * and continuous image index.
 *
 * \ingroup ImageFunctions
 * 
 */
template <class TInputImage, class TCoordRep = float>
class ITK_EXPORT BinaryThresholdSurfaceVoxelImageFunction : 
  public ImageFunction<TInputImage,bool,TCoordRep> 
{
public:
  /** Standard class typedefs. */
  typedef BinaryThresholdSurfaceVoxelImageFunction              Self;
  typedef ImageFunction<TInputImage,bool,TCoordRep> Superclass;
  typedef SmartPointer<Self>                        Pointer;
  typedef SmartPointer<const Self>                  ConstPointer;
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(BinaryThresholdSurfaceVoxelImageFunction, ImageFunction);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** InputImageType typedef support. */
  typedef typename Superclass::InputImageType InputImageType;
  
  /** Typedef to describe the type of pixel. */
  typedef typename TInputImage::PixelType PixelType;

  /** Dimension underlying input image. */
  itkStaticConstMacro(ImageDimension, unsigned int,Superclass::ImageDimension);

  /** Point typedef support. */
  typedef typename Superclass::PointType PointType;

  /** Index typedef support. */
  typedef typename Superclass::IndexType IndexType;

  /** Region typedef support. */
  typedef typename InputImageType::RegionType RegionType;

  /** ContinuousIndex typedef support. */
  typedef typename Superclass::ContinuousIndexType ContinuousIndexType;

  /** BinaryThreshold the image at a point position
   *
   * Returns true if the image intensity at the specified point position
   * satisfies the threshold criteria.  The point is assumed to lie within
   * the image buffer.
   *
   * ImageFunction::IsInsideBuffer() can be used to check bounds before
   * calling the method. */

  virtual bool Evaluate( const PointType& point ) const
    {
    IndexType index;
    this->ConvertPointToNearestIndex( point, index );
    return ( this->EvaluateAtIndex( index ) );
    }

  /** BinaryThreshold the image at a continuous index position
   *
   * Returns true if the image intensity at the specified point position
   * satisfies the threshold criteria.  The point is assumed to lie within
   * the image buffer.
   *
   * ImageFunction::IsInsideBuffer() can be used to check bounds before
   * calling the method. */
  virtual bool EvaluateAtContinuousIndex( 
    const ContinuousIndexType & index ) const
    {
    IndexType nindex;

    this->ConvertContinuousIndexToNearestIndex (index, nindex);
    return this->EvaluateAtIndex(nindex);
    }

  /** BinaryThreshold the image at an index position.
   *
   * Returns true if the image intensity at the specified point position
   * satisfies the threshold criteria.  The point is assumed to lie within
   * the image buffer.
   *
   * ImageFunction::IsInsideBuffer() can be used to check bounds before
   * calling the method. */
  virtual bool EvaluateAtIndex( const IndexType & index ) const
    {
      PixelType value = this->GetInputImage()->GetPixel(index);

      RegionType region = this->GetInputImage()->GetBufferedRegion();
      if ( m_Lower <= value && value <= m_Upper) {
	
	IndexType indexAbove = index;
	indexAbove[1]--;

	if ( region.IsInside( indexAbove ) ) {
	  PixelType valueAbove = this->GetInputImage()->GetPixel(indexAbove);
	      
	  if ( ! ( m_Lower <= valueAbove && valueAbove <= m_Upper ) )
	    return true;
	}
#if 0		
	int i;
	for (i=-1; i<=1; i+=2) {
	  indexAbove = index;
	  indexAbove[0] = index[0] + i;
	  indexAbove[1]--;

	  if ( region.IsInside( indexAbove ) ) {
	    PixelType valueAbove = this->GetInputImage()->GetPixel(indexAbove);
	      
	    if ( ! ( m_Lower <= valueAbove && valueAbove <= m_Upper ) )
	      return true;
	  }
	}
	
	for (i=-1; i<=1; i+=2) {
	  indexAbove = index;
	  indexAbove[1]--;
	  indexAbove[2] = index[2] + i;

	  if ( region.IsInside( indexAbove ) ) {
	    PixelType valueAbove = this->GetInputImage()->GetPixel(indexAbove);
	      
	    if ( ! ( m_Lower <= valueAbove && valueAbove <= m_Upper ) )
	      return true;
	  }
	}
#endif
      }
      return false;
    }

  /** Get the lower threshold value. */
  itkGetConstReferenceMacro(Lower,PixelType);

  /** Get the upper threshold value. */
  itkGetConstReferenceMacro(Upper,PixelType);

  /** Values greater than or equal to the value are inside. */
  void ThresholdAbove(PixelType thresh);
  
  /** Values less than or equal to the value are inside. */
  void ThresholdBelow(PixelType thresh);

  /** Values that lie between lower and upper inclusive are inside. */
  void ThresholdBetween(PixelType lower, PixelType upper);

protected:
  BinaryThresholdSurfaceVoxelImageFunction();
  ~BinaryThresholdSurfaceVoxelImageFunction(){};
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  BinaryThresholdSurfaceVoxelImageFunction( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented

  PixelType m_Lower;
  PixelType m_Upper;
};

} // end namespace itk


// Define instantiation macro for this template.
#define ITK_TEMPLATE_BinaryThresholdSurfaceVoxelImageFunction(_, EXPORT, x, y) namespace itk { \
  _(2(class EXPORT BinaryThresholdSurfaceVoxelImageFunction< ITK_TEMPLATE_2 x >)) \
  namespace Templates { typedef BinaryThresholdSurfaceVoxelImageFunction< ITK_TEMPLATE_2 x > \
                               BinaryThresholdSurfaceVoxelImageFunction##y; } \
  }

#if ITK_TEMPLATE_EXPLICIT
# include "Templates/itkBinaryThresholdSurfaceVoxelImageFunction+-.h"
#endif

#if ITK_TEMPLATE_TXX
# include "itkBinaryThresholdSurfaceVoxelImageFunction.txx"
#endif

#endif
