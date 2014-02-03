/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramLeftOrRightSideCalculator_h
#define __itkMammogramLeftOrRightSideCalculator_h

#include <itkImage.h>
#include <itkObject.h>
#include <itkMacro.h>
#include <itkImageLinearConstIteratorWithIndex.h>

namespace itk
{

/** \class MammogramLeftOrRightSideCalculator
 *  \brief Computes whether a mammogram is of the left or right breast from the center of mass.
 */

template < class TInputImage >
class ITK_EXPORT MammogramLeftOrRightSideCalculator 
  : public Object
{

public:

  /** Standard class typedefs. */
  typedef MammogramLeftOrRightSideCalculator Self;
  typedef Object                        Superclass;
  typedef SmartPointer< Self >          Pointer;
  typedef SmartPointer< const Self >    ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MammogramLeftOrRightSideCalculator, Object);

  /** Type definition for the input image. */
  typedef TInputImage ImageType;

  /** Pointer type for the image. */
  typedef typename TInputImage::Pointer ImagePointer;

  /** Const Pointer type for the image. */
  typedef typename TInputImage::ConstPointer ImageConstPointer;

  /** Type definition for the input image pixel type. */
  typedef typename TInputImage::PixelType PixelType;

  /** Type definition for the input image index type. */
  typedef typename TInputImage::IndexType IndexType;

  /** Type definition for the input image region type. */
  typedef typename TInputImage::RegionType RegionType;

  typedef typename itk::ImageLinearConstIteratorWithIndex< ImageType > LineIteratorType;


  /// Breast side
  typedef enum {
    UNKNOWN_BREAST_SIDE,
    LEFT_BREAST_SIDE,
    RIGHT_BREAST_SIDE
  } BreastSideType;

  /** Set the input image. */
  itkSetConstObjectMacro(Image, ImageType);

  /** Compute which breast was imaged in the mammogram. */
  void Compute(void) throw (ExceptionObject);

  /** Return the threshold intensity value. */
  itkGetConstMacro(BreastSide, BreastSideType);

  void SetVerbose( bool flag )  { m_FlgVerbose = flag; }
  void SetVerboseOn( void )  { m_FlgVerbose = true;  }
  void SetVerboseOff( void ) { m_FlgVerbose = false; }

protected:

  MammogramLeftOrRightSideCalculator();
  virtual ~MammogramLeftOrRightSideCalculator() { }

  bool m_FlgVerbose;

  BreastSideType m_BreastSide;
  ImageConstPointer  m_Image;

  void PrintSelf(std::ostream & os, Indent indent) const;

private:

  MammogramLeftOrRightSideCalculator(const Self &); //purposely not implemented
  void operator=(const Self &);                //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMammogramLeftOrRightSideCalculator.txx"
#endif

#endif /* __itkMammogramLeftOrRightSideCalculator_h */
