/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 12:21:57 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7449 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef ITKIMAGEUPDATEPROCESSOR_H
#define ITKIMAGEUPDATEPROCESSOR_H

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "itkImage.h"
#include "itkExtractImageFilter.h"
#include "itkPasteImageFilter.h"

namespace itk
{

/**
 * \class ImageUpdateProcessor
 * \brief Class that takes a pointer to a destination image, and applies changes
 * directly to it, storing the affected sub-region to enable undo/redo. In practice,
 * this may result in large memory overhead, so, if we are using this for undo/redo
 * we should consider using a small undo/redo stack or small regions.
 *
 * Essentially, this base class takes care of undo/redo within a set region.
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT ImageUpdateProcessor : public Object {

public:

  /** Standard class typedefs */
  typedef ImageUpdateProcessor      Self;
  typedef Object                    Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageUpdateProcessor, Object);

  /** Dimension of the image.  This constant is used by functions that are
   * templated over image type (as opposed to being templated over pixel type
   * and dimension) when they need compile time access to the dimension of
   * the image. */
  itkStaticConstMacro(ImageDimension, unsigned int, VImageDimension);

  /** Additional typedefs */
  typedef TPixel PixelType;
  typedef Image<TPixel, VImageDimension>  ImageType;
  typedef typename ImageType::Pointer     ImagePointer;
  typedef typename ImageType::IndexType   IndexType;
  typedef typename ImageType::SizeType    SizeType;
  typedef typename ImageType::RegionType  RegionType;
  typedef itk::ExtractImageFilter<ImageType, ImageType> ExtractImageFilterType;
  typedef typename ExtractImageFilterType::Pointer      ExtractImageFilterPointer;
  typedef itk::PasteImageFilter<ImageType, ImageType>   PasteImageFilterType;
  typedef typename PasteImageFilterType::Pointer        PasteImageFilterPointer;

  /** Set the destination image, which is the image actually modified. This is not a filter with separate input/output. */
  itkSetObjectMacro(DestinationImage, ImageType);
  itkGetObjectMacro(DestinationImage, ImageType);

  /** Set the destination region of interest, which controls the region that is copied into m_BeforeImage and m_After image for Undo/Redo purposes. */
  itkSetMacro(DestinationRegionOfInterest, RegionType);
  itkGetMacro(DestinationRegionOfInterest, RegionType);

  /** This will copy the m_BeforeImage into the m_InputImage */
  void Undo();

  /** This will copy the m_AfterImage into the m_InputImage. This method should also be called to execute the whole process first time round. */
  void Redo();

protected:
  ImageUpdateProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~ImageUpdateProcessor() {}

  /** Returns the after image, so derived classes can apply an update. */
  itkGetObjectMacro(AfterImage, ImageType);
  itkSetObjectMacro(AfterImage, ImageType);

  /** Derived classes calculate whatever update they like, but can only affect the m_AfterImage, which must be within the destination region of interest. */
  virtual void ApplyUpdateToAfterImage() = 0;

private:
  ImageUpdateProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  void ValidateInputs();
  void CopyImageRegionToDestination(ImagePointer sourceImage);

  bool         m_UpdateCalculated;
  ImagePointer m_DestinationImage;
  RegionType   m_DestinationRegionOfInterest;
  ImagePointer m_BeforeImage;
  ImagePointer m_AfterImage;

}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageUpdateProcessor.txx"
#endif

#endif // ITKIMAGEUPDATEPROCESSOR_H
