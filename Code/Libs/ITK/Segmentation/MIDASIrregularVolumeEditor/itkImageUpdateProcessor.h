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

namespace itk
{

/**
 * \class ImageUpdateProcessor
 * \brief Class that takes a pointer to a destination image, and applies changes
 * directly to it and enablng undo/redo. In practice, this may result in large
 * memory overhead, so, if we are using this for undo/redo we should consider
 * using a small undo/redo stack or small regions.
 *
 * At this level of the hierarchy, we basically store a reference to the output
 * image, and define Undo/Redo methods. Its up to sub-classes to do the rest.
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

  /** Set the destination image, which is the image actually modified. This is not a filter with separate input/output. */
  itkSetObjectMacro(DestinationImage, ImageType);
  itkGetObjectMacro(DestinationImage, ImageType);

  /** Sub-classes decide how to implement this. */
  virtual void Undo() = 0;

  /** Sub-classes decide how to implement this. */
  virtual void Redo() = 0;

protected:
  ImageUpdateProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~ImageUpdateProcessor() {}

  virtual void ValidateInputs();

private:
  ImageUpdateProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  ImagePointer m_DestinationImage;

}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageUpdateProcessor.txx"
#endif

#endif // ITKIMAGEUPDATEPROCESSOR_H
