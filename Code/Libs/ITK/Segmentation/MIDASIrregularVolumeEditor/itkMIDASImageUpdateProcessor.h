/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKMIDASIMAGEUPDATEPROCESSOR_H
#define ITKMIDASIMAGEUPDATEPROCESSOR_H

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "itkImage.h"

namespace itk
{

/**
 * \class MIDASImageUpdateProcessor
 * \brief Class that takes a pointer to a destination image, and applies changes
 * directly to it and enablng undo/redo. In practice, this may result in large
 * memory overhead, so, if we are using this for undo/redo we should consider
 * using a small undo/redo stack or small regions.
 *
 * At this level of the hierarchy, we basically store a reference to the output
 * image, and define Undo/Redo methods. Its up to sub-classes to do the rest.
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT MIDASImageUpdateProcessor : public Object {

public:

  /** Standard class typedefs */
  typedef MIDASImageUpdateProcessor      Self;
  typedef Object                    Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASImageUpdateProcessor, Object);

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
  MIDASImageUpdateProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~MIDASImageUpdateProcessor() {}

  virtual void ValidateInputs();

private:
  MIDASImageUpdateProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  ImagePointer m_DestinationImage;

}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASImageUpdateProcessor.txx"
#endif

#endif // ITKMIDASIMAGEUPDATEPROCESSOR_H
