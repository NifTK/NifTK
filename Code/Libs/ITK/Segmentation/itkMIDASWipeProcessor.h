/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-30 22:53:06 +0100 (Fri, 30 Sep 2011) $
 Revision          : $Revision: -1 $
 Last modified by  : $Author: $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef ITKMIDASWIPEPROCESSOR_H
#define ITKMIDASWIPEPROCESSOR_H

#include "itkObject.h"
#include "itkMIDASRegionProcessor.h"
#include "itkImageUpdateClearRegionProcessor.h"

namespace itk
{

/**
 * \class MIDASWipeProcessor
 * \brief Base class to support the MIDAS Wipe, Wipe+ and Wipe-
 * operations in the Irregular Volume Editor. The template type TPixel should
 * be that of the segmentation image type (eg. unsigned char for instance).
 *
 * \sa MIDASRegionProcessor
 * \sa ImageUpdateClearRegionProcessor
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT MIDASWipeProcessor : public MIDASRegionProcessor<TPixel, VImageDimension> {

public:

  /** Standard class typedefs */
  typedef MIDASWipeProcessor                            Self;
  typedef MIDASRegionProcessor<TPixel, VImageDimension> Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASWipeProcessor, MIDASRegionProcessor);

  /** Additional typedefs */
  typedef TPixel                                                                       SegmentationPixelType;
  typedef Image<SegmentationPixelType, VImageDimension>                                SegmentationImageType;
  typedef itk::ImageUpdateClearRegionProcessor<SegmentationPixelType, VImageDimension> ClearProcessorType;
  typedef typename ClearProcessorType::Pointer                                         ClearProcessorPointer;

  /** Set/Get the value that is used to wipe the image (normally zero). */
  void SetWipeValue(const SegmentationPixelType value);
  SegmentationPixelType GetWipeValue() const;

protected:

  MIDASWipeProcessor();
  virtual ~MIDASWipeProcessor() {}

private:
  MIDASWipeProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASWipeProcessor.txx"
#endif

#endif // ITKMIDASWIPEPROCESSOR_H
