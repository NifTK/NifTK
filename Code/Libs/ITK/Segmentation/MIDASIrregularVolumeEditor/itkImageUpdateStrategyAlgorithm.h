/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 12:21:57 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7494 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef ITKIMAGEUPDATESTATEGYALGORITHM_H
#define ITKIMAGEUPDATESTATEGYALGORITHM_H

#include "itkObject.h"
#include "itkImage.h"

namespace itk
{

/**
 * \class ImageUpdateStrategyAlgorithm
 * \brief Base class for algorithms that can be plugged into a Strategy pattern.
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT ImageUpdateStrategyAlgorithm : public Object {

public:

  /** Standard class typedefs */
  typedef ImageUpdateStrategyAlgorithm    Self;
  typedef Object                          Superclass;
  typedef SmartPointer<Self>              Pointer;
  typedef SmartPointer<const Self>        ConstPointer;

  /** Additional typedefs, as this is the ITK way. */
  typedef TPixel                          PixelType;
  typedef Image<TPixel, VImageDimension>  ImageType;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageUpdateStrategyAlgorithm, Object);

  /** Main Virtual Method That All Subclasses Must Implement */
  virtual ImageType* Execute(ImageType* imageToBeModified) = 0;

protected:
  ImageUpdateStrategyAlgorithm();
  virtual ~ImageUpdateStrategyAlgorithm() {}

private:
  ImageUpdateStrategyAlgorithm(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageUpdateStrategyAlgorithm.txx"
#endif

#endif // ITKIMAGEUPDATESTATEGYPROCESSOR_H
