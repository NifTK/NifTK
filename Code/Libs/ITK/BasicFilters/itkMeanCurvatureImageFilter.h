/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3326 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkMeanCurvatureImageFilter_h
#define __itkMeanCurvatureImageFilter_h
#include "itkBasicFiniteDifferenceBaseClassImageFilter.h"

namespace itk
{

/**
 * \class MeanCurvatureImageFilter
 * \brief Class to calculate mean curvature of a scalar image.
 *
 * The output datatype should really be float or double.
 * Implements the formula as written on page 70 of [1] (in 2006 printing of the book).
 *
 * \par REFERENCES
 * \par
 * [1] Sethian, J.A. Level Set Methods and Fast Marching Methods Cambridge University Press. 1996.
 *
 * \sa itkGaussianCurvatureImageFilter
 * \sa itkBasicFiniteDifferenceBaseClassImageFilter
 * \sa itkMinimumCurvatureImageFilter
 * \sa itkMaximumCurvatureImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT MeanCurvatureImageFilter
  : public BasicFiniteDifferenceBaseClassImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef MeanCurvatureImageFilter                                                 Self;
  typedef BasicFiniteDifferenceBaseClassImageFilter<TInputImage, TOutputImage>     Superclass;
  typedef SmartPointer<Self>                                                       Pointer;
  typedef SmartPointer<const Self>                                                 ConstPointer;
  typedef typename TInputImage::RegionType                                         ImageRegionType;
  typedef typename TInputImage::PixelType                                          PixelType;
  typedef typename TInputImage::IndexType                                          IndexType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MeanCurvatureImageFilter, BasicFiniteDifferenceBaseClassImageFilter);

  /** Print internal ivars */
  void PrintSelf(std::ostream& os, Indent indent) const;

protected:

  MeanCurvatureImageFilter();
  virtual ~MeanCurvatureImageFilter();

  // The main method to implement in derived classes, note, its threaded.
  virtual void ThreadedGenerateData( const ImageRegionType &outputRegionForThread, int);

private:

  MeanCurvatureImageFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMeanCurvatureImageFilter.txx"
#endif

#endif
