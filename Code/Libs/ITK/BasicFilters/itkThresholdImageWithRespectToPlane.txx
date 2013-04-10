/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkThresholdImageWithRespectToPlane_txx
#define __itkThresholdImageWithRespectToPlane_txx

#include "itkThresholdImageWithRespectToPlane.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageFileWriter.h"
#include "itkProgressReporter.h"
#include "ConversionUtils.h"

#include "itkLogHelper.h"


namespace itk
{
/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
ThresholdImageWithRespectToPlane<TInputImage,TOutputImage>
::ThresholdImageWithRespectToPlane()
{
  m_ThresholdValue = (OutputImagePixelType)(0.0);
  
  m_PlanePosition.Fill( 0.0 );

  m_PlaneNormal[0] = 0.0;
  m_PlaneNormal[1] = 0.0;
  m_PlaneNormal[2] = 1.0;
}


/* -----------------------------------------------------------------------
   SetPlanePosition()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
ThresholdImageWithRespectToPlane<TInputImage,TOutputImage>
::SetPlanePosition( double px, double py, double pz )
{
  m_PlanePosition[0] = px;
  m_PlanePosition[1] = py;
  m_PlanePosition[2] = pz;

  this->Modified();
}


/* -----------------------------------------------------------------------
   SetPlaneNormal()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
ThresholdImageWithRespectToPlane<TInputImage,TOutputImage>
::SetPlaneNormal( double px, double py, double pz )
{
  m_PlaneNormal[0] = px;
  m_PlaneNormal[1] = py;
  m_PlaneNormal[2] = pz;

  this->Modified();
}


/* -----------------------------------------------------------------------
   BeforeThreadedGenerateData()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
ThresholdImageWithRespectToPlane<TInputImage,TOutputImage>
::BeforeThreadedGenerateData(void)
{

}


/* -----------------------------------------------------------------------
   ThreadedGenerateData(const OutputImageRegionType&, int)
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
ThresholdImageWithRespectToPlane<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       int threadId)
{
  double distanceToPlane;
  OutputImagePointType point;

  // Equation of plane with normal (nx,ny,nz) through point (px,py,pz) is
  //    nx.x + ny.y + nz.z + d = 0
  // where:
  //    d = - nx.px - ny.py - nz.pz
  
  double d = 
    - m_PlaneNormal[0]*m_PlanePosition[0]
    - m_PlaneNormal[1]*m_PlanePosition[1]
    - m_PlaneNormal[2]*m_PlanePosition[2];

  // The magnitude of the plane normal

  double mag = vcl_sqrt(   m_PlaneNormal[0]*m_PlaneNormal[0]
                         + m_PlaneNormal[1]*m_PlaneNormal[1]
                         + m_PlaneNormal[2]*m_PlaneNormal[2]);

  // Obtain image pointers

  InputImageConstPointer inImage  = this->GetInput();
  OutputImagePointer     outImage = this->GetOutput();

  // Support progress methods/callbacks

  ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());

  // Iterate over pixels in the input image

  ImageRegionConstIterator< InputImageType > inputIterator
    = ImageRegionConstIterator< InputImageType >( inImage, outputRegionForThread );

  ImageRegionIterator< OutputImageType > outputIterator
    = ImageRegionIterator< OutputImageType >( outImage, outputRegionForThread );

  for ( ; ! outputIterator.IsAtEnd(); ++outputIterator, ++inputIterator) {

    outImage->TransformIndexToPhysicalPoint( outputIterator.GetIndex(), point  );

    // Compute the signed distance of this point to the plane which is
    // positive if the point is on the same side of the plane as the normal
    // vector and negative if it is on the opposite side.

    distanceToPlane = (   m_PlaneNormal[0]*point[0]
                        + m_PlaneNormal[1]*point[1]
                        + m_PlaneNormal[2]*point[2]
                        + d ) / mag;

    if ( distanceToPlane >= 0. ) 
      outputIterator.Set( inputIterator.Get() );

    else 
      outputIterator.Set( m_ThresholdValue );
  }
}


/* -----------------------------------------------------------------------
   AfterThreadedGenerateData()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
ThresholdImageWithRespectToPlane<TInputImage,TOutputImage>
::AfterThreadedGenerateData(void)
{

}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
ThresholdImageWithRespectToPlane<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << "Threshold value: " << m_ThresholdValue << std::endl;
  os << "Plane position: " << m_PlanePosition << std::endl;
  os << "Plane normal: " << m_PlaneNormal << std::endl;
}

} // end namespace itk

#endif
