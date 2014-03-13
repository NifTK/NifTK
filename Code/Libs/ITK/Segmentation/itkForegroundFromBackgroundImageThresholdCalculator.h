/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkForegroundFromBackgroundImageThresholdCalculator_h
#define __itkForegroundFromBackgroundImageThresholdCalculator_h

#include <itkImage.h>
#include <itkObject.h>
#include <itkMacro.h>
#include <itkImageToHistogramFilter.h>

namespace itk
{

/** \class ForegroundFromBackgroundImageThresholdCalculator
 *  \brief Computes the threshold required to separate an object or
 *  patient in the foreground of an image from a dark background.
 *
 * This calculator computes a threshold to separate an object
 * (i.e. the patient) from the background using the following
 * criteria:
 *
 * 1. Intensities below this threshold have a lower variance than
 * intensities above it.
 *
 * 2. The backgound is dark therefore the threshold will be close to zero.
 *
 * 3. The number of voxels below this threshold is large compared to
 * the number of voxels at intensity levels in the foreground.
 *
 * The threshold is therefore equal to:
 *
 * ( MaxIntensity - t )*( CDF( t ) - Variance( t )/Max_Variance )
 *
 * where:
 *
 * CDF( t ) is the cummulative distribution of intensities below t
 * 
 * Variance( t ) is the variance of intensities below t.
 */

template < class TInputImage >
class ITK_EXPORT ForegroundFromBackgroundImageThresholdCalculator 
  : public Object
{

public:

  /** Standard class typedefs. */
  typedef ForegroundFromBackgroundImageThresholdCalculator Self;
  typedef Object                        Superclass;
  typedef SmartPointer< Self >          Pointer;
  typedef SmartPointer< const Self >    ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ForegroundFromBackgroundImageThresholdCalculator, Object);

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

  /** Type definition for the image histogram. */
  typedef typename itk::Statistics::ImageToHistogramFilter< TInputImage > ImageToHistogramFilterType;

  /** Type definition for the image histogram. */
  typedef typename ImageToHistogramFilterType::HistogramType HistogramType;
  /** Set the input image. */
  itkSetConstObjectMacro(Image, ImageType);

  /** Type definition for the arrays used */
  typedef itk::Array< double > ArrayType;

  /** Compute the threshold to separate the forground from the background. */
  void Compute( void ) throw (ExceptionObject);

  /** Return the threshold intensity value. */
  itkGetConstMacro(Threshold, PixelType);

  /** Set the region over which the values will be computed */
  void SetRegion(const RegionType & region);

  void SetVerbose( bool flag )  { m_FlgVerbose = flag; }
  void SetVerboseOn( void )  { m_FlgVerbose = true;  }
  void SetVerboseOff( void ) { m_FlgVerbose = false; }

protected:

  ForegroundFromBackgroundImageThresholdCalculator();
  virtual ~ForegroundFromBackgroundImageThresholdCalculator() { 
    DeleteArrays(); 
  }

  bool m_FlgVerbose;

  PixelType         m_Threshold;
  ImageConstPointer m_Image;

  RegionType m_Region;
  bool       m_RegionSetByUser;

  void PrintSelf(std::ostream & os, Indent indent) const;

  void WriteHistogramToTextFile( std::string fileName,
                                 HistogramType *histogram );
  
  void WriteDataToTextFile( std::string fileName,
                            itk::Array< double > *x,
                            itk::Array< double > *y );

  void Normalise( itk::Array< double > *y );

  void ComputeVariances( int iStart, int iInc,
                         unsigned int nIntensities, 
                         PixelType firstIntensity );

private:

  ForegroundFromBackgroundImageThresholdCalculator(const Self &); //purposely not implemented
  void operator=(const Self &);                //purposely not implemented

  HistogramType *m_Histogram;

  ArrayType *m_Intensities;
  ArrayType *m_NumberOfPixelsCummulative;
  ArrayType *m_Sums;
  ArrayType *m_Means;
  ArrayType *m_Variances;
  ArrayType *m_IntensityBias;
  ArrayType *m_Thresholds;

  void DeleteArrays( void ) {
    if ( m_Intensities ) delete m_Intensities;
    if ( m_NumberOfPixelsCummulative ) delete m_NumberOfPixelsCummulative;
    if ( m_Sums ) delete m_Sums;
    if ( m_Means ) delete m_Means;
    if ( m_Variances ) delete m_Variances;
    if ( m_IntensityBias ) delete m_IntensityBias;
    if ( m_Thresholds ) delete m_Thresholds;

    m_Intensities              = 0;
    m_NumberOfPixelsCummulative = 0;
    m_Sums                     = 0;
    m_Means                    = 0;
    m_Variances                = 0;
    m_IntensityBias            = 0;
    m_Thresholds               = 0;
  }

  void CreateArrays( unsigned int n ) {
    DeleteArrays();

    m_Intensities              = new ArrayType( n );
    m_NumberOfPixelsCummulative = new ArrayType( n );
    m_Sums                     = new ArrayType( n );
    m_Means                    = new ArrayType( n );
    m_Variances                = new ArrayType( n );
    m_IntensityBias            = new ArrayType( n );
    m_Thresholds               = new ArrayType( n );
  }

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkForegroundFromBackgroundImageThresholdCalculator.txx"
#endif

#endif /* __itkForegroundFromBackgroundImageThresholdCalculator_h */
