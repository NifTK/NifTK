/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMammographicTumourDistribution_h
#define itkMammographicTumourDistribution_h

#include <itkMammogramAnalysis.h>


/*!
 * \file niftkMammographicTumourDistribution.cxx
 * \page niftkMammographicTumourDistribution
 * \section niftkMammographicTumourDistributionSummary Calculates the distribution of tumour locations from a set of diagnostic images
 *
 * \section niftkMammographicTumourDistributionCaveats Caveats
 * \li None
 */


namespace itk
{


// -----------------------------------------------------------------------------------
// Class to store the data for control and diagnostic images of a patient
// -----------------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension=2>
class ITK_EXPORT MammographicTumourDistribution
  : public MammogramAnalysis< InputPixelType, InputDimension >
{
public:

  typedef MammographicTumourDistribution                         Self;
  typedef MammogramAnalysis< InputPixelType, InputDimension > Superclass;
  typedef SmartPointer<Self>                                  Pointer;
  typedef SmartPointer<const Self>                            ConstPointer;

  itkNewMacro(Self);
  itkTypeMacro(MammographicTumourDistribution, Object);

  itkStaticConstMacro( ParametricDimension, unsigned int, 2 );
  itkStaticConstMacro( DataDimension, unsigned int, 1 );

  typedef typename Superclass::ImageType ImageType;
  typedef typename Superclass::OutputImageType OutputImageType;
  typedef typename Superclass::LabelImageType LabelImageType;
  typedef typename Superclass::ImageTypeUCHAR ImageTypeUCHAR;

  typedef typename Superclass::ReaderType ReaderType;
  typedef typename Superclass::PolygonType PolygonType;

  typedef typename Superclass::LeftOrRightSideCalculatorType LeftOrRightSideCalculatorType;

  void SetOutputCSV( std::ofstream *foutOutputCSV ) { m_foutOutputCSV = foutOutputCSV; };

  void WriteDataToCSVFile( std::ofstream *foutCSV );

  void Compute();

protected:

  /// Constructor
  MammographicTumourDistribution();

  /// Destructor
  virtual ~MammographicTumourDistribution();

  /// Add a point to a polygon of the transformed tumour region
  void AddPointToTumourPolygon( typename PolygonType::Pointer &polygon, int i, int j);

  /// Draw the tranformed tumour region on the reference image
  typename ImageType::Pointer DrawTumourOnReferenceImage();

  /// Register the images
  void RunRegistration();

  /// The output CSV file stream
  std::ofstream *m_foutOutputCSV;

private:

  MammographicTumourDistribution(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMammographicTumourDistribution.txx"
#endif

#endif


