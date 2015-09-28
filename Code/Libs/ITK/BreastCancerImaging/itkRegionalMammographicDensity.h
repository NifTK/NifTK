/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkRegionalMammographicDensity_h
#define itkRegionalMammographicDensity_h

#include <itkMammogramAnalysis.h>

/*!
 * \file niftkRegionalMammographicDensity.cxx
 * \page niftkRegionalMammographicDensity
 * \section niftkRegionalMammographicDensitySummary Calculates the density within regions on interest across a mammogram.
 *
 * \section niftkRegionalMammographicDensityCaveats Caveats
 * \li None
 */


namespace itk
{


// -----------------------------------------------------------------------------------
// Class to store the data for diagnostic and pre-diagnostic images of a patient
// -----------------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension=2>
class ITK_EXPORT RegionalMammographicDensity
  : public MammogramAnalysis< InputPixelType, InputDimension >
{
public:

  typedef RegionalMammographicDensity                         Self;
  typedef MammogramAnalysis< InputPixelType, InputDimension > Superclass;
  typedef SmartPointer<Self>                                  Pointer;
  typedef SmartPointer<const Self>                            ConstPointer;

  itkNewMacro(Self);
  itkTypeMacro(RegionalMammographicDensity, Object);

  itkStaticConstMacro( ParametricDimension, unsigned int, 2 );
  itkStaticConstMacro( DataDimension, unsigned int, 1 );

  typedef typename Superclass::ImageType ImageType;
  typedef typename Superclass::LabelImageType LabelImageType;

  typedef typename Superclass::ReaderType ReaderType;

  typedef typename Superclass::LeftOrRightSideCalculatorType LeftOrRightSideCalculatorType;

  bool Compute( void );

protected:

  /// Constructor
  RegionalMammographicDensity();

  /// Destructor
  virtual ~RegionalMammographicDensity();

  /// Register the images
  void RunRegistration();

private:

  RegionalMammographicDensity(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRegionalMammographicDensity.txx"
#endif

#endif


