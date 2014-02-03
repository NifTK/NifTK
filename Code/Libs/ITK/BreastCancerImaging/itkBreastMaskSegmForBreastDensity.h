/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkBreastMaskSegmForBreastDensity_h
#define itkBreastMaskSegmForBreastDensity_h

#include "itkBreastMaskSegmentationFromMRI.h"

namespace itk
{

/** \class BreastMaskSegmForBreastDensity
 * \brief Class to segment the breast mask from MRI for modelling purposes.
 *
 *
 */
template < const unsigned int ImageDimension = 3, class InputPixelType = float >
class ITK_EXPORT BreastMaskSegmForBreastDensity 
  : public BreastMaskSegmentationFromMRI< ImageDimension, InputPixelType >
{
public:
    
  typedef BreastMaskSegmForBreastDensity                                   Self;
  typedef BreastMaskSegmentationFromMRI< ImageDimension, InputPixelType >  Superclass;
  typedef SmartPointer<Self>                                               Pointer;
  typedef SmartPointer<const Self>                                         ConstPointer;

  itkNewMacro(Self); 
  itkTypeMacro( BreastMaskSegmForBreastDensity, BreastMaskSegmentationFromMRI );

  typedef typename Superclass::InternalImageType               InternalImageType;
  typedef typename Superclass::PointSetType                    PointSetType;
  typedef typename Superclass::RealType                        RealType;
  typedef typename Superclass::IteratorType                    IteratorType;
  typedef typename Superclass::ConnectedSurfaceVoxelFilterType ConnectedSurfaceVoxelFilterType;
  typedef typename Superclass::VectorType                      VectorType;
  typedef typename Superclass::LineIteratorType                LineIteratorType;

  /// Execute the segmentation 
  virtual void Execute( void );


protected:

  /// Constructor
  BreastMaskSegmForBreastDensity();

  /// Destructor
  ~BreastMaskSegmForBreastDensity();

  /// Mask the pectoral muscle using a B-Spline surface
  void MaskThePectoralMuscleAndLateralChestSkinSurface( RealType rYHeightOffset, 
							typename PointSetType::Pointer &pecPointSet,
							unsigned long &iPointPec );

private:

  BreastMaskSegmForBreastDensity(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBreastMaskSegmForBreastDensity.txx"
#endif

#endif




