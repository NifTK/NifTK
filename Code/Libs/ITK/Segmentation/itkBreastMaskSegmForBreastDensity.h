/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author: $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkBreastMaskSegmForBreastDensity_h
#define __itkBreastMaskSegmForBreastDensity_h

#include "BreastMaskSegmentationFromMRI.h"

namespace itk
{

/** \class BreastMaskSegmForBreastDensity
 * \brief Class to segment the breast mask from MRI for modelling purposes.
 *
 *
 */
template < const unsigned int ImageDimension = 3, class PixelType = float >
class ITK_EXPORT BreastMaskSegmForBreastDensity 
  : public BreastMaskSegmentationFromMRI< ImageDimension, PixelType >
{
public:
    
  typedef BreastMaskSegmForBreastDensity    Self;
  typedef BreastMaskSegmentationFromMRI Superclass;
  typedef SmartPointer<Self>            Pointer;
  typedef SmartPointer<const Self>      ConstPointer;
  
  itkTypeMacro( BreastMaskSegmForBreastDensity, BreastMaskSegmentationFromMRI );


  /// Execute the segmentation 
  virtual void Execute( void );


protected:

  /// Constructor
  BreastMaskSegmForBreastDensity();

  /// Destructor
  ~BreastMaskSegmForBreastDensity();

  /// Mask the pectoral muscle using a B-Spline surface
  virtual void MaskThePectoralMuscleAndLateralChestSkinSurface( void );

private:

  BreastMaskSegmForBreastDensity(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBreastMaskSegmForBreastDensity.txx"
#endif

#endif




