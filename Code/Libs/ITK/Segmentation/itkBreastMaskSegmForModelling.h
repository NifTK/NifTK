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

#ifndef __itkBreastMaskSegmForModelling_h
#define __itkBreastMaskSegmForModelling_h

#include "itkBreastMaskSegmentationFromMRI.h"

namespace itk
{

/** \class BreastMaskSegmForModelling
 * \brief Class to segment the breast mask from MRI for modelling purposes.
 *
 *
 */
template < const unsigned int ImageDimension = 3, class InputPixelType = float >
class ITK_EXPORT BreastMaskSegmForModelling 
  : public BreastMaskSegmentationFromMRI< ImageDimension, InputPixelType >
{
public:
    
  typedef BreastMaskSegmForModelling                                      Self;
  typedef BreastMaskSegmentationFromMRI< ImageDimension, InputPixelType > Superclass;
  typedef SmartPointer<Self>                                              Pointer;
  typedef SmartPointer<const Self>                                        ConstPointer;
  
  itkNewMacro(Self); 
  itkTypeMacro( BreastMaskSegmForModelling, BreastMaskSegmentationFromMRI );

  typedef typename Superclass::InternalImageType InternalImageType;
  typedef typename Superclass::PointSetType      PointSetType;
  typedef typename Superclass::RealType          RealType;
  typedef typename Superclass::IteratorType      IteratorType;

  /// Execute the segmentation 
  virtual void Execute( void );


protected:

  /// Constructor
  BreastMaskSegmForModelling();

  /// Destructor
  ~BreastMaskSegmForModelling();

  /// Mask the pectoral muscle using a B-Spline surface
  void MaskThePectoralMuscleOnly( RealType rYHeightOffset, 
				  typename PointSetType::Pointer &pecPointSet );

private:

  BreastMaskSegmForModelling(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBreastMaskSegmForModelling.txx"
#endif

#endif




