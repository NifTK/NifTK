/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author:  $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef NiftyRegParameters_h
#define NiftyRegParameters_h

#include "RegAladinParameters.h"
#include "RegF3dParameters.h"

#include <QString>


/**
 * \class NiftyRegParameters
 * \brief Class to store and initialise the parameters of the affine Aladin registration.
 * \ingroup uk.ac.ucl.cmic.niftyreg
*/


template <class PRECISION_TYPE>
class NiftyRegParameters
{  

  public:  

    NiftyRegParameters();
    virtual ~NiftyRegParameters();

    /// \brief Set the default parameters
    void SetDefaultParameters();

    /// \brief Print the object
    void PrintSelf( std::ostream& os );

    /// Assignment operator
    NiftyRegParameters<PRECISION_TYPE> &operator=(const NiftyRegParameters<PRECISION_TYPE> &p);


    /// \brief The number of multi-resolution levels
    int m_LevelNumber;
    /// \brief The number of (coarse to fine) multi-resolution levels to use 
    int m_Level2Perform;    

    // Smooth the target image using the specified sigma (mm) 
    float m_TargetSigmaValue;
    // Smooth the source image using the specified sigma (mm)
    float m_SourceSigmaValue;

    /// Flag indicating whether to do an initial rigid registration
    bool m_FlagDoInitialRigidReg;
    /// Flag indicating whether to do the non-rigid registration
    bool m_FlagDoNonRigidReg;

    /// The filename of the initial affine transformation
    QString m_InputAffineName;  // -inaff
    /// Flag indicating whether an initial affine transformation is specified
    bool m_FlagInputAffine;
    /// Flag indicating whether the initial affine transformation is FLIRT
    bool m_FlagFlirtAffine;   // -affFlirt


    /// The 'reg_aladin' parameters
    RegAladinParameters m_AladinParameters;

    /// The 'reg_f3d' parameters
    RegF3dParameters<PRECISION_TYPE> m_F3dParameters;

};

#ifndef ITK_MANUAL_INSTANTIATION
#include "NiftyRegParameters.txx"
#endif

#endif // NiftyRegParameters_h

