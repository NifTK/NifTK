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

#ifndef RegF3dParameters_h
#define RegF3dParameters_h

#include <QString>

#include "NiftyRegCommon.h"


/**
 * \class RegF3dParameters
 * \brief Class to store and initialise the parameters of the affine Aladin registration.
 * \ingroup uk.ac.ucl.cmic.niftyreg
*/


template <class PRECISION_TYPE>
class RegF3dParameters
{  

  public:  

    RegF3dParameters();
    virtual ~RegF3dParameters();

    /// \brief Set the default parameters
    void SetDefaultParameters();

    /// \brief Print the object
    void PrintSelf( std::ostream& os );

    /// Assignment operator
    RegF3dParameters<PRECISION_TYPE> &operator=(const RegF3dParameters<PRECISION_TYPE> &p);

    QString referenceImageName; // -ref
    QString floatingImageName; // -flo

    QString referenceMaskName; // -rmask

    // Initial transformation options:
 
    bool inputControlPointGridFlag;
    QString inputControlPointGridName;// -incpp

    // Output options:
 
    QString outputControlPointGridName; // -cpp
    QString outputWarpedName;		// -res

    // Input image options:

    PRECISION_TYPE referenceThresholdUp;  // -rLwTh
    PRECISION_TYPE referenceThresholdLow; // -rUpTh 

    PRECISION_TYPE floatingThresholdUp;   // -fLwTh
    PRECISION_TYPE floatingThresholdLow;  // -fUpTh

    // Spline options:
 
    PRECISION_TYPE spacing[3];   // -sx, -sy, -sz

    // Objective function options:
 
    unsigned int referenceBinNumber;   // -rbn
    unsigned int floatingBinNumber;    // -fbn

    PRECISION_TYPE bendingEnergyWeight;   // -be

    PRECISION_TYPE linearEnergyWeight0;   // -le 
    PRECISION_TYPE linearEnergyWeight1;   // -le 

    PRECISION_TYPE jacobianLogWeight;     // -jl 

    bool jacobianLogApproximation;       // -noAppJL

    SimilarityType similarity;           // -ssd, -kld 

    // Optimisation options:
 
    bool useConjugate;                   // -noConj
    int maxiterationNumber;              // -maxit
    bool noPyramid;                      // -nopy

    // GPU-related options:

    bool checkMem;   // -mem
    bool useGPU;     // -gpu
    int cardNumber;  // -card

    // Other options:

    InterpolationType interpolation;

    PRECISION_TYPE gradientSmoothingSigma;  // -smoothGrad
    PRECISION_TYPE warpedPaddingValue;      // -pad
    bool verbose;                          // -voff

};

#ifndef ITK_MANUAL_INSTANTIATION
#include "RegF3dParameters.txx"
#endif

#endif // RegF3dParameters_h

