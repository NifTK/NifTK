/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
    QString referenceImagePath; // -ref

    QString floatingImageName; // -flo
    QString floatingImagePath; // -flo

    QString referenceMaskName; // -rmask
    QString referenceMaskPath; // -rmask

    // Initial transformation options:
 
    bool inputControlPointGridFlag;
    QString inputControlPointGridName;// -incpp

    // Output options:
 
    QString outputControlPointGridName; // -cpp

    QString outputWarpedName;		// -res
    QString outputWarpedPath;		// -res

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

    InterpolationType interpolation;        // -interp

    PRECISION_TYPE gradientSmoothingSigma;  // -smoothGrad
    PRECISION_TYPE warpedPaddingValue;      // -pad
    bool verbose;                           // -voff

};

#ifndef ITK_MANUAL_INSTANTIATION
#include "RegF3dParameters.txx"
#endif

#endif // RegF3dParameters_h

