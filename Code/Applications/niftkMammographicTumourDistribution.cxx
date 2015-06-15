/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

#include <itkLogHelper.h>

#include <niftkFileHelper.h>
#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>

#include <itkDOMReader.h>


#include <itkMammographicTumourDistribution.h>

#include <niftkMammographicTumourDistributionCLP.h>

namespace fs = boost::filesystem;



// -----------------------------------------------------------------------------------
// Read the pectoralis XML file
// -----------------------------------------------------------------------------------

void ReadXMLFile( std::string fileInputXML,
                 itk::DOMNode::Pointer &dom )
{
  itk::DOMNodeXMLReader::Pointer domReader = itk::DOMNodeXMLReader::New();

  domReader->SetFileName( fileInputXML );

  try {
    std::cout << "Reading XML file: " << fileInputXML << std::endl;
    domReader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Failed to read XML file: " << err << std::endl;
    exit( EXIT_FAILURE );
  }

  dom = domReader->GetOutput();
}


// -----------------------------------------------------------------------------------
// Get the text value of a node
// -----------------------------------------------------------------------------------

std::string GetText( itk::DOMNode::ChildrenListType::iterator &itNode, std::string id )
{
  std::string str;

  itk::DOMNode::Pointer child = (*itNode)->GetChildByID( id );

  if ( ! child )
  {
    std::cerr << "ERROR: Could not find child node: " << id << std::endl;
    exit( EXIT_FAILURE );
  }

  itk::DOMTextNode::Pointer textNode = child->GetTextChild();

  if ( ! textNode )
  {
    std::cerr << "ERROR: Could not find text node: " << id << std::endl;
    exit( EXIT_FAILURE );
  }

  str = textNode->GetText();


  if ( ! str.length() )
  {
    std::cerr << "ERROR: Empty text value for: " << id << std::endl;
    exit( EXIT_FAILURE );
  }

  return str;
}


// --------------------------------------------------------------------------
// WriteHeaderToCSVFile()
// --------------------------------------------------------------------------

void WriteToCSVFile( std::ofstream *foutOutputCSV )
{
  //                                   123456789012345678901234567890

  *foutOutputCSV
    << std::right << std::setw(10) << "Diagnostic SNO"  << ", "
    << std::right << std::setw(17) << "Diagnostic ID"   << ", "
    << std::right << std::setw(60) << "Diagnostic file" << ", "

    << std::right << std::setw(10) << "Reference SNO"  << ", "
    << std::right << std::setw(17) << "Reference ID"   << ", "
    << std::right << std::setw(60) << "Reference file" << ", "

    << std::right << std::setw( 9) << "Tumour ID"         << ", "
    << std::right << std::setw(17) << "Tumour image ID"   << ", "

    << std::right << std::setw(17) << "Tumour diagnostic center (x)" << ", "
    << std::right << std::setw(17) << "Tumour diagnostic center (y)" << ", "

    << std::right << std::setw(17) << "Tumour reference center (x)" << ", "
    << std::right << std::setw(17) << "Tumour reference center (y)" << ", "

    << std::endl;
};


// -----------------------------------------------------------------------------------
/** \brief Calculates the distribution of tumours in a set of diagnostic mammograms */
// -----------------------------------------------------------------------------------

int main(int argc, char** argv)
{
  int result = EXIT_SUCCESS;

  std::string dirOutputFullPath;

  std::ofstream *foutOutputCSV = 0;

  const unsigned int   InputDimension = 2;
  typedef float InputPixelType;

  typedef itk::DOMNode::IdentifierType NodeIdentifierType;

  NodeIdentifierType iNode = 0, iNodeRecord = 0;

  itk::DOMNode::SizeType nChildren;

  itk::DOMNode::Pointer node;
  itk::DOMNode::Pointer nodeRecord;

  itk::DOMTextNode::Pointer textNode;

  typedef itk::MammographicTumourDistribution<InputPixelType, InputDimension> TumourDistribType;

  std::list< TumourDistribType::Pointer > listOfPatients;
  std::list< TumourDistribType::Pointer >::iterator itPatient;


  // To pass around command line args
  PARSE_ARGS;

  std::cout << std::endl
            << "Input directory: " << dirInput << std::endl
            << "Input image info XML file: " << fileImageXML << std::endl
            << "Input tumour mask info XML file: " << fileTumourXML << std::endl
            << "Input breast edge points XML file: " << fileBreastEdgeXML << std::endl
            << "Input pectoralis XML file: " << filePectoralisLinePointXML << std::endl
            << "Non-rigid registration? " << flgRegisterNonRigid << std::endl
            << "Ouput directory: " << dirOutput << std::endl
            << "Output CSV file: " << fileOutputCSV << std::endl
            << "Verbose? " << flgVerbose << std::endl
            << "Overwrite? " << flgOverwrite << std::endl
            << "Debug? " << flgDebug << std::endl
            << std::endl;

  // Validate command line args
  if ( ( filePectoralisLinePointXML.length() == 0 ) ||
       ( fileImageXML.length() == 0 ) ||
       ( fileTumourXML.length() == 0 ) ||
       ( fileBreastEdgeXML.length() == 0 ) )
  {
    std::cerr << "ERROR: Input XML file(s) missing" << std::endl;
    return EXIT_FAILURE;
  }


  // Open the output CSV density measurements file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputCSV.length() != 0 ) {
    foutOutputCSV
      = new std::ofstream( fileOutputCSV.c_str(), std::ios::binary );

    if ((! foutOutputCSV) || foutOutputCSV->bad() || foutOutputCSV->fail()) {
      std::cerr << "ERROR: Could not open CSV output file: " << fileOutputCSV << std::endl;
      return EXIT_FAILURE;
    }
  }

  WriteToCSVFile( foutOutputCSV );


  // Read the XML files
  // ~~~~~~~~~~~~~~~~~~

  itk::DOMNode::Pointer domPectoral, domImage, domTumour, domBreastEdge;

  ReadXMLFile( filePectoralisLinePointXML, domPectoral );
  ReadXMLFile( fileImageXML, domImage );
  ReadXMLFile( fileTumourXML, domTumour );
  ReadXMLFile( fileBreastEdgeXML, domBreastEdge );


  // Iterate through the images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  TumourDistribType::Pointer reference = 0;

  for ( iNode = 0;
        iNode < static_cast<NodeIdentifierType>( domImage->GetNumberOfChildren() );
        iNode++ )
  {
    std::string strBreastArea;

    std::string strCase;
    std::string strComments;
    std::string strFilename;
    std::string strImageID;
    std::string strImageOrder;
    std::string strReferenceImage;
    std::string strSetNumber;
    std::string strThreshold;

    std::string strPatientID;

    node = domImage->GetChild( iNode );

    std::cout << std::endl << "Image: " << iNode << std::endl;

    if ( flgDebug )
      std::cout << "  tag: " << node->GetName()
                << " path: " << node->GetPath()
                << " num: "  << node->GetNumberOfChildren() << std::endl;


    // Get the records for this image
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for ( iNodeRecord = 0;
          iNodeRecord < static_cast<NodeIdentifierType>( node->GetNumberOfChildren() );
          iNodeRecord++ )
    {
      nodeRecord = node->GetChild( iNodeRecord );

      textNode = nodeRecord->GetTextChild();

      if ( flgDebug )
        std::cout << "    tag: " << std::left << std::setw(18) << nodeRecord->GetName()
                  << " path: " << std::left << std::setw(12) << nodeRecord->GetPath()
                  << " num: " << std::left << std::setw(12) << nodeRecord->GetNumberOfChildren()
                  << " value: " << std::left << textNode->GetText()
                  << std::endl;

      if ( nodeRecord->GetName() == std::string( "id" ) )
      {
        strImageID    = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "file_name" ) )
      {
        strFilename   = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "DensityThreshold" ) )
      {
        strThreshold  = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "BreastArea_raster" ) )
      {
        strBreastArea = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "image_order" ) )
      {
        strImageOrder = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "sno" ) )
      {
        strSetNumber = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "comments" ) )
      {
        strComments = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "ref" ) )
      {
        strReferenceImage = textNode->GetText();
      }
    }

    strPatientID = fs::path( strFilename ).branch_path().string();

    std::cout << "    patient id:           " << strPatientID            << std::endl
              << "    image id:             " << strImageID              << std::endl
              << "    file_name:            " << strFilename             << std::endl
              << "    threshold:            " << strThreshold            << std::endl
              << "    breast area:          " << strBreastArea           << std::endl
              << "    order:                " << strImageOrder           << std::endl
              << "    comments:             " << strComments             << std::endl
              << "    setnum:               " << strSetNumber            << std::endl
              << "    type:                 " << strReferenceImage       << std::endl;

    // Find a patient with this ID

    bool flgFound = false;

    TumourDistribType::Pointer patient = 0;

    for ( itPatient = listOfPatients.begin();
          itPatient != listOfPatients.end();
          itPatient++ )
    {
      if ( (*itPatient)->GetPatientID() == strPatientID )
      {
        patient = (*itPatient);
        flgFound = true;
        break;
      };
    }

    if ( ! flgFound )
    {
      patient = TumourDistribType::New();

      patient->SetPatientID( strPatientID );
      patient->SetInputDirectory( dirInput );
      patient->SetOutputDirectory( dirOutput );

      patient->SetOutputCSV( foutOutputCSV );

      if ( flgOverwrite ) patient->SetOverwriteOn();

      if ( flgRegisterNonRigid ) patient->SetRegisterNonRigidOn();

      if ( flgVerbose )   patient->SetVerboseOn();
      if ( flgDebug )     patient->SetDebugOn();
    }

    if  ( strReferenceImage == std::string( "reference_image" ) )
    {
      patient->SetIDControlImage(   strImageID );
      patient->SetFileControl(      strFilename );
      patient->SetThresholdControl( atoi( strThreshold.c_str() ) );
      patient->SetBreastAreaControl( atoi( strBreastArea.c_str() ) );
      patient->SetSetNumberControl( atoi(strSetNumber.c_str() ) );

      if ( reference )
      {
        std::cerr << "ERROR: Reference image is already defined" << std::endl;
        exit( EXIT_FAILURE );
      }
      else
      {
        reference = patient;
        listOfPatients.push_back( patient );
      }
    }

    else
    {
      patient->SetIDDiagnosticImage(   strImageID );
      patient->SetFileDiagnostic(      strFilename );
      patient->SetThresholdDiagnostic( atoi( strThreshold.c_str() ) );
      patient->SetBreastAreaDiagnostic( atoi( strBreastArea.c_str() ) );
      patient->SetSetNumberDiagnostic( atoi( strSetNumber.c_str() ) );

      if ( ! flgFound )
      {
        listOfPatients.push_back( patient );
      }
    }
  }

  if ( ! reference )
  {
    std::cerr << "ERROR: No reference image found" << std::endl;
    exit( EXIT_FAILURE );
  }
  else
  {
    for ( itPatient = listOfPatients.begin();
          itPatient != listOfPatients.end();
          itPatient++ )
    {
      (*itPatient)->SetIDControlImage(    reference->GetIDControlImage() );
      (*itPatient)->SetFileControl(       reference->GetFileControl() );
      (*itPatient)->SetThresholdControl(  reference->GetThresholdControl() );
      (*itPatient)->SetSetNumberControl(  reference->GetSetNumberControl() );
      (*itPatient)->SetBreastAreaControl( reference->GetBreastAreaControl() );
    }
  }


  // Iterate through the tumours
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for ( iNode = 0;
        iNode < static_cast<NodeIdentifierType>( domTumour->GetNumberOfChildren() );
        iNode++ )
  {
    std::string strTumourDiameter;
    std::string strTumourID;
    std::string strTumourImageID;

    std::string strTumourLeft;
    std::string strTumourRight;
    std::string strTumourTop;
    std::string strTumourBottom;

    node = domTumour->GetChild( iNode );

    if ( flgDebug )
      std::cout << std::endl << "Tumour: " << iNode << std::endl
                << "  tag: " << node->GetName()
                << " path: " << node->GetPath()
                << " num: "  << node->GetNumberOfChildren() << std::endl;


    // Get the records for this tumour

    for ( iNodeRecord = 0;
          iNodeRecord < static_cast<NodeIdentifierType>( node->GetNumberOfChildren() );
          iNodeRecord++ )
    {
      nodeRecord = node->GetChild( iNodeRecord );

      textNode = nodeRecord->GetTextChild();

      if ( flgDebug )
        std::cout << "    tag: " << std::left << std::setw(18) << nodeRecord->GetName()
                  << " path: " << std::left << std::setw(12) << nodeRecord->GetPath()
                  << " num: " << std::left << std::setw(12) << nodeRecord->GetNumberOfChildren()
                  << " value: " << std::left << textNode->GetText()
                  << std::endl;

      if ( nodeRecord->GetName() == std::string( "ID" ) )
      {
        strTumourID = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "left" ) )
      {
        strTumourLeft = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "right" ) )
      {
        strTumourRight  = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "top" ) )
      {
        strTumourTop = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "bottom" ) )
      {
        strTumourBottom = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "Image_ID" ) )
      {
        strTumourImageID = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "tumor_diameter" ) )
      {
        strTumourDiameter = textNode->GetText();
      }
    }

    std::cout << "    tumour id: " << std::left << std::setw(6) << strTumourID
              << "    left: "      << std::left << std::setw(6) << strTumourLeft
              << "    right: "     << std::left << std::setw(6) << strTumourRight
              << "    top: "       << std::left << std::setw(6) << strTumourTop
              << "    bottom: "    << std::left << std::setw(6) << strTumourBottom
              << "    image id: "  << std::left << std::setw(6) << strTumourImageID
              << "    diameter: "  << std::left << std::setw(6) << strTumourDiameter
              << std::endl << std::endl;

    // Find a patient with this image ID

    TumourDistribType::Pointer patient = 0;

    if ( reference->GetIDControlImage() == strTumourImageID )
    {
      std::cout << "IDReferenceImage" << reference->GetIDControlImage()
                << ", tumour ID " <<  strTumourImageID << std::endl;

      patient = reference;
    }

    else
    {
      for ( itPatient = listOfPatients.begin();
            itPatient != listOfPatients.end();
            itPatient++ )
      {
        if ( (*itPatient)->GetIDDiagnosticImage() == strTumourImageID )
        {
          std::cout << "IDDiagnosticImage" << (*itPatient)->GetIDDiagnosticImage()
                    << ", tumour ID " <<  strTumourImageID << std::endl;

          patient = (*itPatient);
          break;
        };
      }
    }

    if ( ! patient )
    {
      std::cerr << "WARNING: Failed to find a patient corresponding to image id: "
                << strTumourImageID << std::endl;
    }
    else
    {
      patient->SetTumourID(      strTumourID );
      patient->SetTumourImageID( strTumourImageID );

      patient->SetTumourLeft(   atoi( strTumourLeft.c_str() ) );
      patient->SetTumourRight(  atoi( strTumourRight.c_str() ) );
      patient->SetTumourTop(    atoi( strTumourTop.c_str() ) );
      patient->SetTumourBottom( atoi( strTumourBottom.c_str() ) );
      patient->SetTumourDiameter( atof( strTumourDiameter.c_str() ) );
    }
  }


  // Iterate through the breast edge points
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::cout << std::endl;

  for ( iNode = 0;
        iNode < static_cast<NodeIdentifierType>( domBreastEdge->GetNumberOfChildren() );
        iNode++ )
  {
    std::string strBreastEdgeID;
    std::string strBreastEdgeImageID;
    std::string strXCoord;
    std::string strYCoord;

    node = domBreastEdge->GetChild( iNode );

    std::cout << "BreastEdge: " << std::left << std::setw(6) << iNode;

    if ( flgDebug )
      std::cout << "  tag: " << node->GetName()
                << " path: " << node->GetPath()
                << " num: "  << node->GetNumberOfChildren() << std::endl;


    // Get the records for this image
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for ( iNodeRecord = 0;
          iNodeRecord < static_cast<NodeIdentifierType>( node->GetNumberOfChildren() );
          iNodeRecord++ )
    {
      nodeRecord = node->GetChild( iNodeRecord );

      textNode = nodeRecord->GetTextChild();

      if ( flgDebug )
        std::cout << "    tag: " << std::left << std::setw(18) << nodeRecord->GetName()
                  << " path: " << std::left << std::setw(12) << nodeRecord->GetPath()
                  << " num: " << std::left << std::setw(12) << nodeRecord->GetNumberOfChildren()
                  << " value: " << std::left << textNode->GetText()
                  << std::endl;

      if ( nodeRecord->GetName() == std::string( "ID" ) )
      {
        strBreastEdgeID = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "x" ) )
      {
        strXCoord = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "y" ) )
      {
        strYCoord = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "Image_ID" ) )
      {
        strBreastEdgeImageID = textNode->GetText();
      }
    }

    std::cout << " breast edge id: "       << std::left << std::setw(6) << strBreastEdgeID
              << " x coord: "              << std::left << std::setw(6) << strXCoord
              << " y coord: "              << std::left << std::setw(6) << strYCoord
              << " breast edge image id: " << std::left << std::setw(6) << strBreastEdgeImageID
              << std::endl;

    // Find a patient with this image ID

    TumourDistribType::Pointer patient = 0;

    if ( reference->GetIDControlImage() == strBreastEdgeImageID )
    {
      std::cout << "IDReferenceImage" << reference->GetIDControlImage()
                << ", BreastEdgeImageID " <<  strBreastEdgeImageID << std::endl;

      patient = reference;
    }

    else
    {
      for ( itPatient = listOfPatients.begin();
            itPatient != listOfPatients.end();
            itPatient++ )
      {
        if ( (*itPatient)->GetIDDiagnosticImage() == strBreastEdgeImageID )
        {
          patient = (*itPatient);
          break;
        }
      }
    }

    if ( ! patient )
    {
      std::cerr << "WARNING: Failed to find a patient corresponding to image id: "
                << strBreastEdgeImageID << std::endl;
    }
    else
    {
      patient->PushBackBreastEdgeCoord( strBreastEdgeImageID,
                                        atoi( strBreastEdgeID.c_str() ),
                                        atoi( strXCoord.c_str() ),
                                        atoi( strYCoord.c_str() ) );
    }
  }


  // Iterate through the pectoral points
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for ( iNode = 0;
        iNode < static_cast<NodeIdentifierType>( domPectoral->GetNumberOfChildren() );
        iNode++ )
  {
    std::string strPectoralID;
    std::string strPectoralImageID;
    std::string strXCoord;
    std::string strYCoord;

    node = domPectoral->GetChild( iNode );

    std::cout << "Pectoral:   " << std::left << std::setw(6) << iNode;

    if ( flgDebug )
      std::cout << "    tag: " << node->GetName()
                << " path: " << node->GetPath()
                << " num: "  << node->GetNumberOfChildren() << std::endl;


    // Get the records for this image
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for ( iNodeRecord = 0;
          iNodeRecord < static_cast<NodeIdentifierType>( node->GetNumberOfChildren() );
          iNodeRecord++ )
    {
      nodeRecord = node->GetChild( iNodeRecord );

      textNode = nodeRecord->GetTextChild();

      if ( flgDebug )
        std::cout << " tag: " << std::left << std::setw(18) << nodeRecord->GetName()
                  << " path: " << std::left << std::setw(12) << nodeRecord->GetPath()
                  << " num: " << std::left << std::setw(12) << nodeRecord->GetNumberOfChildren()
                  << " value: " << std::left << textNode->GetText()
                  << std::endl;

      if ( nodeRecord->GetName() == std::string( "ID" ) )
      {
        strPectoralID    = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "x" ) )
      {
        strXCoord   = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "y" ) )
      {
        strYCoord  = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "Image_ID" ) )
      {
        strPectoralImageID = textNode->GetText();
      }
    }

    std::cout << " pectoral id: "          << std::left << std::setw(6) << strPectoralID
              << " x coord: "              << std::left << std::setw(6) << strXCoord
              << " y coord: "              << std::left << std::setw(6) << strYCoord
              << " breast edge image id: " << std::left << std::setw(6) << strPectoralImageID
              << std::endl;

    // Find a patient with this image ID

    TumourDistribType::Pointer patient = 0;

    if ( reference->GetIDControlImage() == strPectoralImageID )
    {
      std::cout << "IDReferenceImage " << reference->GetIDControlImage()
                 << std::endl;

      patient = reference;
    }

    else
    {
      for ( itPatient = listOfPatients.begin();
            itPatient != listOfPatients.end();
            itPatient++ )
      {
        if ( (*itPatient)->GetIDDiagnosticImage() == strPectoralImageID )
        {
          patient = (*itPatient);
          break;
        }
      }
    }

    if ( ! patient )
    {
      std::cerr << "WARNING: Failed to find a patient corresponding to image id: "
                << strPectoralImageID << std::endl;
    }
    else
    {
      patient->PushBackPectoralCoord( strPectoralImageID,
                                      atoi( strPectoralID.c_str() ),
                                      atoi( strXCoord.c_str() ),
                                      atoi( strYCoord.c_str() ) );
    }
  }


  // Iterate through the patients analysing each image pair
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  float progress = 0.;
  float iFile = 0.;
  float nFiles = listOfPatients.size();

  for ( itPatient = listOfPatients.begin();
        itPatient != listOfPatients.end();
        itPatient++, iFile += 1. )
  {
    progress = iFile/nFiles;
    std::cout << "<filter-progress>" << std::endl
	      << progress << std::endl
	      << "</filter-progress>" << std::endl;

    (*itPatient)->LoadImages();

    progress = (iFile + 0.25)/nFiles;
    std::cout << "<filter-progress>" << std::endl
	      << progress << std::endl
	      << "</filter-progress>" << std::endl;

    try
    {
      (*itPatient)->Compute();
    }

    catch (itk::ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could not compute patient: " << iFile << std::endl
                << ex << std::endl;

      (*itPatient)->UnloadImages();
      continue;
    }

    progress = (iFile + 0.5)/nFiles;
    std::cout << "<filter-progress>" << std::endl
	      << progress << std::endl
	      << "</filter-progress>" << std::endl;

    (*itPatient)->Print( flgVerbose );

    progress = (iFile + 0.75)/nFiles;
    std::cout << "<filter-progress>" << std::endl
	      << progress << std::endl
	      << "</filter-progress>" << std::endl;


    (*itPatient)->UnloadImages();
  }

  progress = iFile/nFiles;
  std::cout << "<filter-progress>" << std::endl
            << progress << std::endl
            << "</filter-progress>" << std::endl;


  // Close the CSV file?
  // ~~~~~~~~~~~~~~~~~~~

  if ( foutOutputCSV )
  {
    foutOutputCSV->close();
    delete foutOutputCSV;
  }

  return result;
}
