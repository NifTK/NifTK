/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkLogHelper.h"
#include <itkDOMNodeXMLReader.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkDOMNode.h>
#include <itkMacro.h>
#include <itkTileImageFilter.h>
#include <iostream>
#include <cstddef>


/*!
* \file niftkOCTVolumeConstructor.cxx
* \page niftkOCTVolumeConstructor
* \section niftkOCTVolumeConstructorSummary Reconstructs an OCT 3D volume from 2D tif images.
*/

std::string xml_octvolumeconstructor=
"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
"<executable>\n"
"   <category>Utils</category>\n"
"   <title>OCT Volume Constructor</title>\n"
"   <description>Vesselness filter</description>\n"
"   <version>0.0.1</version>\n"
"   <documentation-url></documentation-url>\n"
"   <license>BSD</license>\n"
"   <contributor>Maria A. Zuluaga (UCL)</contributor>\n"
"   <parameters>\n"
"      <label>Inputs</label>\n"
"      <description>Input and output images</description>\n"
"      <file fileExtensions=\"*.xml\">\n"
"          <name>inputxml</name>\n"
"          <flag>i</flag>\n"
"          <description>XML descriptor</description>\n"
"          <label>XML file</label>\n"
"          <channel>input</channel>\n"
"      </file>\n"
"      <directory>\n"
"          <name>brainImageName</name>\n"
"          <flag>f</flag>\n"
"          <description>Image directory</description>\n"
"          <label>Image Directory</label>\n"
"          <channel>input</channel>\n"
"      </directory>\n"
"   </parameters>\n"
"</executable>\n";

typedef itk::Image<unsigned short, 2> ImageType;
typedef itk::Image<unsigned short, 3> Image3DType;
typedef itk::TileImageFilter< ImageType, Image3DType > TilerType;
typedef itk::ImageFileWriter< Image3DType > WriterType;


void Usage(char *exec)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << " " << std::endl;
  std::cout << "Converts a set of 2D OCT tif files into a 3D volume." << std::endl;
  std::cout << " " << std::endl;
  std::cout << " " << exec << " [-i inputXML -f image folder]" << std::endl;
  std::cout << " " << std::endl;
}

static std::string CLI_PROGRESS_UPDATES = std::string(getenv("NIFTK_CLI_PROGRESS_UPD") != 0 ? getenv("NIFTK_CLI_PROGRESS_UPD") : "");

void startProgress()
{
  if (CLI_PROGRESS_UPDATES.find("ON") != std::string::npos ||
      CLI_PROGRESS_UPDATES.find("1") != std::string::npos)
  {
    std::cout << "<filter-start>\n";
    std::cout << "<filter-name>niftkOCTVolumeConstructor</filter-name>\n";
    std::cout << "<filter-comment>niftkOCTVolumeConstructor</filter-comment>\n";
    std::cout << "</filter-start>\n";
    std::cout << std::flush;
  }
}

void progressXML(int p, std::string text)
{
  if (CLI_PROGRESS_UPDATES.find("ON") != std::string::npos ||
      CLI_PROGRESS_UPDATES.find("1") != std::string::npos)
  {
    float k = static_cast<float>((float) p / 100);
    std::cout << "<filter-progress>" << k <<"</filter-progress>\n";
    std::cout << std::flush;
  }
}

void GetImageDimensions(itk::DOMNode* n, int &width, int &height)
{
  width = atoi(n->GetChild("Width")->GetTextChild()->GetText().c_str());
  height = atoi(n->GetChild("Height")->GetTextChild()->GetText().c_str());
  std::cout << height << std::endl;
}

void GetImageScale(itk::DOMNode* n, float &sx, float &sy)
{
  sx = atof(n->GetChild("ScaleX")->GetTextChild()->GetText().c_str());
  sy = atof(n->GetChild("ScaleY")->GetTextChild()->GetText().c_str());
}

ImageType::Pointer GetImage(itk::DOMNode* n, std::string folder)
{
  std::string full_path = n->GetChild("ExamURL")->
      GetTextChild()->GetText();
  std::size_t i = full_path.rfind("\\");
  if (i == std::string::npos)
    return NULL;

  std::string new_path = folder + "/" +
      full_path.substr(i+1, full_path.length() -i);
  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(new_path);
  reader->UpdateLargestPossibleRegion();
  return reader->GetOutput();
}

void closeProgress(std::string img, std::string status)
{
  if (CLI_PROGRESS_UPDATES.find("ON") != std::string::npos ||
      CLI_PROGRESS_UPDATES.find("1") != std::string::npos)
  {
    std::cout << "<filter-result name=outputImageName>"  << img << "</filter-result>\n";
    std::cout << "<filter-result name=exitStatusOutput>" << status << "</filter-result>\n";
    std::cout << "<filter-progress>100</filter-progress>\n";
    std::cout << "<filter-end>\n";
    std::cout << "<filter-name>niftkOCTVolumeConstructor</filter-name>\n";
    std::cout << "<filter-comment>Finished</filter-comment></filter-end>\n";
    std::cout << std::flush;
  }
}

int main( int argc, char* argv[] )
{
  std::string inputxml = "";
  std::string imgFolder = "";

  for(int i=1; i < argc; i++)
  {
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0)
    {
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0)
    {
      inputxml=argv[++i];
      std::cout << "Set -i=" << inputxml << std::endl;
    }
    else if(strcmp(argv[i], "-f") == 0)
    {
      imgFolder=argv[++i];
      std::cout << "Set -o=" << imgFolder << std::endl;
    }
    else if(strcmp(argv[i], "--xml") == 0){
      std::cout << xml_octvolumeconstructor;
      return EXIT_SUCCESS;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Validate command line args
  if (inputxml.length() == 0 || imgFolder.length() == 0)
  {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }

  //Check for the extension
  if (inputxml.rfind(".xml") == std::string::npos)
  {
    std::cerr << "No XML file has been recognised." <<std::endl;
    Usage(argv[0]);
    return EXIT_FAILURE;
  }

  itk::FixedArray< unsigned int, 3 > layout;
  layout[0] = 1;
  layout[1] = 1;
  layout[2] = 0;

  startProgress();
  int tasks=3;
  int progress_unit = floor(100.0f / (float)tasks +0.5);
  int progresscounter = progress_unit;
  
  typedef Image3DType::SpacingType SpacingType;
  typedef Image3DType::IndexType OriginType;
  
  try
  {
    progressXML(progresscounter, "Reading XML file...");
    progresscounter+=progress_unit;
    itk::DOMNodeXMLReader::Pointer reader = itk::DOMNodeXMLReader::New();
    reader->SetFileName( inputxml );
    reader->Update();
    itk::DOMNode::Pointer dom = reader->GetOutput();
    itk::DOMNode::Pointer root = dom->Find( "BODY" );


    std::vector<itk::DOMNode*>patients;
    root->GetChildren("Patient",patients);
    for (unsigned int i = 0; i < patients.size(); ++i)  //For every patient
    {
      std::string imageOut = imgFolder+"/vol";
      itk::DOMNode* tmp = patients[i]->GetChild("ID");
      if (tmp == NULL)
      {
        std::cerr << "XML has errors. Program will exit" << std::endl;
        return EXIT_FAILURE;
      }
      std::string pat_base = tmp->GetTextChild()->GetText();

      std::vector<itk::DOMNode*>studies;
      patients[i]->GetChildren("Study",studies);    //For every study
      for (unsigned int j = 0; j < studies.size(); ++j)
      {
        itk::DOMNode* tmp = studies[i]->GetChild("ID");
        if (tmp ==NULL)
        {
          std::cerr << "XML has errors. Program will exit" << std::endl;
          return EXIT_FAILURE;
        }
        TilerType::Pointer tiler = TilerType::New();
        tiler->SetLayout( layout );

        std::string stu_base = pat_base + "_" + tmp->GetTextChild()->GetText();
        std::vector<itk::DOMNode*>series;
        studies[j]->GetChildren("Series",series);

        for (unsigned int k = 0; k < series.size(); ++k)
        {
          itk::DOMNode* tmp = series[i]->GetChild("ID");
          if (tmp ==NULL)
          {
            std::cerr << "XML has errors. Program will exit" << std::endl;
            return EXIT_FAILURE;
          }
          imageOut += stu_base + "_"
              + tmp->GetTextChild()->GetText() +".nii";
          std::vector<itk::DOMNode*>images;
          series[k]->GetChildren("Image",images);
          float scale_x, scale_y;
          int size_x, size_y;
          SpacingType sp;

          for (unsigned int l = 1; l < images.size(); ++l) //image 0 should be skipped
          {

            itk::DOMNode* context = images[l]->
                GetChild("OphthalmicAcquisitionContext");
            unsigned int imgNum = atoi(images[l]->GetChild("ImageNumber")->GetTextChild()
                                       ->GetText().c_str());
            if (l == 1)
            {
              GetImageDimensions(context,size_x,size_y);
              GetImageScale(context,scale_x,scale_y);
              sp[0] = scale_x;
              sp[1] = scale_y;
              sp[3] = 1; // please fix!
            }
            else      // For the purpose of validating that things are ok.
            {
              float tmp_sx, tmp_sy;
              int tmp_dimx, tmp_dimy;
              GetImageDimensions(context,tmp_dimx,tmp_dimy);
              GetImageScale(context,tmp_sx,tmp_sy);
              if (tmp_sx != scale_x || tmp_sy != scale_y
                  || tmp_dimx != size_x || tmp_dimy != size_y)
              {
                std::cerr << "The images do not match. Program will exit" << std::endl;
                return EXIT_FAILURE;
              }
            }

            ImageType::Pointer inputImageTile = GetImage(images[l]->GetChild("ImageData"),imgFolder);
            if (inputImageTile.IsNull())
            {
              std::cerr << "error reading image." << std::endl;
              return EXIT_FAILURE;
            }
            inputImageTile->DisconnectPipeline();
            tiler->SetInput( imgNum-1, inputImageTile );
          }
          progressXML(progresscounter, "Tiling images...");
          progresscounter+=progress_unit;
          tiler->SetDefaultPixelValue( 0 );
          tiler->Update();
          Image3DType::Pointer imgdata = tiler->GetOutput();
          imgdata->SetSpacing(sp);

          progressXML(progresscounter, "Writing result...");
          progresscounter+=progress_unit;
          WriterType::Pointer writer = WriterType::New();
          writer->SetInput(imgdata);
          writer->SetFileName(imageOut);
          writer->Update();
          closeProgress(imageOut, "Normal exit");
        }
      }
    }
  }
  catch ( const itk::ExceptionObject& eo )
  {
    eo.Print( std::cerr );
    closeProgress("FAILED", "Failed");
    return EXIT_FAILURE;
  }
  catch ( ... )
  {
    std::cerr << "Unknown exception caught!" << std::endl;
    closeProgress("FAILED", "Failed");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
