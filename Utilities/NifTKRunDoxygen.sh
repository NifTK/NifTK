GIT_SSL_NO_VERIFY=1
NIFTK_DRC_ANALYZE=ON
NIFTK_MAKE_PACKAGE=ON
DISPLAY=mespak3.cs.ucl.ac.uk:1
PATH=/scratch0/NOT_BACKED_UP/clarkson/install/kwstyle/bin:/scratch0/NOT_BACKED_UP/clarkson/install/cmake/bin:/scratch0/NOT_BACKED_UP/clarkson/install/git/bin:/scratch0/NOT_BACKED_UP/clarkson/install/doxygen/bin:/scratch0/NOT_BACKED_UP/clarkson/install/qt/bin:/usr/local/bin:/usr/bin:/bin
BASE=/scratch0/NOT_BACKED_UP/clarkson/auto

cd $BASE/niftk-nightly-release/NifTK-build
chmod a+rx bin/GenerateCommandLineDoxygen 
bin/GenerateCommandLineDoxygen
chmod a+rx bin/GenerateTestingReports
bin/GenerateTestingReports

/scratch0/NOT_BACKED_UP/clarkson/install/doxygen/bin/doxygen doxygen.config 
cp -rpu Doxygen/15.05.0/* /cs/sys/www0/marine/html/cmic.cs.ucl.ac.uk/platform/niftk/15.05.0 
cd $BASE/NiftyGuide-SuperBuild-Release/NiftyGuide-build
/scratch0/NOT_BACKED_UP/clarkson/install/doxygen/bin/doxygen doxygen.config 
cp -rpu Doxygen/html/* /cs/sys/www0/marine/html/cmic.cs.ucl.ac.uk/platform/niftk/15.05.0/NiftyGuide 
cd $BASE/NiftyLink-SuperBuild-Release/NiftyLink-build 
/scratch0/NOT_BACKED_UP/clarkson/install/doxygen/bin/doxygen doxygen.config 
cp -rpu Doxygen/html/* /cs/sys/www0/marine/html/cmic.cs.ucl.ac.uk/platform/niftk/15.05.0/NiftyLink 
cp -rpu Doxygen/html/* /cs/sys/www0/marine/html/cmic.cs.ucl.ac.uk/NiftyLink-API 
cd /cs/sys/www0/marine/html/cmic.cs.ucl.ac.uk/platform/niftk/ 
chmod a+rx 15.05.0 
cd 15.05.0 
find . -type f -exec chmod a+r {} \; 
find . -type d -exec chmod a+rx {} \; 
cd /cs/sys/www0/marine/html/cmic.cs.ucl.ac.uk/NiftyLink-API 
find . -type f -exec chmod a+r {} \; 
find . -type d -exec chmod a+rx {}  \;
