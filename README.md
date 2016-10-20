HOW TO COMPILE THIS PROJECT IN WINDOWS </br>
Step 0:Open cmd, and change directory to project directory. Use this command "cd /your/project/path/SemiCRFSegmentation". </br>
Step 1:Create a new directory in SemiCRFSegmentation.For example, use this command "mkdir build" </br>
Step 2:Change your directory. Use this command "cd build". </br>
Step 3:Build project. Use this command "cmake .. -DEIGEN3_INCLUDE_DIR=/your/eign/path -DN3L_INCLUDE_DIR=/your/LibN3L-2.0/path". </br>
Step 4:Then you can double click "SemiCRFSegmentation.sln" to open this project. </br>
Step 5:Now you can complie this project by Visual Studio. </br>
Step 6:If you want to run this project.Please open project properties and add this argument. </br>
"-train /your/training/corpus -dev /your/development/corpus -test /your/test/corpus -option /your/option/file -l" </br>

NOTE: Make sure you have eigen and LibN3L-2.0. </br>
Eigen:http://eigen.tuxfamily.org/index.php?title=Main_Page </br>
LibN3L-2.0:https://github.com/zhangmeishan/LibN3L-2.0 </br>