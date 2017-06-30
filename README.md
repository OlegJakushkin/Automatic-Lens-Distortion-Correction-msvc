# Automatic-Lens-Distortion-Correction-msvc
Based on http://www.ipol.im/pub/art/2014/106/ Communicated by Rafael Grompone von Gioi Demo edited by Agustín Salgado. 
MSVC port with OpenCV support and simple wrapper example

Wrapper API is quick and dirty, specified in main.cpp inside the example:
```
struct Model {
	Model();
	std::vector<double> d /** RADIAL DISTORSION MODEL POLYNOMIAL */;
	double cx; /** CENTER OF THE DISTORSION MODEL */;
	double cy; /** CENTER OF THE DISTORSION MODEL */;
	double error; // model error
	double image_amplification_factor; // integer index to fix the way the corrected image is scaled to fit input size image
};

struct UndistortEngineSettings {
	UndistortEngineSettings();

	float canny_high_threshold;// high threshold for canny detector
	float initial_distortion_parameter;//left side of allowed distortion parameter interval
	float final_distortion_parameter;//Hough parameter
	float distance_point_line_max_hough;//Hough parameter
	//maximum difference allowed (in degrees) between edge and line orientation
	float angle_point_orientation_max_difference;

	int max_lines; //maximun number of lines estimated
	float angle_resolution; // angle discretization step (in degrees)
	float distance_resolution; // line distance discretization step
	float distortion_parameter_resolution;//distortion parameter discretization step

};

struct UndistortEngine {
	Model model;
	bool CreateModel(const UndistortEngineSettings & settings, const cv::Mat & frame);
	Mat ApplyModel(cv::Mat frame);
};
```

Built for MSVC 2012
OpenCV - single dependency
Note: for VS2015 use NuGet
```
Install-Package opencvdefault 
Install-Package opencvcontrib
```

Copyright (c) 2010-2013, AMI RESEARCH GROUP <lalvarez@dis.ulpgc.es>;
	      2017 OJ MSVC port
License : CC Creative Commons "Attribution-NonCommercial-ShareAlike"
see http://creativecommons.org/licenses/by-nc-sa/3.0/es/deed.en
