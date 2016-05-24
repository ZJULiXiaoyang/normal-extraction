/**
 * Normal vector extraction using
 * author: Kanzhi Wu
 */



#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>


#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

const float fx = 570.3;
const float fy = 570.3;
const float cx = 319.5;
const float cy = 239.5;


// convert from rgb image and depth image to point cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr img2cloud(cv::Mat rgb_img, cv::Mat depth_img) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZRGB>() );
  cloud->width = rgb_img.cols;
  cloud->height = rgb_img.rows;
  cloud->is_dense = false;
  float bad_point = std::numeric_limits<float>::quiet_NaN();


  for ( int y = 0; y < rgb_img.rows; ++ y ) {
    for ( int x = 0; x < rgb_img.cols; ++ x ) {
      pcl::PointXYZRGB pt;
      pt.b = rgb_img.at<cv::Vec3b>(y,x)[0];
      pt.g = rgb_img.at<cv::Vec3b>(y,x)[1];
      pt.r = rgb_img.at<cv::Vec3b>(y,x)[2];
      if (depth_img.at<unsigned short>(y, x) == 0) {
        pt.x = bad_point; //std::numeric_limits<float>::quiet_NaN();
        pt.y = bad_point; //std::numeric_limits<float>::quiet_NaN();
        pt.z = bad_point; //std::numeric_limits<float>::quiet_NaN();
      }
      else {
        pt.z = depth_img.at<unsigned short>(y, x)/1000.;
        pt.x = pt.z*(x-cx)/fx;
        pt.y = pt.z*(y-cy)/fy;
      }
      cloud->points.push_back(pt);
    }
  }
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//  viewer->setBackgroundColor (0, 0, 0);
//  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
//  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "cloud");
//  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
//  viewer->addCoordinateSystem (1.0);
//  viewer->initCameraParameters ();
//  while (!viewer->wasStopped ()) {
//    viewer->spinOnce (100);
//    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//  }
  return cloud;
}


int main( int argc, char ** argv ) {
  if ( argc != 4 ) {
    cout << "Please input the following:\n";
    cout << "\t" << argv[0] << " <rgb image> <depth image> <method>\n";
    cout << "\t" << "methods:\n";
    cout << "\t\t" << "1 - least square\n";
    cout << "\t\t" << "2 - least square in multi-thread computation\n";
    return -1;
  }



  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  std::string rgbf = std::string(argv[1]);
  std::string depthf = std::string(argv[2]);

  cv::Mat rgb_img = cv::imread( rgbf, CV_LOAD_IMAGE_COLOR );
  cv::Mat depth_img = cv::imread( depthf, CV_LOAD_IMAGE_ANYDEPTH );

  cout << "Convert images to point cloud ...\n";
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = img2cloud( rgb_img, depth_img );

  cout << "Computing normal vectors ...\n";
  pcl::PointCloud<pcl::Normal>::Ptr normals( new pcl::PointCloud<pcl::Normal> () );

  if ( atoi(argv[3]) == 2 ) {
    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> norm_est;
    norm_est.setRadiusSearch(0.10);
    norm_est.setInputCloud (cloud);
    norm_est.compute (*normals);
  }
  else {
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> norm_est;
    norm_est.setRadiusSearch(0.10);
    norm_est.setInputCloud( cloud );
    norm_est.compute( *normals );
  }


  pcl::io::savePCDFileASCII( "normal.pcd", *normals );

  ofstream file;
  file.open("normal.txt");
  for ( int y = 0; y < normals->height; ++ y ) {
    for ( int x = 0; x < normals->width; ++ x ) {
      int idx = y*normals->width+x;
      if ( pcl_isfinite( normals->at(idx).normal_x ) ) {
          file<<(x+1)<<" "<<(y+1)<<" "<<normals->at(idx).normal_x<<" "<<normals->at(idx).normal_y<<" "<<normals->at(idx).normal_z<<endl;
      }
      else {
         file<<(x+1)<<" "<<(y+1)<<" "<<0<<" "<<0<<" "<<0<<endl;
      }
    }
  }
  file.close();


  cv::Mat normal_img( rgb_img.rows, rgb_img.cols, CV_8UC3 );
  for ( int y = 0; y < normals->height; ++ y ) {
    for ( int x = 0; x < normals->width; ++ x ) {
      int idx = y*normals->width+x;
      if ( pcl_isfinite( normals->at(idx).normal_x ) ) {
        normal_img.at<cv::Vec3b>(y, x)[0] = (int)((normals->at(idx).normal_x+1)/2*255);
        normal_img.at<cv::Vec3b>(y, x)[1] = (int)((normals->at(idx).normal_y+1)/2*255);
        normal_img.at<cv::Vec3b>(y, x)[2] = (int)((normals->at(idx).normal_z+1)/2*255);
      }
      else {
        normal_img.at<cv::Vec3b>(y, x)[0] = 0;
        normal_img.at<cv::Vec3b>(y, x)[1] = 0;
        normal_img.at<cv::Vec3b>(y, x)[2] = 0;
      }
    }
  }

  cv::imshow( "normal", normal_img );
  cv::waitKey(0);

  cout << "save normal vector to normal.png ...\n";
  cv::imwrite( "normal.png", normal_img );




}
