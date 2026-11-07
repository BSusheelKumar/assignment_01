#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>

class CartonDetectorNode : public rclcpp::Node
{
public:
    CartonDetectorNode() : Node("carton_detector_node")
    {
        // ROI Parameters (Now relative to the "world" frame)
        this->declare_parameter("roi_min_x", 1.0);
        this->declare_parameter("roi_max_x", 2.0);
        this->declare_parameter("roi_min_y", -0.8);
        this->declare_parameter("roi_max_y", 0.0);
        this->declare_parameter("roi_min_z", -0.4);
        this->declare_parameter("roi_max_z", 0.4);
        this->declare_parameter("surface_thickness", 0.02);

        // TF Broadcaster & Listener
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Publishers for Visualization
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("detected_carton_marker", 10);
        filtered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_carton_points", 10);

        // Subscriber
        point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera/points", rclcpp::SensorDataQoS(),
            std::bind(&CartonDetectorNode::pointCloudCallback, this, std::placeholders::_1));
            
        RCLCPP_INFO(this->get_logger(), "Carton Detector Node (World Frame) Initialized.");
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // 1. Convert ROS PointCloud2 to PCL format
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);
        if (cloud->empty()) return;

        // 2. Fetch the camera-to-world transform
        geometry_msgs::msg::TransformStamped transformStamped;
        try {
            // Lookup transform from camera frame to "world" frame
            transformStamped = tf_buffer_->lookupTransform("world", msg->header.frame_id, tf2::TimePointZero);
        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for TF: %s", ex.what());
            return;
        }

        // 3. Transform the point cloud to the world frame
        Eigen::Affine3f transform_eigen = Eigen::Affine3f::Identity();
        transform_eigen.translation() << transformStamped.transform.translation.x, 
                                         transformStamped.transform.translation.y, 
                                         transformStamped.transform.translation.z;
        Eigen::Quaternionf q_eigen(transformStamped.transform.rotation.w, 
                                   transformStamped.transform.rotation.x, 
                                   transformStamped.transform.rotation.y, 
                                   transformStamped.transform.rotation.z);
        transform_eigen.rotate(q_eigen);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_world(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud, *cloud_world, transform_eigen);

        // 4. CropBox Filter (using world-relative ROI)
        pcl::CropBox<pcl::PointXYZ> crop_box;
        crop_box.setInputCloud(cloud_world);
        crop_box.setMin(Eigen::Vector4f(
            this->get_parameter("roi_min_x").as_double(),
            this->get_parameter("roi_min_y").as_double(),
            this->get_parameter("roi_min_z").as_double(), 1.0));
        crop_box.setMax(Eigen::Vector4f(
            this->get_parameter("roi_max_x").as_double(),
            this->get_parameter("roi_max_y").as_double(),
            this->get_parameter("roi_max_z").as_double(), 1.0));
            
        pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        crop_box.filter(*cropped_cloud);

        if (cropped_cloud->empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "No cartons found in the ROI.");
            return;
        }

        // 5. Find Maximum Z (since world Z points UP towards the ceiling)
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cropped_cloud, min_pt, max_pt);

        // 6. PassThrough Filter: Extract strictly the top flat surface
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cropped_cloud);
        pass.setFilterFieldName("z");
        double thickness = this->get_parameter("surface_thickness").as_double();
        pass.setFilterLimits(max_pt.z - thickness, max_pt.z);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr top_surface(new pcl::PointCloud<pcl::PointXYZ>);
        pass.filter(*top_surface);

        if (top_surface->points.size() < 100) return;

        // 7. Compute Centroid
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*top_surface, centroid);

        // 8. PCA for Orientation (Yaw)
        pcl::PCA<pcl::PointXYZ> pca;
        pca.setInputCloud(top_surface);
        Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
        double yaw = std::atan2(eigen_vectors(1, 0), eigen_vectors(0, 0));

        // 9. Broadcast Target TF
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = msg->header.stamp;
        t.header.frame_id = "world"; 
        t.child_frame_id = "next_target_carton";

        t.transform.translation.x = centroid.x();
        t.transform.translation.y = centroid.y();
        t.transform.translation.z = centroid.z();

        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw);
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();

        tf_broadcaster_->sendTransform(t);

        // 10. Publish Isolated Points
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*top_surface, cloud_msg);
        cloud_msg.header.stamp = msg->header.stamp;
        cloud_msg.header.frame_id = "world";
        filtered_cloud_pub_->publish(cloud_msg);

        // 11. Publish 3D Bounding Box Marker
        visualization_msgs::msg::Marker marker;
        marker.header.stamp = msg->header.stamp;
        marker.header.frame_id = "world";
        marker.ns = "carton";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.x = centroid.x();
        marker.pose.position.y = centroid.y();
        // Shift marker center down by half the box height (0.1m) so the top aligns with the points
        marker.pose.position.z = centroid.z() - 0.1; 
        
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();
        marker.pose.orientation.w = q.w();
        
        // Exact carton dimensions: 400mm x 300mm x 200mm
        marker.scale.x = 0.4;
        marker.scale.y = 0.3;
        marker.scale.z = 0.2;
        
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 0.5f; 
        
        marker_pub_->publish(marker);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_pub_;
    
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CartonDetectorNode>());
    rclcpp::shutdown();
    return 0;
}