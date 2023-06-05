#include "Context.h"
#include "SyntheticAnnotation.h"
#include "Visualizer.h"
#include "CanopyGenerator.h"
#include "LiDAR.h"

using namespace std;
using namespace helios;


struct SyntheticAnnotationConfig {
public:
    int num_images;
    vector<string> annotation_type;
    vector<string> simulation_type;
    vector<string> labels;
    string xml_path;
    string output_path;
    void load_config(const char* path);
};


/**
    * Config files for a synthetic annotation generation have six lines:
    *
    *  1. The number of unique canopies to generate (the `num_images`
           parameter in Python, but the real number of images is the
           number of canopies times the number of camera views).
    *  2. The type of annotations to generate.
    *  3. The type of simulation of the imagery (LiDAR vs. RGB).
    *  4. The different labels to generate.
    *  5. The path to the XML file with generation parameters.
    *  6. The path to the output directory where images are rendered.
    *
    * This method parses the config file for the program.
    * @param path: The provided path to the config file.
    */
void SyntheticAnnotationConfig::load_config(const char* path) {
    ifstream file;
    file.open(path);

    string line;
    for (int i = 0; i < 6; i++) {
        getline(file, line);
        if (i == 0) {
            this->num_images = stoi(line);
        } else if (i == 1) {
            string delimeter = " "; size_t pos;
            vector<string> annotation_types;
            while ((pos = line.find(' ')) != string::npos) {
                this -> annotation_type.push_back(line.substr(0, pos));
                line.erase(0, pos + delimeter.length());
            }
            this->annotation_type.push_back(line);
        } else if (i == 2) {
            string delimeter = " "; size_t pos;
            vector<string> simulation_type;
            while ((pos = line.find(' ')) != string::npos)
            {
                this -> simulation_type.push_back(line.substr(0,pos));
                line.erase(0, pos + delimeter.length());
            }
            this->simulation_type.push_back(line);
        } else if (i == 3) {
            string delimeter = " "; size_t pos;
            vector<string> labels;
            while ((pos = line.find(' ')) != string::npos) {
                labels.push_back(line.substr(0, pos));
                line.erase(0, pos + delimeter.length());
            }
            labels.push_back(line); // final token
            this->labels = labels;
        } else if (i == 4) {
            this->xml_path = line;
        } else {
            this->output_path = line;
        }
    }
}


/**
    * Check if the string vector contains a specific element.
    */
bool contains(const vector<string>& v, const string& element) {
    return find(v.begin(), v.end(), element) != v.end();
}


int main(int argc, char** argv) {
    // Load in the configuration from the provided file.
    if (argc != 2)  throw invalid_argument("Expected path to the config file.");

    // Load the configuration from the file.
    SyntheticAnnotationConfig config;
    config.load_config(argv[1]);

    // Run the same procedure `num_images` number of times.
    for (int i = 0; i < config.num_images; i++) {
        // If there are no labels, then no need to instantiate the synthetic annotation class.
        if (!config.annotation_type.empty() && config.annotation_type[0] == "none") {

            // Declare the context.
            Context context;
            context.loadXML(config.xml_path.c_str());

            // Get the camera positions and lookat.
            vector<vec3> camera_position;
            context.getGlobalData("camera_position", camera_position);
            vector<vec3> camera_lookat;
            context.getGlobalData("camera_lookat", camera_lookat);

            // Declare the Canopy Generator and load the geometry from the XML file.
            CanopyGenerator cgen(&context);
            cgen.loadXML(config.xml_path.c_str());

            // Construct the Helios visualizer.
            Visualizer vis(800);
            vis.buildContextGeometry(&context);
            vis.hideWatermark();

            // Generate a new image for each camera view.
            string image_dir = config.output_path + "/" + string("image" + to_string(i));
            system(("mkdir -p " + image_dir).c_str());
            for (int i = 0; i < camera_position.size(); i++) {
                // Update the camera position.
                vis.setCameraPosition(camera_position[i], camera_lookat[i]);

                // Add the sky model.
                vis.addSkyDomeByCenter(20, make_vec3(0, 0, 0), 30,
                                       "plugins/visualizer/textures/SkyDome_clouds.jpg" );
                vis.setLightDirection(sphere2cart(
                        make_SphericalCoord(30 * M_PI / 180.f, 205 * M_PI / 180.f)));
                vis.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);
                vis.plotUpdate(true);

                // Save the image to the file.
                string this_view_path =  string("view" + to_string(i));
                system(("mkdir -p " + this_view_path).c_str());
                string image_view_path = this_view_path + "/" "RGB_rendering.jpeg";
                vis.printWindow(image_view_path.c_str());
            }

            // Skip the annotation stage.
            continue;
        }

        // Declare the context.
        Context context;
        context.loadXML(config.xml_path.c_str());

        // Declare the Canopy Generator and load the geometry from the XML file.
        CanopyGenerator cgen(&context);
        cgen.loadXML(config.xml_path.c_str());

        // Declare the Synthetic Annotation class.
        SyntheticAnnotation annotation(&context);

        // Choose either the LiDAR or RGB image simulation.
        if (!config.simulation_type.empty() && config.simulation_type[1] == "lidar") {
            // Get the UUID of all the elements on the scene
            vector<uint> UUID_trunk = cgen.getTrunkUUIDs();
            vector<uint> UUID_shoot = cgen.getBranchUUIDs();
            vector<uint> UUID_leaf = cgen.getLeafUUIDs();
            vector<uint> UUID_fruit = cgen.getFruitUUIDs();
            vector<uint> UUID_ground = cgen.getGroundUUIDs();

            // Add labels according to whatever scheme we want.
            vector<string> vlidar = config.labels;
            LiDARcloud lidarcloud;

            // Update the Primitive data with valid labels.
            if (contains(vlidar, "ground")) {
                context.setPrimitiveData( UUID_ground, "object_label", 1 );
            }
            if (contains(vlidar, "trunks")) {
                context.setPrimitiveData( UUID_trunk, "object_label", 2 );
            }
            if (contains(vlidar, "branches")) {
                context.setPrimitiveData( UUID_shoot, "object_label", 3 );
            }
            if (contains(vlidar, "leaves")) {
                context.setPrimitiveData( UUID_leaf, "object_label", 4 );
            }
            if (contains(vlidar, "fruits")) {
                context.setPrimitiveData( UUID_fruit, "object_label", 5 );
            }

            // Load canopy parameters and run the synthetic scan generation.
            lidarcloud.loadXML(config.xml_path.c_str());
            lidarcloud.syntheticScan( &context);

            // Export point cloud data.
            string this_image_dir = config.output_path + "/" + string("image" + to_string(i));
            system(("mkdir -p " + this_image_dir).c_str());
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // wait until folder is made
            string cloud_export = this_image_dir + "/" + string("point_cloud_" + to_string(i) + ".xyz");
            std::cout << "Writing LiDAR Point cloud to " << cloud_export << " " << std::endl;
            lidarcloud.exportPointCloud(cloud_export.c_str());
        }
        if (!config.simulation_type.empty() && config.simulation_type[0] == "rgb")
        {
            if (!config.annotation_type.empty() && config.annotation_type[0] != "none") {
                // Set the annotation type based on the configuration.
                vector<string> va = config.annotation_type;
                if (!contains(va, "semantic_segmentation")) {
                    annotation.disableSemanticSegmentation();
                }
                if (!contains(va, "object_detection")) {
                    annotation.disableObjectDetection();
                }
                if (!contains(va, "instance_segmentation")) {
                    annotation.disableInstanceSegmentation();
                }

                // Add labels according to whatever scheme we want.
                vector<string> vl = config.labels;
                for (int p = 0; p < cgen.getPlantCount(); p++) { // loop over vines
                    if (!config.simulation_type.empty() && config.simulation_type[0] == "rgb") {
                        if (contains(vl, "trunks")) {
                            annotation.labelPrimitives(cgen.getTrunkUUIDs(p), "trunks");
                        }
                        if (contains(vl, "branches")) {
                            annotation.labelPrimitives(cgen.getBranchUUIDs(p), "branches");
                        }
                        if (contains(vl, "leaves")) {
                            annotation.labelPrimitives(cgen.getLeafUUIDs(p), "leaves");
                        }
                        if (contains(vl, "fruits")) {
                            std::vector<std::vector<std::vector<uint>>> fruitUUIDs = cgen.getFruitUUIDs(p);
                            if( fruitUUIDs.size()==1 ){ // no clusters, only individual fruit
                                for(auto &fruit : fruitUUIDs.front())
                                    annotation.labelPrimitives( fruit, "clusters" );
                            } else if (fruitUUIDs.size() > 1) { // fruit contained within cluster - label by cluster
                                for (auto &cluster : fruitUUIDs)
                                    annotation.labelPrimitives(flatten(cluster), "clusters");
                            }
                        }

                    }
                }
            }

            // Render the annotations.
            string this_image_dir = config.output_path + "/" + string("image" + to_string(i));
            cout << this_image_dir;
            annotation.render(this_image_dir.c_str());
        }
    }
}
