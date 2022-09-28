#include "Context.h"
#include "SyntheticAnnotation.h"
#include "Visualizer.h"
#include "CanopyGenerator.h"

using namespace std;
using namespace helios;


struct SyntheticAnnotationConfig {
public:
    int num_images;
    string annotation_type;
    string simulation_type;
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
            this->annotation_type = line;
        } else if (i == 2) {
            this->simulation_type = line;
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
        if (config.annotation_type == "none") {
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

        if (config.annotation_type != "none") {
            // Set the annotation type based on the configuration.
            if (config.annotation_type != "semantic_segmentation") {
                annotation.disableSemanticSegmentation();
            }
            if (config.annotation_type != "object_detection") {
                annotation.disableObjectDetection();
            }
            if (config.annotation_type != "instance_segmentation") {
                annotation.disableInstanceSegmentation();
            }

            // Add labels according to whatever scheme we want.
            vector<string> v = config.labels;
            for(int p = 0; p < cgen.getPlantCount(); p++) { // loop over vines
                if (config.simulation_type == "rgb") {
                    if (contains(v, "trunks")) {
                        annotation.labelPrimitives(cgen.getTrunkUUIDs(p), "trunks");
                    }
                    if (contains(v, "branches")) {
                        annotation.labelPrimitives(cgen.getBranchUUIDs(p), "branches");
                    }
                    if (contains(v, "cordon")) {
                        // Not implemented?
                        // annotation.labelPrimitives(cgen.getCordonUUIDs(p), "cordon");
                    }
                    if (contains(v, "leaves")) {
                        annotation.labelPrimitives(cgen.getLeafUUIDs(p), "leaves");
                    }
                    if (contains(v, "fruits")) {
                        std::vector<std::vector<std::vector<uint>>> fruitUUIDs = cgen.getFruitUUIDs(p);
                        for(int c = 0; c < fruitUUIDs.size(); c++){ // loop over fruit clusters
                            annotation.labelPrimitives( flatten(fruitUUIDs.at(c)), "clusters" );
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