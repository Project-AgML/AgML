#include "RadiationModel.h"
// #include "Visualizer.h"
#include <filesystem>

using namespace helios;

int main(){

    Context context;

    //Load in geometry
    std::vector<uint> UUIDs_bunny = context.loadPLY( "../../../PLY/StanfordBunny.ply", true );

    std::vector<uint> UUIDs_tile = context.addTile( nullorigin, make_vec2(2,2), nullrotation, make_int2(1000,1000));

    //set radiative properties
    context.setPrimitiveData(UUIDs_bunny, "reflectivity_red", 0.25f);
    context.setPrimitiveData(UUIDs_bunny, "reflectivity_green", 0.25f);
    context.setPrimitiveData(UUIDs_bunny, "reflectivity_blue", 0.5f);
    context.setPrimitiveData(UUIDs_tile, "reflectivity_red", 0.5f);
    context.setPrimitiveData(UUIDs_tile, "reflectivity_green", 0.25f);
    context.setPrimitiveData(UUIDs_tile, "reflectivity_blue", 0.25f);

    //set up IDs for bounding box labels
    context.setPrimitiveData(UUIDs_bunny, "bunny", uint(0));

    RadiationModel radiation(&context);

    uint sunID = radiation.addSunSphereRadiationSource(make_SphericalCoord(deg2rad(45), -deg2rad(45)));
    context.loadXML( "plugins/radiation/spectral_data/solar_spectrum_ASTMG173.xml", true);
    radiation.setSourceSpectrum( sunID, "solar_spectrum_ASTMG173");

    radiation.addRadiationBand("red");
    radiation.disableEmission("red");
    radiation.setSourceFlux(sunID, "red", 2.f);
    radiation.setScatteringDepth("red", 2);

    radiation.copyRadiationBand("red", "green");
    radiation.copyRadiationBand("red", "blue");

    radiation.enforcePeriodicBoundary("xy");

    std::vector<std::string> bandlabels = {"red", "green", "blue"};

    std::string cameralabel = "bunnycam";

    // Set radiation camera parameters
    vec3 camera_position = make_vec3(-0.1, 0, 0.5f);
    vec3 camera_lookat = make_vec3(0, 0, 0);
    CameraProperties cameraproperties;
    cameraproperties.camera_resolution = make_int2(1024, 1024);
    cameraproperties.focal_plane_distance = 0.4;
    cameraproperties.lens_diameter = 0.02f;
    cameraproperties.FOV_aspect_ratio = 1.4;
    cameraproperties.HFOV = 50.f;

    radiation.addRadiationCamera(cameralabel, bandlabels, camera_position, camera_lookat, cameraproperties, 100);

    context.loadXML( "plugins/radiation/spectral_data/camera_spectral_library.xml", true);
    radiation.setCameraSpectralResponse(cameralabel, "red", "calibrated_sun_NikonB500_spectral_response_red");
    radiation.setCameraSpectralResponse(cameralabel, "green","calibrated_sun_NikonB500_spectral_response_green");
    radiation.setCameraSpectralResponse(cameralabel, "blue", "calibrated_sun_NikonB500_spectral_response_blue");

    radiation.updateGeometry();

    radiation.runBand( bandlabels);

    std::string image_dir = "images";
    bool dir = std::filesystem::create_directory(image_dir);
    if (!dir && !std::filesystem::exists(image_dir))
    {
        helios_runtime_error("Error: image output directory " + image_dir + " could not be created. Exiting...");
    }

    radiation.writeCameraImage( cameralabel, bandlabels, "RGB", image_dir);
    radiation.writeDepthImageData( cameralabel, "depth", image_dir);
    radiation.writeNormDepthImage( cameralabel, "normdepth", 3, image_dir);
    radiation.writeImageBoundingBoxes( cameralabel, "bunny", 0, "bbox", image_dir);

    // Visualizer visualizer(800);
    //
    // visualizer.buildContextGeometry(&context);
    // visualizer.colorContextPrimitivesByData("radiation_flux_red");
    //
    // visualizer.plotInteractive();

}
