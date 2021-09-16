#include "Context.h"
#include "CanopyGenerator.h"
#include "Visualizer.h"
#include <iomanip> 
#include <random>
#include "LiDAR.h"

using namespace helios;
using namespace std;

//Function definitions -- always before the main code

helios::RGBcolor int2rgb( const int ID ){

  float R, G, B;
  int r, g, b;
  int rem;

  b = floor(float(ID)/256.f/256.f);
  rem = ID-b*256*256;
  g = floor(float(rem)/256.f);
  rem = rem-g*256;
  r = rem;

  R = float(r)/255.f;
  G = float(g)/255.f;
  B = float(b)/255.f;

  return helios::make_RGBcolor(R,G,B);
  
}

int rgb2int( const helios::RGBcolor color ){

  int ID = color.r*255+color.g*255*256+color.b*255*256*256;

  return ID;
  
}

void writePixelID( const char* filename, int labelminpixels, Visualizer* vis ){

  uint framebufferH, framebufferW;

  vis->getFramebufferSize(framebufferW,framebufferH);

  std::vector<uint> pixels;
  pixels.resize(framebufferH*framebufferW*3);

  vis->getWindowPixelsRGB( &pixels[0] );

  int t=0;
  int xmin = framebufferW;
  int xmax = 0;
  int ymin = framebufferH;
  int ymax = 0;
  int pixelcount = 0;
  
  for( int j=0; j<framebufferH; j++ ){
    for( int i=0; i<framebufferW; i++ ){

      if( pixels[t]==255 && pixels[t+1]==255 && pixels[t+2]==255 ){
	t+=3;
	continue;
      }

      if( i<xmin ){
	xmin=i;
      }
      if( i>xmax ){
	xmax=i;
      }
      if( j<ymin ){
	ymin=j;
      }
      if( j>ymax ){
	ymax=j;
      }
      
      t+=3;
      pixelcount++;
	  
    }
  }

  if( xmin==framebufferW || xmax==0 || ymin==framebufferH || ymax==0 || pixelcount<labelminpixels ){
    return;
  }

  std::ofstream file(filename);

  file << xmin << " " << xmax << " " << ymin << " " << ymax << std::endl;

  t=0;
  for( int j=0; j<framebufferH; j++ ){
    for( int i=0; i<framebufferW; i++ ){

      if( i>=xmin && i<=xmax && j>=ymin && j<=ymax ){
      
	int ID = rgb2int( make_RGBcolor(pixels[t]/255.f,pixels[t+1]/255.f,pixels[t+2]/255.f) );
	file << ID << " " <<  std::flush;

      }
	
      t+=3;
      
    }
    if( j>=ymin && j<=ymax ){
      file << std::endl;
    }
  }
  file.close();

}

float rand_FloatRange(float a, float b)
{
    random_device randdev;
    mt19937 generator(randdev());
    uniform_real_distribution<> distrib(a, b);
    return (distrib(generator));
}

// Main code here

  int main( void ){
    
bool flag=true;
    Context context;
    CanopyGenerator canopygenerator(&context);
    //Reading Geometry
    canopygenerator.loadXML("../xmloutput_for_helios/tmp_canopy_params_image.xml");

    if (flag) { //LiDAR point cloud generation -- with labels --
    
    cout << "Generating Synthetich LiDAR data \n";  
    
    vector<uint> UUID_trunk = canopygenerator.getTrunkUUIDs();
    vector<uint> UUID_shoot = canopygenerator.getBranchUUIDs();
    vector<uint> UUID_leaf = canopygenerator.getLeafUUIDs();
    vector<uint> UUID_fruit = canopygenerator.getFruitUUIDs();
    vector<uint> UUID_ground = canopygenerator.getGroundUUIDs();

    //uint UUID_ground = context.addPatch( make_vec3(0,0,0), make_vec2(20,20), make_SphericalCoord(0,0), RGB::brown );

    context.setPrimitiveData( UUID_ground, "object_label", 1 );
    context.setPrimitiveData( UUID_trunk, "object_label", 2 );
    context.setPrimitiveData( UUID_shoot, "object_label", 3 );
    context.setPrimitiveData( UUID_leaf, "object_label", 4 );
    context.setPrimitiveData( UUID_fruit, "object_label", 5 );
    
 LiDARcloud lidarcloud;

 lidarcloud.loadXML("../xmloutput_for_helios/tmp_canopy_params_image.xml");
    
    Timer timer;
    timer.tic();
 lidarcloud.syntheticScan( &context);
    timer.toc("Time to run synthetic scan.");

 lidarcloud.exportPointCloud( "/home/dguevara/Documents/AgML/src/agml/output/point_cloud/synthetic_scan_0.xyz" );
    }
    else{ //Image Generation

   RGBcolor grape_color(0.18,0.2,0.25); //color of grapes in visualizations 

   Timer timer;

    //Visualizer difinition

   int windowW = 1000;              //width of visualizer window in pixels
   int windowH = 1000;              //height of visualizer window in pixels

   int labelminpixels = 10;         //minimum number of fruit pixels in a label
   bool rectangularlabels = true;   //write rectangular bounding box labels
   bool polygonlabels = true;       //write polygon labels

   //The 'camera' will stop at each of the (x,y,z) positions provided in this vector. It will always be pointed in the row-normal direction looking toward the center of the vineyard.
   std::vector<vec3> camera_pos;
   float x_camera = rand_FloatRange (  0.3, 1.3);
   float y_camera = rand_FloatRange (  -1.2,  -1.7);
   float z_camera = rand_FloatRange (  0.7,  1.1);

   camera_pos.push_back( make_vec3(x_camera, y_camera, z_camera) );

  //testing the r-g-b encoding scheme here
  int ID1 = 77830;
  int ID2 = rgb2int(int2rgb(ID1));

  if( ID1!=ID2 ){
    std::cout << "RGB encoding functions not working." << std::endl;
    std::cout << ID2 << std::endl;
    exit(EXIT_FAILURE);
  }

   timer.tic();
  //Get UUIDs, and set tags for element type
  std::vector<uint> UUIDs_wood, UUIDs_ground;
  std::vector<std::vector<uint> > UUIDs_leaf;
  std::vector<std::vector<std::vector<uint> > > UUIDs_grape;

  std::vector<std::vector<uint> > UUIDs_group;

  for( int p=0; p<canopygenerator.getPlantCount(); p++ ){
    UUIDs_wood = canopygenerator.getTrunkUUIDs(p);
    context.setPrimitiveData( UUIDs_wood, "element_type", "trunk" );
    UUIDs_group.push_back(UUIDs_wood);
    
    UUIDs_wood = canopygenerator.getBranchUUIDs(p);
    context.setPrimitiveData( UUIDs_wood, "element_type", "cane" );
    UUIDs_group.push_back(UUIDs_wood);
    
    UUIDs_leaf = canopygenerator.getLeafUUIDs(p);
    context.setPrimitiveData( UUIDs_leaf, "element_type", "leaf" );
    UUIDs_group.push_back( flatten(UUIDs_leaf) );
    
    UUIDs_grape = canopygenerator.getFruitUUIDs(p);
    context.setPrimitiveData( UUIDs_grape, "element_type", "grape" );
    for( int i=0; i<UUIDs_grape.size(); i++ ){
      UUIDs_group.push_back( flatten(UUIDs_grape.at(i)) );
    }
    
  }
  UUIDs_ground = canopygenerator.getGroundUUIDs();
  context.setPrimitiveData( UUIDs_ground, "element_type", "ground" );
  UUIDs_group.push_back( UUIDs_ground );

  timer.toc("Time to generate geometry");
    
  std::vector<uint> UUIDs_all = context.getAllUUIDs();  


  //looping through camera positions
  
  for( int view=0; view<camera_pos.size(); view++ ){

    //cout << "camera perspective # " << view << endl;

    cout << "Camera position -- x: " << x_camera << ", y: " << y_camera << ", z: " << z_camera << " -- " << endl;

    char outfile[120];
    //sprintf(outfile,"mkdir ../output/images/",view);
    //system(outfile);

    for( int p=0; p<UUIDs_all.size(); p++ ){
      context.getPrimitivePointer(UUIDs_all.at(p))->useTextureColor();
      std::string type;
      if( context.doesPrimitiveDataExist( UUIDs_all.at(p),"element_type" ) ){
	context.getPrimitiveData(UUIDs_all.at(p),"element_type",type);
	if( type.compare("grape")==0 ){
	  context.getPrimitivePointer(UUIDs_all.at(p))->setColor( grape_color );
	}
      }else{
	std::cout << "WARNING: Primitive data ""element_type"" does not exist for primitive " << UUIDs_all.at(p) << std::endl;
      }
    }

    
    Visualizer vis_RGB(windowW,windowH,8,false);

    uint framebufferW, framebufferH;
    vis_RGB.getFramebufferSize(framebufferW,framebufferH);

    timer.tic();
    vis_RGB.buildContextGeometry(&context);
    vis_RGB.hideWatermark();
    vis_RGB.setBackgroundColor( make_RGBcolor(0.9,0.9,1) );

    vis_RGB.setCameraPosition( camera_pos.at(view), make_vec3(camera_pos.at(view).x,0,camera_pos.at(view).z) );

    vis_RGB.setLightDirection( sphere2cart(make_SphericalCoord(30*M_PI/180.f,205*M_PI/180.f)) );
    vis_RGB.setLightingModel( Visualizer::LIGHTING_PHONG_SHADOWED );

    // vis_RGB.addSkyDomeByCenter( 50, make_vec3(0,0,0), 30, "../../../Helios/plugins/visualizer/textures/SkyDome_clouds.jpg" );
    //vis_RGB.plotInteractive();

    vis_RGB.plotUpdate();
    
    wait(5);
    
    

  sprintf(outfile,"/home/dguevara/Documents/AgML/src/agml/output/images/Image_0/RGB_rendering.jpeg");
    vis_RGB.printWindow(outfile);

  
    vis_RGB.closeWindow();

    timer.toc("Time to render and write RGB image");

    Visualizer vis(windowW,windowH,0,false);

    vis.getFramebufferSize(framebufferW,framebufferH);

  sprintf(outfile,"/home/dguevara/Documents/AgML/src/agml/output/images/Image_0/ID_mapping.txt");
    std::ofstream mapping_file(outfile);
    
    int gID=0;
    for( int g=0; g<UUIDs_group.size(); g++ ){

      std::string type;
      context.getPrimitiveData(UUIDs_group.at(g).front(),"element_type",type);
      mapping_file << g << " " << type << std::endl;
      
      gID=g;
      RGBcolor code = int2rgb( gID );
      for( int p=0; p<UUIDs_group.at(g).size(); p++ ){
	
	if( type.compare("grape")==0 ){
	  context.getPrimitivePointer(UUIDs_group.at(g).at(p))->setColor( code );
	  int gID2 = rgb2int(context.getPrimitivePointer(UUIDs_group.at(g).at(p))->getColor());
	  if( gID!=gID2 ){
	    std::cout << "IDs do not match" << std::endl;
	  }
	}else{
	  context.getPrimitivePointer(UUIDs_group.at(g).at(p))->setColor( make_RGBcolor(1,1,1) );
	}
	context.getPrimitivePointer(UUIDs_group.at(g).at(p))->overrideTextureColor();
      }
    }
    
    mapping_file.close();

    vis.setBackgroundColor( make_RGBcolor(1,1,1) );

    timer.tic();
      
    vis.buildContextGeometry(&context);
    vis.hideWatermark();

    vis.setCameraPosition( camera_pos.at(view), make_vec3(camera_pos.at(view).x,0,camera_pos.at(view).z) );

    vis.plotUpdate();
    
    std::vector<uint> pixels;
    pixels.resize(framebufferH*framebufferW*3);
    
    vis.getWindowPixelsRGB( &pixels[0] );

  sprintf(outfile,"/home/dguevara/Documents/AgML/src/agml/output/images/Image_0/pixelID_combined.txt");
    std::ofstream file(outfile);
    std::vector<int> ID;
    int t=0;
    for( int j=framebufferH-1; j>0; j-- ){
      for( int i=0; i<framebufferW; i++ ){
	
	ID.push_back( rgb2int(make_RGBcolor(pixels[t]/255.f,pixels[t+1]/255.f,pixels[t+2]/255.f)) );
	file << ID.back() << " " <<  std::flush;
	
	t+=3;
	
      }
      file << std::endl;
    }
    file.close();
    
    vis.closeWindow();

    timer.toc("Time to render and write full labeled image");
    
    vis.clearGeometry();
    
    int size_old = ID.size();
    std::sort( ID.begin(), ID.end() );
    std::vector<int>::iterator it;
    it = std::unique( ID.begin(), ID.end() );
    ID.resize( std::distance(ID.begin(), it ) );
    std::cout << ID.size() << " unique elements out of " << size_old << std::endl;

    //rectangular bounding box labels
    if( rectangularlabels ){

  sprintf(outfile,"/home/dguevara/Documents/AgML/src/agml/output/images/Image_0/rectangular_labels.txt");
      std::ofstream labelfile(outfile);
      
      for( int p=0; p<ID.size(); p++ ){
      
	if( ID.at(p)>=UUIDs_group.size() ){
	  continue;
	}

	int t=0;
	int xmin = framebufferW;
	int xmax = 0;
	int ymin = framebufferH;
	int ymax = 0;
	int pixelcount = 0;
	
	for( int j=0; j<framebufferH; j++ ){
	  for( int i=0; i<framebufferW; i++ ){
	    
	    if( rgb2int(make_RGBcolor(pixels[t]/255.f,pixels[t+1]/255.f,pixels[t+2]/255.f)) != ID.at(p) ){
	      t+=3;
	      continue;
	    }
	    
	    if( i<xmin ){
	      xmin=i;
	    }
	    if( i>xmax ){
	      xmax=i;
	    }
	    if( j<ymin ){
	      ymin=j;
	    }
	    if( j>ymax ){
	      ymax=j;
	    }
	    
	    t+=3;
	    pixelcount+=1;
	    
	  }
	}
	
	if( xmin==framebufferW || xmax==0 || ymin==framebufferH || ymax==0 ){
	  continue;
	}
	
	if( pixelcount<labelminpixels ){
	  continue;
	}
	
	labelfile << 0 << " " << (xmin+0.5*(xmax-xmin))/float(framebufferW) << " " << (ymin+0.5*(ymax-ymin))/float(framebufferH) << " " << std::setprecision(6) << std::fixed  << (xmax-xmin)/float(framebufferW) << " " << (ymax-ymin)/float(framebufferH) << std::endl;
	
      }
    }
 

    //polygon labels
    if( polygonlabels ){
      for( int p=0; p<ID.size(); p++ ){
	
        if( ID.at(p)>=UUIDs_group.size() ){
          continue;
        }
        
        vis.buildContextGeometry(&context,UUIDs_group.at(ID.at(p)));
        
        vis.setCameraPosition( camera_pos.at(view), make_vec3(camera_pos.at(view).x,0,camera_pos.at(view).z) );
        
        timer.tic();
        
        vis.plotUpdate();
        
        timer.toc("render");
        
  sprintf(outfile,"/home/dguevara/Documents/AgML/src/agml/output/images/Image_0/pixelID2_%07d.txt",ID.at(p));
        
        timer.tic();
        
        writePixelID(outfile,labelminpixels,&vis);
        
        timer.toc("image write");
        
        vis.clearGeometry();
        
            }
      
      vis.closeWindow();
    }
      
  }





    }

   }
