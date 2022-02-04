from agml._internal import syntheticdata

hdg = syntheticdata.HeliosDataGenerator() 

#hdg.canopy_param_ranges['VSPGrapevine']['canopy_origin'] = [[-1.0, 0.0], [0.5, 1.1], [0.2, 0.5]]

#---------------- CAMERA PARAMETERS --------------------
hdg.camera_param_ranges['camera_position'] = [[0,-4,1],[1,-4,1]]
hdg.camera_param_ranges['camera_lookat'] = [[0,0,1],[1,0,1]]
hdg.camera_param_ranges['image_resolution'] = [500, 500]

# for k in ['GobletGrapevine',  'SplitGrapevine',  'UnilateralGrapevine',  'VSPGrapevine']:
#     print(k)
#     # ---------------- LiDAR PARAMETERS --------------------
# hdg.lidar_param_ranges['ASCII_format'] = ['x y z object_label']
# hdg.lidar_param_ranges['thetaMax'] = [180]
# hdg.lidar_param_ranges['thetaMin'] = [30]
# #hdg.lidar_param_ranges['exitDiameter'] = [0.05]
# hdg.lidar_param_ranges['size'] = [250, 450]
# hdg.lidar_param_ranges['origin'] = [[-0.5, 1, 0.5],[0.5, 1, 0.5],[-0.5, -1, 0.5],[0.5, -1, 0.5]]
#     # ---------------- CANOPY PARAMETERS -------------------
hdg.canopy_param_ranges['SplitGrapevine']['plant_count'] = [[1, 1],[1, 1]]
hdg.canopy_param_ranges['SplitGrapevine']['plant_height'] = [[1, 2]]
hdg.canopy_param_ranges['SplitGrapevine']['canopy_origin'] = [[0, 0, 0]]
#     hdg.canopy_param_ranges[k]['leaf_spacing_fraction'] = [[0.3, 0.8]]
#     hdg.canopy_param_ranges[k]['leaf_subdivisions'] = [[1, 5], [1, 5]]
#     hdg.canopy_param_ranges[k]['leaf_width'] = [[0.1, 0.3]]
#     hdg.canopy_param_ranges[k]['grape_color'] = [[0.15, 0.20], [0.15, 0.25], [0.2, 0.3]]
#     hdg.canopy_param_ranges[k]['cluster_radius'] = [[0.025, 0.035]] 
#     # ---------------- GENERATE SYNTHETIC DATA -------------
hdg.generate_data(n_imgs=2, canopy_type='SplitGrapevine', annotation_type = 'semantic', simulation_type='rgb', label_elements='leaves, branches', output_directory='/home/dariojavo/Documents/UCDavis/AgML-dev') #object instance