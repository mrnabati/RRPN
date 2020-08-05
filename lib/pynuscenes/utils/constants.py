################################################################################
## Date Created  : Fri Jun 14 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : July 10th, 2019                                            ##
## Copyright (c) 2019                                                         ##
################################################################################

NUSCENES_SPLITS = {
                    'v1.0-trainval': ['train', 'val'],
                    'v1.0-mini': ['mini_train', 'mini_val'],
                    'v1.0-test': ['test']
                  }

NUSCENES_RETURNS = [
        'lidar',
        'radar',
        'camera']

CAMERAS = {'CAM_FRONT_LEFT':  0,
           'CAM_FRONT_RIGHT': 1,
           'CAM_FRONT':       2,
           'CAM_BACK_LEFT':   3,
           'CAM_BACK_RIGHT':  4,
           'CAM_BACK':        5}

RADARS = {'RADAR_FRONT_LEFT':  0,
          'RADAR_FRONT_RIGHT': 1,
          'RADAR_FRONT':       2,
          'RADAR_BACK_LEFT':   3,
          'RADAR_BACK_RIGHT':  4}

LIDARS = {'LIDAR_TOP':  0}

NAMEMAPPING = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'}

DETECTION_ID = {'car': 1, 
                'truck': 2, 
                'bus': 3,
                'trailer': 4, 
                'construction_vehicle': 5, 
                'pedestrian': 6, 
                'motorcycle': 7, 
                'bicycle': 8,
                'traffic_cone': 9,
                'barrier': 10
                }

DETECTION_NAMES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
                   'traffic_cone', 'barrier']

COCO_CLASSES = {'pedestrian': {'id': 1, 'category': 'person', 'supercategory': 'person'},
                'bicycle': {'id': 2, 'category': 'bicylce', 'supercategory': 'vehicle'},
                'car': {'id': 3, 'category': 'car', 'supercategory': 'vehicle'},
                'motorcycle': {'id': 4, 'category': 'motorcycle', 'supercategory': 'vehicle'},
                'bus': {'id': 5, 'category': 'bus', 'supercategory': 'vehicle'},
                'truck': {'id': 6, 'category': 'truck', 'supercategory': 'vehicle'}
                }

PRETTY_DETECTION_NAMES = {'car': 'Car',
                          'truck': 'Truck',
                          'bus': 'Bus',
                          'trailer': 'Trailer',
                          'construction_vehicle': 'Constr. Veh.',
                          'pedestrian': 'Pedestrian',
                          'motorcycle': 'Motorcycle',
                          'bicycle': 'Bicycle',
                          'traffic_cone': 'Traffic Cone',
                          'barrier': 'Barrier'}

DETECTION_COLORS = {'car': 'C0',
                    'truck': 'C1',
                    'bus': 'C2',
                    'trailer': 'C3',
                    'construction_vehicle': 'C4',
                    'pedestrian': 'C5',
                    'motorcycle': 'C6',
                    'bicycle': 'C7',
                    'traffic_cone': 'C8',
                    'barrier': 'C9'}

ATTRIBUTE_NAMES = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing', 'cycle.with_rider',
                   'cycle.without_rider', 'vehicle.moving', 'vehicle.parked', 'vehicle.stopped']

PRETTY_ATTRIBUTE_NAMES = {'pedestrian.moving': 'Ped. Moving',
                          'pedestrian.sitting_lying_down': 'Ped. Sitting',
                          'pedestrian.standing': 'Ped. Standing',
                          'cycle.with_rider': 'Cycle w/ Rider',
                          'cycle.without_rider': 'Cycle w/o Rider',
                          'vehicle.moving': 'Veh. Moving',
                          'vehicle.parked': 'Veh. Parked',
                          'vehicle.stopped': 'Veh. Stopped'}

TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']

PRETTY_TP_METRICS = {'trans_err': 'Trans.', 'scale_err': 'Scale', 'orient_err': 'Orient.', 'vel_err': 'Vel.',
                     'attr_err': 'Attr.'}

TP_METRICS_UNITS = {'trans_err': 'm',
                    'scale_err': '1-IOU',
                    'orient_err': 'rad.',
                    'vel_err': 'm/s',
                    'attr_err': '1-acc.'}