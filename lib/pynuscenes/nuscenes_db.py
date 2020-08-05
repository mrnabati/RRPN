#!/usr/bin/env python3
################################################################################
## Date Created  : July 6th, 2019                                             ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : July 12th, 2019                                            ##
## Copyright (c) 2019                                                         ##
################################################################################

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import Box
import threading
import queue
import pickle
from multiprocessing import RLock, Pool, freeze_support
import multiprocessing
import os
import logging
from pynuscenes.utils import constants
from pynuscenes.utils import logging
import time

class NuscenesDB(object):
    """
    Token database for the nuscenes dataset
    """
    
    def __init__(self,
                 nusc_root,
                 nusc_version,
                 split,
                 max_cam_sweeps=6,
                 max_lidar_sweeps=10,
                 max_radar_sweeps=6,
                 logging_level="INFO",
                 logger=None,
                 nusc=None):
        """
        Image database object that holds the sample data tokens for the nuscenes
        dataset.
        :param root_path: location of the nuscenes dataset
        :param nusc_version: the version of the dataset to use ('v1.0-trainval', 
                             'v1.0-test', 'v1.0-mini')
        :param max_cam_sweeps: number of sweep tokens to return for each camera
        :param max_lidar_sweeps: number of sweep tokens to return for lidar
        :param max_radar_sweeps: number of sweep tokens to return for each radar
        """

        self.nusc_root = nusc_root
        self.nusc_version = nusc_version
        self.split = split
        self.max_cam_sweeps = max_cam_sweeps
        self.max_lidar_sweeps = max_lidar_sweeps
        self.max_radar_sweeps = max_radar_sweeps
        self.id_length = 8
        self.db = {}
        
        assert nusc_version in constants.NUSCENES_SPLITS.keys(), \
            "Nuscenes version not valid."
        assert split in constants.NUSCENES_SPLITS[nusc_version], \
            "Nuscenes split ({}) is not valid for {}".format(split, nusc_version)

        if logger is None:
            self.logger = logging.initialize_logger('pynuscenes', logging_level)
        else:
            self.logger = logger
            
        if nusc is not None:
            if self.nusc.version != nusc_version:
                self.logger.info('Loading nuscenes {} dataset'.format(nusc_version))
                self.nusc = NuScenes(version=nusc_version, dataroot=self.nusc_root,
                                     verbose=True)
            else:
                self.nusc = nusc
        else:
            self.logger.info('Loading nuscenes {} dataset'.format(nusc_version))
            self.nusc = NuScenes(version=nusc_version, dataroot=self.nusc_root, 
                                 verbose=True)
        
        self.SENSOR_NAMES = [x['channel'] for x in self.nusc.sensor]

    ##--------------------------------------------------------------------------
    def generate_db(self, out_dir=None) -> None:
        """
        Create an image databaser (db) for the NuScnenes dataset and save it
        to a pickle file
        """
        startTime = time.time()
        self.logger.info('Creating DATABASE for {} {} dataset ...'.format(self.nusc_version, self.split))
        scenes_list = self._split_scenes()
        frames = self._get_frames(scenes_list)
        metadata = {"version": self.nusc_version}
        self.db = {
                    'frames': frames,
                    'metadata': metadata
                    }
        self.logger.info('Done in %.3fs' % (time.time()-startTime))
        self.logger.info('Number of samples in split: {}'.format(str(len(frames))))
        ## if an output directory is specified, write to a pkl file
        if out_dir is not None:
            self.logger.info('Writing pickle file at {}'.format(db_filename))
            
            out_dir = os.path.join(out_dir, self.nusc_version)
            os.mkdirs(out_dir, exist_ok=True)
            db_filename = "{}_db.pkl".format(self.split)
            with open(os.path.join(out_dir, db_filename), 'wb') as f:
                pickle.dump(self.db['test'], f)
            
    ##--------------------------------------------------------------------------
    def _split_scenes(self) -> None:
        """
        Split scenes into train, val and test scenes
        """
        scene_split_names = splits.create_splits_scenes()
        scenes_list = []        
        for scene in self.nusc.scene:
            #NOTE: mini train and mini val are subsets of train and val
            if scene['name'] in scene_split_names[self.split]:
                scenes_list.append(scene['token'])

        self.logger.debug('{}: {} scenes'.format(self.nusc_version, 
                          str(len(scenes_list))))

        return scenes_list
    ##------------------------------------------------------------------------------
    def _get_frames(self, scenes_list) -> list:
        """
        returns (train_nusc_frames, val_nusc_frames) from the nuscenes dataset
        """
        self.sample_id = 0

        self.logger.debug('Generating train frames')
        frames = []
        for scene in scenes_list:
            frames = frames + self.process_scene_samples(scene)

        return frames

    ##--------------------------------------------------------------------------
    def process_scene_samples(self, scene: str) -> list:
        """
        Get sensor data and annotations for all samples in the scene.
        :param scene: scene token
        return samples: a list of dictionaries
        frame (a dictionary with a sample, sweeps)
        """
        scene_rec = self.nusc.get('scene', scene)
        scene_number = scene_rec['name'][-4:]
        self.logger.debug('Processing scene {}'.format(scene_number))

        ## Get the first sample token in the scene
        sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
        sample_sensor_records = {x: self.nusc.get('sample_data',
            sample_rec['data'][x]) for x in self.SENSOR_NAMES}

        ## Loop over all samples in the scene
        returnList = []
        has_more_samples = True
        while has_more_samples:
            sample = {}
            # sample = {
            #           CAM_FRONT_LEFT: token
            #           CAM_FRONT_RIGHT: token
            #           ...
            #           RADAR_BACK_RIGHT: token
            #          }
            sample.update({cam: sample_sensor_records[cam]['token'] for cam in constants.CAMERAS.keys()})
            sample.update({'LIDAR_TOP': sample_sensor_records['LIDAR_TOP']['token']})
            sample.update({x: sample_sensor_records[x]['token'] for x in constants.RADARS.keys()})

            frame = {'sample': sample,
                     'sweeps': self._get_sweeps(sample_sensor_records),
                     'id': str(self.sample_id).zfill(self.id_length)}
            self.sample_id += 1

            ## Get the next sample if it exists
            if sample_rec['next'] == "":
                has_more_samples = False
            else:
                sample_rec = self.nusc.get('sample', sample_rec['next'])
                sample_sensor_records = {x: self.nusc.get('sample_data',
                    sample_rec['data'][x]) for x in self.SENSOR_NAMES}
            returnList.append(frame)
        return returnList

    ##------------------------------------------------------------------------------
    def _get_sweeps(self, sweep_sensor_records) -> dict:
        """
        :param sweep_sensor_records: list of sample data records for the sensors 
        to return sweeps for.
        :return: dictionary of lists
            key is the sensor name
            value is the list sweep tokens
        """
        sweep = {x: '' for x in self.SENSOR_NAMES}
        lidar_sweeps = self._get_previous_sensor_sweeps(sweep_sensor_records['LIDAR_TOP'], 
                                                        self.max_lidar_sweeps)

        ## if the LIDAR has no previous sweeps, we assume this is the first sample
        if lidar_sweeps == []:
            return {}

        sweep.update({'LIDAR_TOP': lidar_sweeps})

        for cam in constants.CAMERAS.keys():
            cam_sweeps = {cam: self._get_previous_sensor_sweeps(sweep_sensor_records[cam], 
                          self.max_cam_sweeps)}
            sweep.update({cam: cam_sweeps[cam]})

        for radar in constants.RADARS.keys():
            radar_sweeps = {radar: self._get_previous_sensor_sweeps(sweep_sensor_records[radar], 
                            self.max_radar_sweeps)}
            sweep.update({radar: radar_sweeps[radar]})

        return sweep

    ##------------------------------------------------------------------------------
    def _get_previous_sensor_sweeps(self, sample_data, num_sweeps) -> list: 
        """
        Gets the previous sweeps for one senser
        :param sample_data: sample_data dictionary for the sensor
        :param num_sweeps: number of sweeps to return
        """
        sweeps = []
        while len(sweeps) < num_sweeps:
            if not sample_data['prev'] == "":
                sweeps.append(sample_data['prev'])
                sample_data = self.nusc.get('sample_data', sample_data['prev'])
            else:
                break
        return sweeps

################################################################################

if __name__ == "__main__":
    test_db()
