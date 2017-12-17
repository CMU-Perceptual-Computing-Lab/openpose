# -*- coding: utf-8 -*-

import json
from pprint import pprint

prefix='Documents/openpose/output_json/video_'
suffix = '_keypoints.json'
output_file = './output.log'
keypoints = 18
nfiles = 205

def main():
    f = open(output_file,'w+')
   
    for k in range(0,nfiles):
        path = prefix + str(k).zfill(12) + suffix
        with open(path) as data_file:
            data = json.load(data_file)
            
        
        people = len(data['people'])
        print people
        
        f.write(str(people))
        f.write('\n')
        
        for i in range(0,people):
            x = data['people'][i]['pose_keypoints']
            for v in x:
                f.write(str(v))
                f.write('\n')
                
    f.close()

if __name__ == "__main__":main() ## with if