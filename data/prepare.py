import os
import json
import numpy as np

VideoAnnotationPath_training = "/home1/huangwencan/training"
VideoAnnotationPath_validation = "/home1/huangwencan/validation"
VidorFeaturePath = "/home1/zhaoyang/dataset/vidor_feature"
VidorAnnotationPath = "/home1/zhaoyang/dataset/vidor_annotation"
CSVPath = "/home1/zhaoyang/dataset/vidor_csv"

def get_invertdict1():
    res = []
    for root_path in [VideoAnnotationPath_training, VideoAnnotationPath_validation]:
        d1 = {}
        dirs = os.listdir(root_path)
        for dir in dirs:
            curp = os.path.join(root_path, dir)
            files = os.listdir(curp)
            for file in files:
                d1[file.split('.')[0]] = os.path.join(curp, file)
        res.append(d1)
    return res


def get_invertdict2():
    d2 = {}
    dirs = os.listdir(VidorFeaturePath)
    for dir in dirs:
        curp = os.path.join(VidorFeaturePath, dir)
        files = os.listdir(curp)
        for file in files:
            d2[file] = os.path.join(curp, file)
    return d2


VideoAnnotationInvertDict_training, VideoAnnotationInvertDict_validation = get_invertdict1()
VidorFeatureInvertDict = get_invertdict2()

f = open(os.path.join(CSVPath, 'result.csv'), 'r')
data_qa = f.readlines()
f.close()

all_answers = []

res = []
for row in data_qa[1:]:
    id, video_id, \
    subject_tid, object_tid, predicate, part, \
    caption, question_1, answer_1, \
    question_2, answer_2 = row.split(',')
    cur = {}
    assert (video_id in VideoAnnotationInvertDict_training)
    assert (video_id in VidorFeatureInvertDict)

    d = json.load(open(VideoAnnotationInvertDict_training[video_id]))
    clips = []
    clip = [np.inf, -1]
    for i in d['relation_instances']:
        if i['subject_tid']==subject_tid and i['object_tid']==object_tid and i['predicate']==predicate:
            clips.append([i['begin_fid'],i['end_fid']])
            if i['begin_fid'] < clip[0]:
                clip[0] = i['begin_fid']
            if i['end_fid'] > clip[1]:
                clip[1] = i['end_fid']
    assert (len(clips) > 0)

    cur['video_feature_path'] = VidorFeatureInvertDict[video_id]
    cur['clip'] = clip
    cur['clips'] = clips
    cur['question_1'] = question_1
    cur['answer_1'] = answer_1
    cur['question_2'] = question_2
    cur['answer_2'] = answer_2
    all_answers.append(answer_1)
    all_answers.append(answer_2)

    sub_traj = []
    sub_traj_fid = []
    ob_traj = []
    ob_traj_fid = []
    for cli in clips:
        for fid in range(cli[0], cli[1]):
            for traj_dic in d['trajectories'][fid]:
                tmp = ['xmin', 'ymin', 'xmax', 'ymax']
                if traj_dic['tid']==int(subject_tid):
                    sub_traj.append([traj_dic['bbox'][k] for k in tmp])
                    sub_traj_fid.append(fid)
                elif traj_dic['tid']==int(object_tid):
                    ob_traj.append([traj_dic['bbox'][k] for k in tmp])
                    ob_traj_fid.append(fid)
    cur['subject_trajectory'] = sub_traj
    cur['subject_trajectory_fid'] = sub_traj_fid
    cur['object_trajectory'] = ob_traj
    cur['object_trajectory_fid'] = ob_traj_fid

    cur['video_frame_count'] = d['frame_count']
    cur['video_fps'] = d['fps']
    cur['video_time_length'] = cur['video_frame_count']/cur['video_fps']

    res.append(cur)

f = open('data.json', 'w')
f.write(str(res))
f.close()

