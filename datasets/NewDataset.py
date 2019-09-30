import os
import numpy as np
from torch.utils.data import Dataset

from utils import load_feature, tokenize


def get_video_feature(video_feature_path):
    files = os.listdir(video_feature_path)
    feature = [np.load(os.path.join(video_feature_path, file)) for file in files]
    return np.vstack(feature)


class NewDataset(Dataset):
    def __init__(self, data, word2vec, is_training=True):
        self.data = data
        self.is_training = is_training
        self.word2vec = word2vec

    def __getitem__(self, index):
        row_index = index//2
        id = index%2 + 1

        label = np.asarray(self.data[row_index]['clip']).astype(np.int32)

        sentence = self.data[row_index]['question_'+str(id)]
        words = tokenize(sentence, self.word2vec)
        words_vec = np.asarray([self.word2vec[word] for word in words])
        words_vec = words_vec.astype(np.float32)

        video_feature_path = self.data[row_index]['video_feature_path']
        feats = get_video_feature(video_feature_path)

        assert (feats.shape[0]==self.data[row_index]['video_frame_count']), "feats.shape[0]!=video_frame_count"

        return feats, words_vec, label

    def __len__(self):
        return len(self.data) * 2
