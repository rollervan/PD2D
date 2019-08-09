import os


class Data():
    def __init__(self):
        self.GT = []
        self.FLAIR = []

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def get_data_list(path):
    '''
    This function returns a class Data.
    To access a pair file names: GT and Flair
    gt_filename = Data.GT[id]
    flair_filename = Data.FLAIR[id]
    where 'id' is an integer indicating the number of the subject
    :param path: path to data # path = './BRATS2015_Training/HGG/'
    :return: Data
    '''
    GT = []
    FLAIR = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            name = os.path.join(root, filename)
            if '.npy' in name:
                if 'OT' in name:
                    GT.append(name)
                    # print(name)
                if 'Flair' in name:
                    FLAIR.append(name)
                    # print(name)
    Data.GT = GT
    Data.FLAIR = FLAIR
    Data.len = len(Data.FLAIR)
    return Data






