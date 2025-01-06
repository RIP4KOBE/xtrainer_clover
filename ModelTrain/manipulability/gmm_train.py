from bimanual_manip_learning import BimanualManipulabilityLearning

def train():
    data_path = '/home/zhuoli/dobot_xtrainer/ModelTrain/dp/split_data/collect_data_test'
    manipulability_model = BimanualManipulabilityLearning()
    manipulability_model.load_data_and_generate_ellipsoids(data_path)
    gmm_ckpt_path = '/home/zhuoli/dobot_xtrainer/ModelTrain/manipulability/ckpt/gmm_ckpt.pth'
    manipulability_model.gmm_learning(gmm_ckpt_path)

if __name__ == '__main__':
    train()