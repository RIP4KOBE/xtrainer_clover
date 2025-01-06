from bimanual_manip_learning import BimanualManipulabilityLearning


def test():
    manipulability_model = BimanualManipulabilityLearning()
    gmm_ckpt_path = '/home/zhuoli/dobot_xtrainer/ModelTrain/manipulability/ckpt/gmm_ckpt.pth'
    xhat, sigma = manipulability_model.gmr_regression(gmm_ckpt_path, 0.1)
    print("GMR result: \n" , "xhat\n", xhat, "sigma\n", sigma)

if __name__ == '__main__':
    test()