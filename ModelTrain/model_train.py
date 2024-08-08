import argparse
from module.train_module import train

def arg_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', default='./ckpt/ckpt_move_cube_new',required=False)
    parser.add_argument('--task_name', action='store', type=str, default='move_cube_new',help='task_name', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=16, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0,required=False)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', default=30000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=2e-5,required=False)
    parser.add_argument('--load_pretrain', action='store_true', default=False)  # Ignore this parameter and leave the default setting
    parser.add_argument('--eval_every', action='store', type=int, default=100, help='eval_every', required=False)  # Ignore this parameter and leave the default setting
    parser.add_argument('--validate_every', action='store', type=int, default=100, help='validate_every',required=False)
    parser.add_argument('--save_every', action='store', type=int, default=1000, help='save_every', required=False)
    # parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', default= "./ckpt/ckpt_closh/ckpt_closh_puping2_new10000/policy_last.ckpt",required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')

    parser.add_argument('--kl_weight', action='store', type=int, help='KL divergence weight,recommended set 10 or 100', default=10,required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='The model predicts the length of the output action sequence at a time', default=45,required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200,required=False)
    parser.add_argument('--temporal_agg', action='store_true', default=True)
    parser.add_argument('--no_encoder', action='store_true', default=False)

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = arg_config()
    train(args)
