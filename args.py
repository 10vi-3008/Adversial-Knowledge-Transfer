from argparse import ArgumentParser

def create_parser():
    parser = ArgumentParser(description='ATdistill', allow_abbrev=False)
    parser.add_argument('--method', type=str, default="distill",
                        help='method')
    parser.add_argument('--alpha', type=float, default=0,
                        help='alpha')
    parser.add_argument('--beta', type=float,
                        help='beta')
    parser.add_argument('--gamma', type=float,
                        help='gamma')
    
    parser.add_argument('--epochs', type=int, default=300,
                        help='epochs')
    parser.add_argument('--batch', type=int, default=128,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', type=float, default=2e-4,
                        help='weight_decay')
    parser.add_argument('--teacher', type=str, default="WRN-70-16",
                        help='teacher model')
    parser.add_argument('--student', type=str, default="RES-18",
                        help='student model')
    parser.add_argument('--dataset', type=str, default="cifar100",
                        help='dataset')
    
    parser.add_argument('--nowand', default=1, choices=[0, 1], type=int, help='Inhibit wandb logging')
    parser.add_argument('--wandb_entity', type=str, default='wandb_name', help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='wandb_name', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default="wandb_name", help='Wandb running name')
    parser.add_argument('--wandb_tags', type=str, default="wandb_name", help='Wandb running tags')

    return parser
