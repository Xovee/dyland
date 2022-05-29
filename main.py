import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import lib.utils as utils
from args import add_args
from lib.dataloader import MyDataSet, load_loc_data
from lib.layers.container import MyModel, GNN
from lib.utils import count_nfe, count_total_time
from lib.utils import evaluation
from lib.utils import standard_normal_logprob, set_random_seed, x2z

SOLVERS = ["dopri5"]
parser = argparse.ArgumentParser('dyland')
add_args(parser, SOLVERS)
args = parser.parse_args()

# logger
save_path = os.path.join('results', args.data, args.log_key)
fig_save_path = os.path.join(save_path, 'figs')
utils.makedirs(save_path)
utils.makedirs(fig_save_path)
logger = utils.get_logger(logpath=os.path.join(save_path, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print("torch.cuda.is_available=" + str(torch.cuda.is_available()))
save_file = args.save_file
hyper_parameters = "seq{}hid{}pre{}_{}_{}".format(args.seq_len, args.hidden_len, args.pre_len, args.num_blocks, args.dims)


def get_transforms(cnf):
    # x->z is forward and reverse=False. z->x is inverse and reverse=True.
    # set perturbed data to zero.

    def sample_fn(z, logpz=None):
        zeros_std = torch.zeros(z.shape[0], 1).to(z)
        if logpz is not None:
            return cnf(z, zeros_std, logpz, reverse=True)
        else:
            return cnf(z, zeros_std, reverse=True)

    def density_fn(x, logpx=None):
        zeros_std = torch.zeros(x.shape[0], 1).to(x)
        if logpx is not None:
            return cnf(x, zeros_std, logpx, reverse=False)
        else:
            return cnf(x, zeros_std, reverse=False)

    return sample_fn, density_fn


def compute_loss(args, model, batch_size=None):
    """
    The main pipeline, compute loss for gradient decent.
    """
    if batch_size is None: batch_size = args.batch_size

    # load data
    x = load_loc_data(file_name=args.data, batch_size=batch_size)
    x = torch.from_numpy(x).type(torch.float32).to(device)
    # see container-MyModel
    loss1, logpz, delta_logp = model(x, args)
    # x2z: get latent representation of x.
    z = x2z(x, model)
    return loss1, logpz, delta_logp, z


if __name__ == '__main__':
    set_random_seed(args.seed)
    # load features
    feature_data_path = r"./data/{}.csv".format(args.data)
    dataset = MyDataSet(feature_data_path)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    feature_iter = iter(train_loader)
    num_nodes = dataset.num_nodes

    if args.manifold == "normal":
        # load normal locations
        feature_data_path = r"./data/{}.csv".format(args.data)
        train_data = np.loadtxt(fname=feature_data_path, delimiter=",", skiprows=1)[:, 0:3]
    else:
        # load manifold locations
        file = r"./{}/{}_MF.csv".format(args.manifold, args.data)
        train_data = np.loadtxt(fname=file, delimiter=",")
    manifold_z = torch.tensor(train_data)

    cnf = MyModel.build_model_tabular(args, args.seq_len + args.hidden_len).to(device)
    model = MyModel(args, prior=standard_normal_logprob, seq_len=args.seq_len, pre_len=args.pre_len, cnf=cnf, eps_g=None).to(device)
    gnn = GNN(args.seq_len, args.hidden_len, args).to(device)
    # load models
    if args.conti:
        ckpt_dyland = torch.load(save_path + r"\checkpt.pth")
        model.load_state_dict(ckpt_dyland['state_dict'])
        ckpt_dyland = torch.load(save_path + r"\gnn_checkpt.pth")
        gnn.load_state_dict(ckpt_dyland['state_dict'])

    # build A.
    gnn.normed_A(manifold_z, device)
    loss2 = torch.nn.MSELoss(reduce=None, size_average=None)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer1 = optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # log net info
    size = 0
    for name, parameters in model.named_parameters():
        size += int(parameters.numel())
    log_message = "Total parameters is {}.".format(size)
    logger.info(log_message)

    # writer
    writer = SummaryWriter("runs/{}_{}_{}".format(args.log_key, args.data, time.strftime("%m%d-%H-%M")))
    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    nfef_meter = utils.RunningAverageMeter(0.93)
    nfeb_meter = utils.RunningAverageMeter(0.93)
    tt_meter = utils.RunningAverageMeter(0.93)
    overall_start = time.time()
    end = time.time()
    test_best = float('inf')
    train_best = float('inf')
    early_stop = 0
    model.train()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        optimizer1.zero_grad()
        torch.cuda.empty_cache()
        # load data
        try:
            x, y = feature_iter.next()
        except StopIteration:
            feature_iter = iter(train_loader)
            x, y = feature_iter.next()
        x = x.reshape([num_nodes, args.seq_len]).to(device)
        y = y.reshape([num_nodes, args.pre_len]).to(device)

        # Feed to model
        if args.VGAE:
            W, mu, sigma = gnn(x)
        else:
            W = gnn(x)
        writer.add_scalar('Eva/W', torch.mean(W), itr)
        writer.add_scalar('Eva/mu', torch.mean(mu), itr)
        writer.add_scalar('Eva/sigma', torch.mean(sigma), itr)

        C = torch.cat((x, W), 1)

        zero_logpC = torch.zeros(x.shape[0], 1).to(C)
        std_in = torch.zeros(x.shape[0], 1).to(C)
        # input (x+eps, how noisy, zero). eps is the perturbation and std_in is how noisy the perturbation is
        Y, delta_logp = cnf(C, std_in, zero_logpC)
        delta_logp = torch.mean(delta_logp)

        # shape transformation, seq_len to pre_len
        sss = Y[:, 0:args.seq_len]
        Y_pre = model.shape_trans(sss)
        loss = loss2(input=Y_pre, target=y)

        loss_meter.update(loss.item())
        total_time = count_total_time(model.cnf)
        nfe_forward = count_nfe(model.cnf)
        loss.backward()
        optimizer.step()
        optimizer1.step()
        nfe_total = count_nfe(model)
        nfe_backward = nfe_total - nfe_forward
        nfef_meter.update(nfe_forward)
        nfeb_meter.update(nfe_backward)
        time_meter.update(time.time() - end)
        tt_meter.update(total_time)

        log_message = (
            'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | NFE Forward {:.0f}({:.1f})'
            ' | NFE Backward {:.0f}({:.1f}) | CNF Time {:.4f}({:.4f})'.format(
                itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, nfef_meter.val, nfef_meter.avg,
                nfeb_meter.val, nfeb_meter.avg, tt_meter.val, tt_meter.avg
            )
        )

        writer.add_scalar('Loss/train_val', loss_meter.val, itr)
        writer.add_scalar('Loss/train_avg', loss_meter.avg, itr)
        # wish logpz to increase, and delta_logp to decrease
        writer.add_scalar('F_loss/delta_logp', delta_logp, itr)
        writer.add_scalar('Time/val_time', time_meter.val, itr)
        with torch.no_grad():
            rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = evaluation(y.cpu().numpy(), Y_pre.cpu().numpy())
            writer.add_scalar('Loss/rmse_ts', rmse_ts, itr)
            writer.add_scalar('Loss/acc_ts', acc_ts, itr)
            writer.add_scalar('Loss/var_ts', var_ts, itr)

        if itr % args.log_freq == 0:
            logger.info(log_message)

        if loss_meter.val < train_best or loss_meter.avg < train_best:
            train_best = min(loss_meter.val, loss_meter.avg)
            early_stop = 0
        elif early_stop < args.patient:
            early_stop += 1
        else:
            log_message = "Early stop, train loss doesn't decrease for {} iters.".format(args.patient)
            logger.info(log_message)
            os._exit(0)

        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                model.eval()
                rmse_ts, mae_ts, acc_ts, r2_ts, var_ts = evaluation(y.cpu().numpy(), Y_pre.cpu().numpy())
                writer.add_scalar('TestLoss/rmse_ts', rmse_ts, itr)
                writer.add_scalar('TestLoss/mae_ts', mae_ts, itr)
                writer.add_scalar('TestLoss/acc_ts', acc_ts, itr)
                writer.add_scalar('TestLoss/r2_ts', r2_ts, itr)
                writer.add_scalar('TestLoss/var_ts', var_ts, itr)

                time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                with open(save_file, mode='a') as fin:
                    result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{}".format(rmse_ts, mae_ts, acc_ts,
                                                                                       r2_ts, var_ts,
                                                                                       time_stamp, hyper_parameters,
                                                                                       args.data, args.log_key)
                    fin.write(result)

                if rmse_ts < test_best:
                    test_best = rmse_ts
                    print("current best is:{:.6f}".format(test_best))
                    utils.makedirs(save_path)
                    torch.save({
                        'args': args,
                        'state_dict': model.state_dict(),
                    }, os.path.join(save_path, 'checkpt.pth'))
                    torch.save({
                        'args': args,
                        'state_dict': gnn.state_dict(),
                    }, os.path.join(save_path, 'gnn_checkpt.pth'))
                model.train()

        end = time.time()
    log_message = str(time.time() - overall_start)
    writer.close()
    logger.info('Training has finished. Total cost is {}.'.format(log_message))
    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(save_file, mode='a') as fin:
        result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{}".format(
            rmse_ts, mae_ts, acc_ts, r2_ts, var_ts, time_stamp, hyper_parameters, args.data, args.log_key)
        fin.write(result)
