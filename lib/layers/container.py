import torch.nn as nn
import torch
import math
from lib.layers import *


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, std, logpx=None, reverse=False, inds=None):
        """
        return the transformed x and related log_p(x). It can be bidirectional.
        """
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, std, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, std, logpx, reverse=reverse)
            return x, logpx


class EpsGenerator(nn.Module):
    def __init__(self, input_size, output_size):
        super(EpsGenerator, self).__init__()
        self.hidden1 = nn.Linear(input_size, input_size)
        self.hidden4 = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.hidden1(x)
        out = self.hidden4(out)
        out = torch.relu(out)
        return out


# class MyModel(nn.Module):
#     def __init__(self, args, cnf: SequentialFlow = None, eps_g: EpsGenerator = None):
#         """
#         Need to init cnf and eps_g first.
#         todo, eps_g is too simple.
#         """
#         super(MyModel, self).__init__()
#         if not cnf:
#             self.cnf = self.build_model_tabular(args, args.input_dim)
#         self.cnf = cnf
#         self.eps_g = eps_g
#
#     def forward(self, x, args):
#         """
#         Transform x to z, and compute loss.
#         x: torch tensor of [N, dims] shape
#
#         Compared with FFJORD https://github.com/rtqichen/ffjord/blob/master/train_toy.py
#         """
#         # if self.eps_g:
#         #     # if eps_g is defined. Note std can't be too big.
#         #     std = self.eps_g(x)
#         #     std = std / (std.max()+0.001) * args.std_max  # rescale to 0-std_max
#         #     eps = torch.randn_like(x) * std
#         #     std_in = std * args.std_weight  # scale up
#         if self.eps_g:
#             # 检验增加随机数的思路
#             std = self.eps_g(x)
#             std = std / (std.max()+0.001) * args.std_max  # rescale to 0-std_max
#             eps = torch.randn_like(x) * std
#             std_in = std * args.std_weight  # scale up
#         else:
#             # std ~ U(std_min, std_max). std is c, therefore eps~N(0, std), in eq 9 of paper, eps is v_i
#             std = (args.std_max - args.std_min) * torch.rand_like(x[:, 0]).view(-1, 1) + args.std_min
#             # rescale
#             eps = torch.randn_like(x) * std * args.eps_weight
#             std_in = std * args.std_weight
#
#         zero_logpx = torch.zeros(x.shape[0], 1).to(x)
#         # input (x+eps, how noisy, zero). eps is the perturbation and std_in is how noisy the perturbation is
#         z, delta_logp = self.cnf(x + eps, std_in, zero_logpx)
#
#         # Compared with zero input
#         # std_z = torch.zeros(x.shape[0], 1).to(x)
#         # z, delta_logp = model(x + eps, std_z, zero_logpx)
#
#         # compute log q(z)
#         logpz = self.standard_normal_logprob(z).sum(1, keepdim=True)
#         logpx = logpz - delta_logp
#         loss = -torch.mean(logpx)
#         logpz = torch.mean(logpz)
#         delta_logp = torch.mean(delta_logp)
#         # wish logpz to increase, and delta_logp to decrease
#         return loss, logpz, delta_logp
#
#     @staticmethod
#     def build_model_tabular(args, dims, regularization_fns=[]):
#         """
#         args.num_blocks: number of cnf
#         """
#         hidden_dims = tuple(map(int, args.dims.split("-")))
#
#         def build_cnf():
#             diffeq = ODEnet(
#                 hidden_dims=hidden_dims,
#                 input_shape=(dims,),
#                 strides=None,
#                 conv=False,
#                 layer_type=args.layer_type,
#                 nonlinearity=args.nonlinearity,
#             )
#             odefunc = ODEfunc(
#                 diffeq=diffeq,
#                 divergence_fn=args.divergence_fn,
#                 rademacher=args.rademacher,
#             )
#             # the func that gets input
#             cnf = CNF(
#                 odefunc=odefunc,
#                 T=args.time_length,
#                 train_T=args.train_T,
#                 regularization_fns=regularization_fns,
#                 solver=args.solver,
#             )
#             return cnf
#
#         chain = [build_cnf() for _ in range(args.num_blocks)]
#         cnf = SequentialFlow(chain)
#         return cnf
#
#     @staticmethod
#     def standard_normal_logprob(z):
#         # density of z in a standard normal distribution
#         logZ = -0.5 * math.log(2 * math.pi)
#         logZ = logZ - z.pow(2) / 2
#         return logZ

class MyModel(nn.Module):
    def __init__(self, args, prior, seq_len, pre_len, cnf: SequentialFlow = None, eps_g: EpsGenerator = None, reshape=False):
        """
        Need to init cnf and eps_g first.
        """
        super(MyModel, self).__init__()
        if not cnf:
            self.cnf = self.build_model_tabular(args, args.input_dim + args.aug_dim)
        self.cnf = cnf
        self.eps_g = eps_g
        self.prior = prior
        self.fc1 = nn.Linear(seq_len, seq_len, bias=True)
        self.fc2 = nn.Linear(seq_len, pre_len, bias=True)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.reshape = reshape

    def forward(self, x, args):
        """
        Transform x to z, and compute loss.
        x: torch tensor of [N, dims] shape

        Compared with FFJORD https://github.com/rtqichen/ffjord/blob/master/train_toy.py
        """
        # 根据不同的方法对输入进行修正，最后输出的都是变换后的标准z用于评估
        if args.aug_method == 0:
            # original SF
            # std ~ U(std_min, std_max). std is c, therefore eps~N(0, std), in eq 9 of paper, eps is v_i
            std = (args.std_max - args.std_min) * torch.rand_like(x[:, 0]).view(-1, 1) + args.std_min
            # rescale
            eps = torch.randn_like(x) * std * args.eps_weight
            std_in = std * args.std_weight

            zero_logpx = torch.zeros(x.shape[0], 1).to(x)
            # input (x+eps, how noisy, zero). eps is the perturbation and std_in is how noisy the perturbation is
            z, delta_logp = self.cnf(x + eps, std_in, zero_logpx)
        if args.aug_method == 1:
            # aug 增加随机数
            # 增加一维度的随机数，附带其标准差，其范围应该和x差不多大小
            # std ~ U(std_min, std_max). std is c, therefore eps~N(0, std), in eq 9 of paper, eps is v_i
            std = (args.std_max - args.std_min) * torch.rand_like(x[:, 0]).view(-1, 1) + args.std_min
            # rescale
            eps = torch.randn_like(x[:, :args.aug_dim]) * std
            std_in = std * args.std_weight

            zero_logpx = torch.zeros(x.shape[0], 1).to(x)
            # input (x+eps, how noisy, zero). eps is the perturbation and std_in is how noisy the perturbation is
            _x = torch.cat([x, eps], 1)
            z, delta_logp = self.cnf(_x, std_in, zero_logpx)
            # 删除非必要的部分 、或者是可以用0来代替
            # z =_z[:, :args.input_dim]
        if args.aug_method == 2:
            # aug 增加0
            # 增加一维度的随机数，附带其标准差，其范围应该和x差不多大小
            # std ~ U(std_min, std_max). std is c, therefore eps~N(0, std), in eq 9 of paper, eps is v_i
            std = (args.std_max - args.std_min) * torch.rand_like(x[:, 0]).view(-1, 1) + args.std_min
            # rescale
            eps = torch.randn_like(x[:, :args.aug_dim]) * std
            std_in = std * args.std_weight

            zero_logpx = torch.zeros(x.shape[0], 1).to(x)
            # input (x+eps, how noisy, zero). eps is the perturbation and std_in is how noisy the perturbation is
            _x = torch.cat([x, eps], 1)
            z, delta_logp = self.cnf(_x, std_in, zero_logpx)
            # 删除非必要的部分 、或者是可以用0来代替
            # z =_z[:, :args.input_dim]
        if args.aug_method == 3:
            aug = torch.zeros(x.shape[0], args.aug_dim).to(x)
            _x = torch.cat([x, aug], 1)

            # std ~ U(std_min, std_max). std is c, therefore eps~N(0, std), in eq 9 of paper, eps is v_i
            std = (args.std_max - args.std_min) * torch.rand_like(x[:, 0]).view(-1, 1) + args.std_min
            # rescale
            eps = torch.randn_like(x) * std * args.eps_weight
            std_in = std * args.std_weight

            zero_logpx = torch.zeros(x.shape[0], 1).to(x)
            # input (x+eps, how noisy, zero). eps is the perturbation and std_in is how noisy the perturbation is
            z, delta_logp = self.cnf(x + eps, std_in, zero_logpx)

        # Compared with zero input
        # std_z = torch.zeros(x.shape[0], 1).to(x)
        # z, delta_logp = model(x + eps, std_z, zero_logpx)
        if self.reshape:
            z = self.shape_trans(z)

        # compute log q(z)
        logpz = self.prior(z).sum(1, keepdim=True)
        logpx = logpz - delta_logp
        loss = -torch.mean(logpx)
        logpz = torch.mean(logpz)
        delta_logp = torch.mean(delta_logp)
        # wish logpz to increase, and delta_logp to decrease
        return loss, logpz, delta_logp

    @staticmethod
    def build_model_tabular(args, dims, regularization_fns=[]):
        """
        args.num_blocks: number of cnf
        """
        hidden_dims = tuple(map(int, args.dims.split("-")))

        def build_cnf():
            diffeq = ODEnet(
                hidden_dims=hidden_dims,
                input_shape=(dims,),
                strides=None,
                conv=False,
                layer_type=args.layer_type,
                nonlinearity=args.nonlinearity,
            )
            odefunc = ODEfunc(
                diffeq=diffeq,
                divergence_fn=args.divergence_fn,
                rademacher=args.rademacher,
            )
            # the func that gets input
            cnf = CNF(
                odefunc=odefunc,
                T=args.time_length,
                train_T=args.train_T,
                regularization_fns=regularization_fns,
                solver=args.solver,
            )
            return cnf

        chain = [build_cnf() for _ in range(args.num_blocks)]
        cnf = SequentialFlow(chain)
        return cnf

    def shape_trans(self, x):
        # shape of S to target or U to 2d
        # todo, how to trans?
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return x


class GNN(nn.Module):
    # 每一组的数据都是 N * dim_in的
    def __init__(self, dim_in, dim_out, args):
        super(GNN, self).__init__()
        self.vgae = args.VGAE
        self.varrho = args.varrho
        self.A = None
        self.fc1 = nn.Linear(dim_in, dim_in, bias=False)
        self.fc2 = nn.Linear(dim_in, dim_in, bias=False)
        self.fc3 = nn.Linear(dim_in, dim_out, bias=False)
        if self.vgae:
            self.fc4 = nn.Linear(dim_in, dim_in, bias=False)
            self.fc5 = nn.Linear(dim_in, dim_in, bias=False)
            self.fc6 = nn.Linear(dim_in, dim_out, bias=False)
        self.remain_persent = None

    def forward(self, X, z=None):
        # X 是N * dim_in的特征, 输出N * dim_out
        if z is not None:
            self.normed_A(z)
            print("Sparsity of A (0 is empty):", self.remain_persent * 100, "%")
            print("Cut persent:", 100 - self.remain_persent * 100, "%")
        if self.vgae:
            mu = torch.nn.functional.relu(self.fc1(self.A.mm(X)))
            mu = torch.nn.functional.relu(self.fc2(self.A.mm(mu)))
            mu = self.fc3(self.A.mm(mu))

            sigma = torch.nn.functional.relu(self.fc4(self.A.mm(X)))
            sigma = torch.nn.functional.relu(self.fc5(self.A.mm(sigma)))
            sigma = self.fc6(self.A.mm(sigma))

            out = mu + torch.exp(sigma) * torch.randn_like(sigma)
            return out, mu, torch.exp(sigma)
        else:
            mu = torch.nn.functional.relu(self.fc1(self.A.mm(X)))
            X = torch.nn.functional.relu(self.fc2(self.A.mm(mu)))
            out = self.fc3(self.A.mm(X))
            return out

    def normed_A(self, z, device):
        # z 是N*3的原始坐标，varrho是限制，生成可以直接使用的A和保留比例
        Dis = torch.cdist(z, z, p=2).float()  # distance matrix
        m = Dis.mean()

        # 不带权A
        A = torch.eye(Dis.size(0), dtype=torch.float32)
        # A[Dis < self.varrho] = 1
        A[Dis < m] = 1
        self.remain_persent = (A.sum() / (A.shape[0] * A.shape[1])).item()
        print("Sparsity of A (0 is empty):", self.remain_persent * 100, "%")
        print("Cut persent:", 100 - self.remain_persent * 100, "%")
        # 带权A
        # A = torch.where(Dis < m, Dis, torch.eye(Dis.size(0), dtype=torch.float32)).float()
        # degree matrix
        degree = A.sum(1)
        D = torch.diag(torch.pow(degree, -0.5))
        normed_A = D.mm(A).mm(D)
        self.A = normed_A.to(device)


class Fusion(nn.Module):
    def __init__(self, args, device, prior):
        """
        Need to init cnf and eps_g first.
        """
        super(Fusion, self).__init__()
        self.args = args
        self.cnf1 = MyModel.build_model_tabular(args, args.input_dim).to(device)
        self.model1 = MyModel(args, prior=prior, seq_len=args.seq_len, pre_len=args.pre_len, cnf=self.cnf1,
                         eps_g=None).to(device)
        self.gcnmu = GCN(args.seq_len, args.hidden_len).to(device)
        if args.VAGE:
            self.gcnsigma = GCN(args.seq_len, args.hidden_len).to(device)
        self.loss2 = torch.nn.MSELoss(reduce=None, size_average=None)
        self.cnf = MyModel.build_model_tabular(args, args.seq_len + args.hidden_len).to(device)
        self.model = MyModel(args, prior=prior, seq_len=args.seq_len, pre_len=args.pre_len, cnf=self.cnf,
                        eps_g=None).to(device)

    def forward(self, x, args):
        return 1

    def get_models(self):
        if self.args.VAGE:
            return self.model1, self.model, self.gcnmu, self.gcnsigma
        else:
            return self.model1, self.model, self.gcnmu
