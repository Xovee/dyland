import lib.layers.odefunc as odefunc


def add_args(parser, SOLVERS):
    parser.add_argument("-f", type=str, default="")
    parser.add_argument("--layer_type", type=str, default="concatsquash", choices=["concatsquash"])
    parser.add_argument('--time_length', type=float, default=0.5)
    parser.add_argument('--train_T', type=eval, default=True)
    parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
    parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
    parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
    parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=1234, help='Seed for initializing training. ')
    parser.add_argument('--patient', type=int, default=50, help='Patient for early stop.')
    # batch size not only depends on memory, but data size.
    parser.add_argument('--aug_dim', type=int, default=0, help="dim of input, must be set")
    parser.add_argument('--aug_method', type=int, default=0, help="dim of input, must be set")
    parser.add_argument('--eps_g', type=eval, default=False, help="Whether to use eps generator or eps from uniform")
    parser.add_argument('--special_design', type=eval, default=False, help="special")
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--test_batch_size', type=int, default=3000)
    parser.add_argument('--visualize_batch_size', type=int, default=3000)
    # eps range
    parser.add_argument('--std_min', type=float, default=0.0)
    parser.add_argument('--std_max', type=float, default=0.1)
    # rescale random variables
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--std_weight', type=float, default=2)
    parser.add_argument('--eps_weight', type=float, default=0.1)
    parser.add_argument('--viz_freq', type=int, default=100, help="visualize the flow1 shape")
    parser.add_argument('--val_freq', type=int, default=1, help="frequency of saving")
    parser.add_argument('--niters', type=int, default=10)
    parser.add_argument('--log_freq', type=int, default=10)

    # main parameters
    parser.add_argument('--input_dim', type=int, default=3, help="dim of input, must be set")
    parser.add_argument('--seq_len', type=int, default=3, help="len of S")
    parser.add_argument('--hidden_len', type=int, default=3, help="len of W")
    parser.add_argument('--pre_len', type=int, default=1, help="len of Y")
    parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
    parser.add_argument('--dims', type=str, default='64-64-64')
    parser.add_argument('--save_file', type=str, default=r".\results\pure_result.csv")
    parser.add_argument('--varrho', type=float, default=5.0, help='Bigger rho makes denser A.')
    parser.add_argument('--VGAE', type=eval, default=True, help="VGAE")
    parser.add_argument('--fusion', type=eval, default=True, help="fusion training")

    # Most important args
    parser.add_argument('--conti', type=eval, default=False, help="Whether to use eps generator or eps from uniform")
    parser.add_argument(
        '--data',
        choices=['hzy_east', 'hzy_west', 'pbg_east', 'pbg_west'],
        type=str, default='pbg_east'
    )
    parser.add_argument('--log_key', type=str, default="dyland", help="The tensorboardx log dir")
    parser.add_argument('--manifold', type=str, default="normal", help="The adjacency matrix")

