"""
Running this script for reweighting requires two input files:
Phi-Psi information: Nframes * Nangles * 2
E-Prop information: Nframes * 2 (E in the first column and Prop in the second)
"""
import os
import time
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, default='/lustre/home/acct-clschf/clschf/xyji/basicDihedrals/AAA_ff19SB_opc.dat')
# target (required in float type)
parser.add_argument('--target', '-t', type=str, default='./ALA.dat')
parser.add_argument('--output', '-o', type=str, default='./result.dat')
parser.add_argument('--steps', '-s', type=int, default=2000)
parser.add_argument('--optimizer', '-opt', type=str, default='adam')
parser.add_argument('--learning_rate', '-l', type=float, default=0.1)
parser.add_argument('--verbose', '-v', action='store_true', default=False)
parser.add_argument('--schedule', action='store_true', default=False)
parser.add_argument('--early_stop', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cpu')  # Currently not support CUDA
parser.add_argument('--reweight', '-r', type=str, default='./result.dat')

parser.add_argument('--weight', type=float, default=1e-2)
args, _ = parser.parse_known_args()

# Constants that may be required
invkT = -1.0 / (0.0019872041 * 300)
RT = 8.314 * 300


def main():
    phi, psi = [], []
    with open(args.input, 'r') as f:
        for lines in f.readlines():
            if lines[0] != '#':
                phi.append(float(lines.split()[1]))
                psi.append(float(lines.split()[2]))
    print(phi)
    print(psi)
    ramachandran = []
    for i in range(len(phi)):
        ramachandran.append(np.histogram2d([phi[i]], [psi[i]],
                                           bins=(24, 24),
                                           range=[[-180, 180], [-180, 180]])[0])
#    if args.reweight.endswith('.dat'):
#        np.savetxt(args.reweight, ramachandran)
    ramachandran = torch.Tensor(np.array(ramachandran)).to(args.device)
    print(ramachandran.shape) 
    ramachandran.requires_grad = True
    with open(args.target, 'r') as f:
        data = [list(map(float, line.split())) for line in f]
    data_array = np.array(data)
    Prop = torch.Tensor(data_array)
    Prop = Prop.to(args.device)
    Prop.requires_grad = True
    print(f"Prop.shape is {Prop.shape}")
    # Start training
    print('Start training...')
    model = DeepReweighting(initial_parameter=torch.zeros((1, 1, 24, 24))).to(args.device)
    CMAP_old = torch.zeros((24, 24)).to(args.device)
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate)

    min_lr = 1e-4
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=20,
                                                           factor=0.6, verbose=True,
                                                           threshold=1e-2, min_lr=min_lr, cooldown=5)
    early_stopping = EarlyStopping(patience=5)

#    cmap_record = []
#    prediction_record = []
#    step_record = []

    time0 = time.time()
    for steps in tqdm.tqdm(range(args.steps)):
        
        model.train()

        delta_E, CMAP_update = model(ramachandran)
        print(f" delta_E is {delta_E.shape}")
        losses, Prop_pred = loss(delta_E, ramachandran, Prop)
        print(f'losses is {losses}')
        # Regularization
        losses += regulization(CMAP_update, weight=args.weight)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Getting training record
        if args.verbose and (steps % 10 == 0):
            print(f'##### STEP {steps} #####')
            print(f'Step {steps}: loss {losses}\n')
        if steps == 1999:
            if args.reweight.endswith('.dat'):
                np.savetxt(args.reweight, Prop_pred.detach().numpy())
            print(f"{Prop_pred}")
        # Check training state
        if args.schedule:
            scheduler.step(losses)
            if optimizer.param_groups[0]['lr'] <= min_lr * 1.5:
                print('converged')
                break

        if args.early_stop:
            early_stopping(losses)
            if early_stopping.early_stop:
                break

        # check NaN
        if torch.isnan(model.update).any().item() or torch.isnan(Prop_pred).any().item():
            print("NaN encoutered, exiting...")
            break

#        step_record.append(steps + 1)
        CMAP_old = CMAP_update
#        print(CMAP_old.shape)
#        prediction_record.append(Prop_pred.detach().squeeze().numpy())

    print('Optimization complete, taking time: %.3f s' % (time.time() - time0))
    # Save re-weighting results
    if args.output.endswith('.dat'):
        output_file = args.output
    else:
        suffix = args.output.split('.')[-1]
        output_file = args.output.replace(suffix, 'dat')

    np.savetxt(output_file, CMAP_old.detach().numpy())
#    np.savetxt(output_file.replace('dat', 'record.dat'), np.array([step_record, prediction_record]).T)

#    print('final cross account: %.2f' % (prediction_record[-1]))

def loss(delta_E, Prop_sim, Prop_exp):
    """
    IMPORTANT: Here the loss should be pre-defined before reweighting

    :param delta_E: Delta energy of each conformation from simulation
    :param Prop_sim: Target property of conformations in simulation
    :param Prop_exp: Expected property of the system
    :return: loss value
    """
    # Reweighting
    weights = torch.exp(delta_E * invkT)
    
    # Predict Reweighted property vector
    weighted_Prop_sum = Prop_sim * weights[:, None, None]
    print(f"weighted_Prop_sum is {weighted_Prop_sum.shape}")
    Prop_pred = torch.sum(weighted_Prop_sum, dim=0) / torch.sum(weights)
#    print(f"coil is {Prop_exp}")
    # Calculate loss
    _loss = F.kl_div((Prop_pred+1e-12).log(), Prop_exp, reduction='sum')
    
    return _loss, Prop_pred


def regulization(parameter, weight=1e-2):
    """
    :param weight: loss weight
    :param parameter: The parameter to be regularized
    :return: The regularization loss
    """

    # calculate mse on parameter and zero
    parameter = parameter.view(-1) + 1e-12  # avoid zero
    print(f'parameter shape is {parameter.shape}')
    mse = torch.nn.MSELoss()
    loss = mse(parameter, torch.zeros_like(parameter))

    return weight * torch.sqrt(loss)


class DeepReweighting(nn.Module):

    def __init__(self, initial_parameter):
        super(DeepReweighting, self).__init__()

        self.update = nn.Parameter(initial_parameter, requires_grad=True)

    def forward(self, ramachandran):
        """
        :param ramachandran: The phi-psi distribution calculated on (24 x 24) grid
        :return: Energy added from CMAP for each conformation
        """

        # The input should be of shape [Batch * 24 * 24]
        output = ramachandran * self.update
        update_copy = self.update.squeeze().squeeze() * 1.0
        return torch.sum(output.squeeze(0), dim=(-2, -1)), update_copy


class EarlyStopping:
    """from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/"""

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):

        if self.best_loss is None:
            self.best_loss = val_loss

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True


def get_optimizer(opt_name, model, lr):
    if opt_name.lower() == 'adam':
        optimizer = torch.optim.Adam([model.update], lr)
    elif opt_name.lower() == 'sgd':
        optimizer = torch.optim.SGD([model.update], lr=lr, momentum=0.8)
    elif opt_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW([model.update], lr)
    elif opt_name.lower() == 'asgd':
        optimizer = torch.optim.ASGD([model.update], lr)
    elif opt_name.lower() == 'rprop':
        optimizer = torch.optim.Rprop([model.update], lr)
    elif opt_name.lower() == 'adadelta':
        optimizer = torch.optim.Adadelta([model.update], lr)
    elif opt_name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop([model.update], lr)
    else:
        raise ValueError('Optimizer <%s> currently not supported' % opt_name)

    return optimizer


if __name__ == '__main__':
    main()
