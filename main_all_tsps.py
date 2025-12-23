import os
import json
import argparse
import time
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.utils.class_weight import compute_class_weight
from tensorboardX import SummaryWriter
from fastprogress import master_bar, progress_bar
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)
from config import *
from utils.graph_utils import *
from utils.google_tsp_reader import GoogleTSPReader, VariableTSPReader
from utils.plot_utils import *
from models.gcn_model import ResidualGatedGCNModel
from utils.model_utils import *



parser = argparse.ArgumentParser(description='gcn_tsp_parser')
parser.add_argument('-c','--config', type=str, default="configs/euc_2d.json")
args = parser.parse_args()
config_path = args.config


config = get_config(config_path)
log_dir = f"./logs/{config.expt_name}/"
os.makedirs(log_dir, exist_ok=True)

print("Loaded {}:\n{}".format(config_path, config))




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)  



if torch.cuda.is_available():
    print("CUDA available, using GPU ID {}".format(config.gpu_id))
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print("CUDA not available")
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

# # Test data loading




batch_size = config.batch_size
train_filepath = config.train_filepath

# Agora o reader não precisa de num_nodes nem knn_ratio
dataset = VariableTSPReader(train_filepath, batch_size,  knn_ratio=config.knn_ratio, iteration_mode=config.iteration_mode, metric=config.metric)

# Conta quantos batches existem por tamanho
batch_info = {
    str(N): int(dataset.num_batches[N])
    for N in dataset.sizes
}

batch_info_path = os.path.join(log_dir, "batch_counts_by_N.json")

with open(batch_info_path, "w") as f:
    json.dump(batch_info, f, indent=2)

print("Saved batch counts by N to:", batch_info_path)

# Gera um batch (pode ser N diferente a cada execução)
t = time.time()
batch = next(iter(dataset))
print("Batch generation took: {:.3f} sec".format(time.time() - t))

# Descobre automaticamente o N do batch
num_nodes = batch.nodes.shape[1]
print(f"Batch node count (N): {num_nodes}")

# Print shapes
print("edges:", batch.edges.shape)
print("edges_values:", batch.edges_values.shape)
print("edges_targets:", batch.edges_target.shape)
print("nodes:", batch.nodes.shape)
print("nodes_target:", batch.nodes_target.shape)
print("nodes_coord:", batch.nodes_coord.shape)
print("tour_nodes:", batch.tour_nodes.shape)
print("tour_len:", batch.tour_len.shape)

# Plotar um exemplo da batch
idx = 0  # escolha outro índice se quiser
f = plt.figure(figsize=(5, 5))
a = f.add_subplot(111)
plot_tsp(
    a,
    batch.nodes_coord[idx],
    batch.edges[idx],
    batch.edges_values[idx],
    batch.edges_target[idx],
)
f.savefig(
    os.path.join(log_dir, f"tsp_example_N{batch.nodes.shape[1]}.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close(f)


# # Instantiate model

# In[40]:



# Instantiate the network
net = nn.DataParallel(ResidualGatedGCNModel(config, dtypeFloat, dtypeLong))
if torch.cuda.is_available():
    net.cuda()
print(net)

# Compute number of network parameters
nb_param = 0
for param in net.parameters():
    nb_param += np.prod(list(param.data.size()))
print('Number of parameters:', nb_param)

# Define optimizer
learning_rate = config.learning_rate
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
print(optimizer)




def train_one_epoch(net, optimizer, config, master_bar):
    # Set training mode
    net.train()

    # Assign parameters



    batch_size = config.batch_size
    batches_per_epoch = config.batches_per_epoch
    accumulation_steps = config.accumulation_steps
    train_filepath = config.train_filepath
    loss_by_size = {} 
    # Load TSP data
    dataset = VariableTSPReader(train_filepath, batch_size,  knn_ratio=config.knn_ratio, iteration_mode=config.iteration_mode, metric=config.metric)
    if batches_per_epoch != -1:
        batches_per_epoch = min(batches_per_epoch, dataset.max_iter)
    else:
        batches_per_epoch = dataset.max_iter

    # Convert dataset to iterable
    dataset = iter(dataset)
    
    # Initially set loss class weights as None
    edge_cw = None

    # Initialize running data
    running_loss = 0.0
    # running_err_edges = 0.0
    # running_err_tour = 0.0
    # running_err_tsp = 0.0
    running_pred_tour_len = 0.0
    running_gt_tour_len = 0.0
    running_nb_data = 0
    running_nb_batch = 0

    start_epoch = time.time()
    for batch_num in progress_bar(range(batches_per_epoch), parent=master_bar):
        # Generate a batch of TSPs
        try:
            batch = next(dataset)
        except StopIteration:
            break

        # Convert batch to torch Variables
        x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
        x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
        x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
        x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
        y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
        y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)
        
        # Compute class weights (if uncomputed)
        edge_labels = y_edges.cpu().numpy().flatten()
        edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
        
        # Forward pass
        y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
        loss = loss.mean()  # Take mean of loss across multiple GPUs
        loss = loss / accumulation_steps  # Scale loss by accumulation steps
        loss.backward()

        N = x_nodes.shape[1]   # número de nós deste batch
        if N not in loss_by_size:
            loss_by_size[N] = []

        loss_by_size[N].append(loss.item())
        # Backward pass
        if (batch_num+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Compute error metrics and mean tour lengths
        # err_edges, err_tour, err_tsp, tour_err_idx, tsp_err_idx = edge_error(y_preds, y_edges, x_edges)
        pred_tour_len = mean_tour_len_edges(x_edges_values, y_preds)
        gt_tour_len = np.mean(batch.tour_len)

        # Update running data
        running_nb_data += batch_size
        running_loss += batch_size* loss.data.item()* accumulation_steps  # Re-scale loss
        # running_err_edges += batch_size* err_edges
        # running_err_tour += batch_size* err_tour
        # running_err_tsp += batch_size* err_tsp
        running_pred_tour_len += batch_size* pred_tour_len
        running_gt_tour_len += batch_size* gt_tour_len
        running_nb_batch += 1
        
        # Log intermediate statistics
        result = ('loss:{loss:.4f} pred_tour_len:{pred_tour_len:.3f} gt_tour_len:{gt_tour_len:.3f}'.format(
            loss=running_loss/running_nb_data,
            pred_tour_len=running_pred_tour_len/running_nb_data,
            gt_tour_len=running_gt_tour_len/running_nb_data))
        master_bar.child.comment = result

    # Compute statistics for full epoch
    loss = running_loss/ running_nb_data
    err_edges = 0 # running_err_edges/ running_nb_data
    err_tour = 0 # running_err_tour/ running_nb_data
    err_tsp = 0 # running_err_tsp/ running_nb_data
    pred_tour_len = running_pred_tour_len/ running_nb_data
    gt_tour_len = running_gt_tour_len/ running_nb_data

    return time.time()-start_epoch, loss, err_edges, err_tour, err_tsp, pred_tour_len, gt_tour_len, loss_by_size

def metrics_to_str(epoch, time, learning_rate, loss, err_edges, err_tour, err_tsp,
                   pred_tour_len, gt_tour_len, loss_by_size=None):

    result = (
        'epoch:{epoch:0>2d}\t'
        'time:{time:.1f}h\t'
        'lr:{learning_rate:.2e}\t'
        'loss:{loss:.4f}\t'
        'pred_tour_len:{pred_tour_len:.3f}\t'
        'gt_tour_len:{gt_tour_len:.3f}'
    ).format(
        epoch=epoch,
        time=time/3600,
        learning_rate=learning_rate,
        loss=loss,
        pred_tour_len=pred_tour_len,
        gt_tour_len=gt_tour_len
    )

    # === FIX: se o valor for lista → tira a média ===
    if loss_by_size is not None:
        parts = []
        for N in sorted(loss_by_size.keys()):
            val = loss_by_size[N]
            if isinstance(val, list):
                if len(val) > 0:
                    v = sum(val)/len(val)
                else:
                    v = float('nan')
            else:
                v = float(val)

            parts.append(f"N{N}:{v:.4f}")

        result += "\t(loss_by_size: " + " ".join(parts) + ")"

    return result



def test(net, config, master_bar, mode='test'):
    # Set evaluation mode
    net.eval()

    # Assign parameters


    batch_size = config.batch_size
    batches_per_epoch = config.batches_per_epoch
    beam_size = config.beam_size
    val_filepath = config.val_filepath
    test_filepath = config.test_filepath

    # Load TSP data
    if mode == 'val':
        dataset = VariableTSPReader(val_filepath, batch_size,  knn_ratio=config.knn_ratio, iteration_mode=config.iteration_mode, metric=config.metric)
    elif mode == 'test':
        dataset = VariableTSPReader(test_filepath, batch_size,  knn_ratio=config.knn_ratio, iteration_mode=config.iteration_mode, metric=config.metric)
    batches_per_epoch = dataset.max_iter

    # Convert dataset to iterable
    dataset = iter(dataset)
    
    # Initially set loss class weights as None
    edge_cw = None

    # Initialize running data
    running_loss = 0.0
    # running_err_edges = 0.0
    # running_err_tour = 0.0
    # running_err_tsp = 0.0
    running_pred_tour_len = 0.0
    running_gt_tour_len = 0.0
    running_nb_data = 0
    running_nb_batch = 0
    
    with torch.no_grad():
        start_test = time.time()
        for batch_num in progress_bar(range(batches_per_epoch), parent=master_bar):
            # Generate a batch of TSPs
            try:
                batch = next(dataset)
            except StopIteration:
                break

            # Convert batch to torch Variables
            x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
            x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
            x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
            x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
            y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
            y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)
            
            num_nodes = batch.nodes.shape[1]


            # Compute class weights (if uncomputed)
            edge_labels = y_edges.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

            # Forward pass
            y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
            loss = loss.mean()  # Take mean of loss across multiple GPUs

            # Compute error metrics
            # err_edges, err_tour, err_tsp, tour_err_idx, tsp_err_idx = edge_error(y_preds, y_edges, x_edges)

     

            # Get batch beamsearch tour prediction
            if mode == 'val':  # Validation: faster 'vanilla' beamsearch
                bs_nodes = beamsearch_tour_nodes(
                    y_preds, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='logits')
            elif mode == 'test':  # Testing: beamsearch with shortest tour heuristic 
                bs_nodes = beamsearch_tour_nodes_shortest(
                    y_preds, x_edges_values, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='logits')
            
            # Compute mean tour length
            pred_tour_len = mean_tour_len_nodes(x_edges_values, bs_nodes)
            gt_tour_len = np.mean(batch.tour_len)

            # Update running data
            running_nb_data += batch_size
            running_loss += batch_size* loss.data.item()
            # running_err_edges += batch_size* err_edges
            # running_err_tour += batch_size* err_tour
            # running_err_tsp += batch_size* err_tsp
            running_pred_tour_len += batch_size* pred_tour_len
            running_gt_tour_len += batch_size* gt_tour_len
            running_nb_batch += 1

            # Log intermediate statistics
            result = ('loss:{loss:.4f} pred_tour_len:{pred_tour_len:.3f} gt_tour_len:{gt_tour_len:.3f}'.format(
                loss=running_loss/running_nb_data,
                pred_tour_len=running_pred_tour_len/running_nb_data,
                gt_tour_len=running_gt_tour_len/running_nb_data))
            master_bar.child.comment = result

    # Compute statistics for full epoch
    loss = running_loss/ running_nb_data
    err_edges = 0 # running_err_edges/ running_nb_data
    err_tour = 0 # running_err_tour/ running_nb_data
    err_tsp = 0 # running_err_tsp/ running_nb_data
    pred_tour_len = running_pred_tour_len/ running_nb_data
    gt_tour_len = running_gt_tour_len/ running_nb_data

    return time.time()-start_test, loss, err_edges, err_tour, err_tsp, pred_tour_len, gt_tour_len

# In[46]:



def main(config):
    # Instantiate the network
    net = nn.DataParallel(ResidualGatedGCNModel(config, dtypeFloat, dtypeLong))
    if torch.cuda.is_available():
        net.cuda()
    print(net)

    # Compute number of network parameters
    nb_param = 0
    for param in net.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters:', nb_param)

    # Create log directory
    json.dump(config, open(f"{log_dir}/config.json", "w"), indent=4)
    writer = SummaryWriter(log_dir)

    # Training parameters
    max_epochs = config.max_epochs
    val_every = config.val_every
    test_every = config.test_every
    accumulation_steps = config.accumulation_steps
    learning_rate = config.learning_rate
    decay_rate = config.decay_rate

    val_loss_old = 1e6
    best_val_pred_tour_len = float("inf")

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print(optimizer)


    patience = 5   # quantos epochs sem melhorar antes de parar
    patience_counter = 0


    # -------------------------------
    # NEW: store epoch-wise metrics
    # -------------------------------
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pred_tour_len": [],
        "test_pred_tour_len": [],
        "test_loss": [],
        "train_loss_by_size": [],   # list of dicts
    }

    epoch_bar = master_bar(range(max_epochs))

    for epoch in epoch_bar:
        writer.add_scalar('learning_rate', learning_rate, epoch)

        # -------------------------
        # TRAIN
        # -------------------------
        (train_time, train_loss, train_err_edges, train_err_tour, 
         train_err_tsp, train_pred_tour_len, train_gt_tour_len, 
         loss_by_size) = train_one_epoch(net, optimizer, config, epoch_bar)

        # save for later export
        history["train_loss"].append(float(train_loss))
        epoch_size_means = {
            N: float(sum(vals)/len(vals)) for N, vals in loss_by_size.items()
        }

        history["train_loss_by_size"].append(epoch_size_means)

        epoch_bar.write(
            't: ' + metrics_to_str(
                epoch, train_time, learning_rate, train_loss,
                train_err_edges, train_err_tour, train_err_tsp,
                train_pred_tour_len, train_gt_tour_len, loss_by_size
            )
        )

        writer.add_scalar('loss/train_loss', train_loss, epoch)
        writer.add_scalar('pred_tour_len/train_pred_tour_len', train_pred_tour_len, epoch)
        writer.add_scalar('optimality_gap/train_opt_gap', train_pred_tour_len/train_gt_tour_len - 1, epoch)

        # -------------------------
        # VALIDATION
        # -------------------------
        if epoch % val_every == 0 or epoch == max_epochs - 1:

            (val_time, val_loss, val_err_edges, val_err_tour,
             val_err_tsp, val_pred_tour_len, val_gt_tour_len) = test(net, config, epoch_bar, mode='val')

            history["val_loss"].append(float(val_loss))
            history["val_pred_tour_len"].append(float(val_pred_tour_len))


            epoch_bar.write(
                'v: ' + metrics_to_str(
                    epoch, val_time, learning_rate, val_loss,
                    val_err_edges, val_err_tour, val_err_tsp,
                    val_pred_tour_len, val_gt_tour_len
                )
            )

            writer.add_scalar('loss/val_loss', val_loss, epoch)

            # Save checkpoint if improved
            if val_pred_tour_len < best_val_pred_tour_len:
                best_val_pred_tour_len = float(val_pred_tour_len)
                patience_counter = 0
                config_name = os.path.splitext(os.path.basename(args.config))[0]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_pred_tour_len': val_pred_tour_len,
                    'val_gt_tour_len': val_gt_tour_len,
                    'val_opt_gap': val_pred_tour_len / val_gt_tour_len - 1,
                }, os.path.join(log_dir, f"best_val_checkpoint.tar"))
            else:
                patience_counter += 1

            # Update LR
            if val_loss > 0.99 * val_loss_old:
                learning_rate /= decay_rate
                optimizer = update_learning_rate(optimizer, learning_rate)

            val_loss_old = val_loss

            # -------------------------------------
            # EARLY STOPPING
            # -------------------------------------
            if patience_counter >= patience:
                epoch_bar.write(f"EARLY STOPPING at epoch {epoch} (no tour lenght improvement for {patience} epochs).")
                break

         

        # -------------------------
        # TEST
        # -------------------------
        if epoch % test_every == 0 or epoch == max_epochs - 1:
            (test_time, test_loss, test_err_edges, test_err_tour,
             test_err_tsp, test_pred_tour_len, test_gt_tour_len) = test(net, config, epoch_bar, mode='test')

            history["test_loss"].append(float(test_loss))
            history["test_pred_tour_len"].append(float(test_pred_tour_len))


            epoch_bar.write(
                'T: ' + metrics_to_str(
                    epoch, test_time, learning_rate, test_loss,
                    test_err_edges, test_err_tour, test_err_tsp,
                    test_pred_tour_len, test_gt_tour_len
                )
            )

            writer.add_scalar('loss/test_loss', test_loss, epoch)

        print(f"\n===== EPOCH {epoch} =====", flush=True)

        print(f"[EPOCH {epoch}] learning_rate = {learning_rate:.6e}", flush=True)
        print(f"[EPOCH {epoch}] train_loss = {train_loss:.6f}", flush=True)
        print(f"[EPOCH {epoch}] train_pred_tour_len = {train_pred_tour_len:.6f}", flush=True)

        train_opt_gap = train_pred_tour_len / train_gt_tour_len - 1
        print(f"[EPOCH {epoch}] train_opt_gap = {train_opt_gap:.6f}", flush=True)

        if epoch % val_every == 0 or epoch == max_epochs - 1:
            print(f"[EPOCH {epoch}] val_loss = {val_loss:.6f}", flush=True)
            print(f"[EPOCH {epoch}] val_pred_tour_len = {val_pred_tour_len:.6f}", flush=True)

        if epoch % test_every == 0 or epoch == max_epochs - 1:
            print(f"[EPOCH {epoch}] test_loss = {test_loss:.6f}", flush=True)
            print(f"[EPOCH {epoch}] test_pred_tour_len = {test_pred_tour_len:.6f}", flush=True)

        # END OF EPOCH CHECKPOINT
        torch.save({
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_pred_tour_len": val_pred_tour_len,
            "val_gt_tour_len": val_gt_tour_len,
            "val_opt_gap": val_pred_tour_len / val_gt_tour_len - 1,
        }, log_dir + "last_train_checkpoint.tar")

    # -------------------------
    # EXPORT HISTORY
    # -------------------------
    with open(log_dir + "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Training history saved at:", log_dir + "training_history.json")

    return net, history


net, history = main(config)



plot_training_losses(history)
plt.savefig(
    os.path.join(log_dir, "training_losses.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# # Load saved checkpoint

# In[52]:



# Load checkpoint
if torch.cuda.is_available():
    checkpoint = torch.load(log_dir+"best_val_checkpoint.tar")
else:
    checkpoint = torch.load(log_dir+"best_val_checkpoint.tar", map_location='cpu')
# Load network state
net.load_state_dict(checkpoint['model_state_dict'])
# Load optimizer state
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# Load other training parameters
epoch = checkpoint['epoch']
train_loss = checkpoint['train_loss']


for param_group in optimizer.param_groups:
    learning_rate = param_group['lr']
print(f"Loaded checkpoint from epoch {epoch}")    

# # Visualize model predictions

# In[53]:



# Set evaluation mode
net.eval()

batch_size = config.batch_size
knn_ratio = config.knn_ratio
beam_size = config.beam_size
test_filepath = config.test_filepath
dataset = iter(VariableTSPReader(test_filepath, batch_size,  knn_ratio=config.knn_ratio, iteration_mode=config.iteration_mode, metric=config.metric))
batch = next(dataset)

N = batch.nodes.shape[1]

with torch.no_grad():
    # Convert batch to torch Variables
    x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
    x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
    x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
    x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
    y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
    y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)
    
    # Compute class weights
    edge_labels = y_edges.cpu().numpy().flatten()
    edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
    print("Class weights: {}".format(edge_cw))
    
    # Forward pass
    y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
    loss = loss.mean()
    
    # Get batch beamsearch tour prediction
    bs_nodes = beamsearch_tour_nodes_shortest(
        y_preds, x_edges_values, beam_size, batch_size, N, dtypeFloat, dtypeLong, probs_type='logits')

  

    # Compute mean tour length
    pred_tour_len = mean_tour_len_nodes(x_edges_values, bs_nodes)
    gt_tour_len = np.mean(batch.tour_len)
    print("Predicted tour length: {:.3f} (mean)\nGroundtruth tour length: {:.3f} (mean)".format(pred_tour_len, gt_tour_len))

    # Sanity check
    for idx, nodes in enumerate(bs_nodes):
        if not is_valid_tour(nodes, N):
            print(idx, " Invalid tour: ", nodes)

    # Plot prediction visualizations
    plot_predictions_beamsearch(x_nodes_coord, x_edges, x_edges_values, y_edges, y_preds, bs_nodes, num_plots=3)

# ## Metrics

# In[54]:


def evaluate_full_testset_per_graph(net, config):
    net.eval()

    batch_size = config.batch_size
    beam_size = config.beam_size
    test_filepath = config.test_filepath
    knn_ratio = config.knn_ratio

    dataset = VariableTSPReader(test_filepath, batch_size, knn_ratio=knn_ratio, iteration_mode=config.iteration_mode, metric=config.metric)
    iterator = iter(dataset)

    # armazenar métricas por grafo
    all_pred_lengths = []
    all_gt_lengths = []
    all_gaps = []
    all_losses = []
    all_sizes = []   # N de cada grafo

    with torch.no_grad():
        for _ in range(dataset.max_iter):
            try:
                batch = next(iterator)
            except StopIteration:
                break

            B = batch.nodes.shape[0]
            N = batch.nodes.shape[1]

            # tensores
            x_edges = torch.LongTensor(batch.edges).type(dtypeLong)
            x_edges_values = torch.FloatTensor(batch.edges_values).type(dtypeFloat)
            x_nodes = torch.LongTensor(batch.nodes).type(dtypeLong)
            x_nodes_coord = torch.FloatTensor(batch.nodes_coord).type(dtypeFloat)
            y_edges = torch.LongTensor(batch.edges_target).type(dtypeLong)

            # class weights
            edge_labels = y_edges.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

            # forward
            y_preds, loss = net.forward(
                x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw
            )
            loss = loss.mean().item()  # ESCALAR

            # beamsearch: retorna uma lista de tours (tuplas de nós)
            bs_nodes = beamsearch_tour_nodes_shortest(
                y_preds, x_edges_values, beam_size, B, N, dtypeFloat, dtypeLong, probs_type='logits'
            )

            # tour length por grafo pred
            pred_len_list = []
            for b in range(B):
                tour = bs_nodes[b]
                length = 0.0
                for i in range(len(tour)):
                    u = tour[i]
                    v = tour[(i + 1) % len(tour)]
                    length += float(x_edges_values[b, u, v])
                pred_len_list.append(length)

            # gt tour length por grafo
            gt_len_list = batch.tour_len.tolist()

            # gaps
            gap_list = [(p/g - 1.0) for p, g in zip(pred_len_list, gt_len_list)]

            # armazenar tudo
            all_pred_lengths.extend(pred_len_list)
            all_gt_lengths.extend(gt_len_list)
            all_gaps.extend(gap_list)
            all_losses.extend([loss] * B)
            all_sizes.extend([N] * B)

    # -------- agregação global --------
    global_results = {
        "mean_pred_tour_len": float(np.mean(all_pred_lengths)),
        "std_pred_tour_len": float(np.std(all_pred_lengths)),
        
        "mean_gt_tour_len": float(np.mean(all_gt_lengths)),
        "std_gt_tour_len": float(np.std(all_gt_lengths)),
        
        "mean_gap": float(np.mean(all_gaps)),
        "std_gap": float(np.std(all_gaps)),
        
        "mean_loss": float(np.mean(all_losses)),
        "std_loss": float(np.std(all_losses)),
        
        "num_graphs": len(all_pred_lengths)
    }

    # -------- agregação por N --------
    results_by_N = {}
    unique_N = sorted(set(all_sizes))
    for N in unique_N:
        idx = [i for i, x in enumerate(all_sizes) if x == N]

        pred = [all_pred_lengths[i] for i in idx]
        gt = [all_gt_lengths[i] for i in idx]
        gaps = [all_gaps[i] for i in idx]
        losses = [all_losses[i] for i in idx]

        results_by_N[N] = {
            "mean_pred_tour_len": float(np.mean(pred)),
            "std_pred_tour_len": float(np.std(pred)),
            
            "mean_gt_tour_len": float(np.mean(gt)),
            "std_gt_tour_len": float(np.std(gt)),
            
            "mean_gap": float(np.mean(gaps)),
            "std_gap": float(np.std(gaps)),
            
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            
            "count": len(idx)
        }

    return global_results, results_by_N



global_results, results_by_N = evaluate_full_testset_per_graph(net, config)

save_path = os.path.join(log_dir, "test_results.json")

results_to_save = {
    "global_results": global_results,
    "results_by_N": {str(N): v for N, v in results_by_N.items()}
}

with open(save_path, "w") as f:
    json.dump(results_to_save, f, indent=2)


plot_results_by_N(results_by_N)

plt.savefig(os.path.join(log_dir, "results_by_N.png"),
            dpi=200,
            bbox_inches="tight")

plt.close()
