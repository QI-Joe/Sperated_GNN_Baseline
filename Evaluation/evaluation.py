import math
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score
import sys
sys.path.append("/mnt/d/CodingArea/Python/TestProject")
from models.Tppr_tgn_model import TGN
import torch as th
from torch.optim import Adam
import torch.nn as nn
from Testing_code.uselessCode import node_index_anchoring

def eval_edge_prediction(model: TGN, negative_edge_sampler, data, n_neighbors, batch_size):

  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc, val_acc = [], [], []
  with torch.no_grad():
    model = model.eval()
    TEST_BATCH_SIZE = batch_size
    num_test_instance = data.n_interactions
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
 
    for batch_idx in range(num_test_batch):
      start_idx = batch_idx * TEST_BATCH_SIZE
      end_idx = min(num_test_instance, start_idx + TEST_BATCH_SIZE)
      sample_inds=np.array(list(range(start_idx,end_idx)))

      sources_batch = data.sources[sample_inds]
      destinations_batch = data.destinations[sample_inds]
      timestamps_batch = data.timestamps[sample_inds]
      edge_idxs_batch = data.edge_idxs[sample_inds]


      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)
      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negative_samples, timestamps_batch, edge_idxs_batch, n_neighbors, train = False)
      
      pos_prob=pos_prob.cpu().numpy() 
      neg_prob=neg_prob.cpu().numpy() 

      pred_score = np.concatenate([pos_prob, neg_prob])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])
      
      true_binary_label= np.zeros(size)
      pred_binary_label = np.argmax(np.hstack([pos_prob,neg_prob]),axis=1)

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))
      val_acc.append(accuracy_score(true_binary_label, pred_binary_label))

  return np.mean(val_ap), np.mean(val_auc), np.mean(val_acc)

def eval_node_classifier(tgn: TGN, data, batch_size, n_neighbors):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    pred_prob = torch.tensor([]).to(device)
    for batch in range(num_batch):
        start_idx = batch * batch_size
        end_idx = min(num_instance, start_idx + batch_size)
        sources_batch = data.sources[start_idx: end_idx]
        destinations_batch = data.destinations[start_idx: end_idx]
        timestamps_batch = data.timestamps[start_idx: end_idx]
        edge_idxs_batch = data.edge_idxs[start_idx: end_idx]

        source_embedding, destination_embedding, _, all_emb = tgn.compute_temporal_embeddings(sources_batch,
                                                                                               destinations_batch,
                                                                                               destinations_batch,
                                                                                               timestamps_batch,
                                                                                               edge_idxs_batch,
                                                                                               n_neighbors, train=False)
        
        pred_prob_batch = torch.argmax(source_embedding, dim=1)
        pred_prob = torch.cat((pred_prob, pred_prob_batch), dim=0)
    node_mask = node_index_anchoring(data.sources)
    nodes = data.sources[node_mask]
    pred_prob = pred_prob.cpu().numpy()[nodes].reshape(-1,1)
    ground_truth = data.labels[nodes].reshape(-1,1)

    print(ground_truth.shape, pred_prob.shape)
    # auc_roc = roc_auc_score(ground_truth, pred_prob, multi_class='ovr')
    prec= precision_score(ground_truth, pred_prob, average='macro', zero_division=0)
    acc = accuracy_score(ground_truth, pred_prob)
    recall = recall_score(ground_truth, pred_prob, average='macro', zero_division=0)
    return prec, acc, recall

def eval_node_classification(tgn: TGN, decoder, data, batch_size, n_neighbors, val_neg_generator):
  device = th.device('cuda' if th.cuda.is_available() else 'cpu')
  # pred_prob = np.zeros(len(data.sources))
  decoder, decoder1 = decoder
  all_decoder = nn.Sequential(decoder, decoder1)
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)
  f = nn.LogSoftmax(dim=-1)
  optimizer = Adam(all_decoder.parameters(), lr=0.01, weight_decay=1e-4)

  loss_fn = nn.CrossEntropyLoss(reduction="mean")
  decoder_epoch = 100

  for epoch in range(decoder_epoch):
    all_decoder.train()
    tgn.eval()
    optimizer.zero_grad()
    pred_prob = torch.tensor([]).to(device)
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]
      _, negative = val_neg_generator.sample(len(sources_batch))

      all_nodes = np.concatenate([sources_batch, destinations_batch, negative])


      source_embedding, destination_embedding, _, all_emb = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   negative,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors, train=True)
      pred_prob_batch = all_decoder(all_emb)
      loss = loss_fn(pred_prob_batch, torch.tensor(data.labels[all_nodes]).to(device))
      loss.backward()
      optimizer.step()
      if (k+1) % 5==0:
        print("epoch {:4d} | batch {:4d} | loss {:4f}".format(epoch+1, k+1, loss.item()))
      if epoch == decoder_epoch-1:
        pred_prob_batch = torch.argmax(pred_prob_batch, dim=1)
        pred_prob = torch.cat((pred_prob, pred_prob_batch), dim=0)

  node_mask = node_index_anchoring(data.sources)
  nodes = data.sources[node_mask]
  pred_prob = pred_prob.cpu().numpy()[nodes].reshape(-1,1)
  ground_truth = data.labels[nodes].reshape(-1,1)

  print(ground_truth.shape, pred_prob.shape)
  # auc_roc = roc_auc_score(ground_truth, pred_prob, multi_class='ovr')
  prec= precision_score(ground_truth, pred_prob, average='macro', zero_division=0)
  acc = accuracy_score(ground_truth, pred_prob)
  recall = recall_score(ground_truth, pred_prob, average='macro', zero_division=0)
  return prec, acc, recall
