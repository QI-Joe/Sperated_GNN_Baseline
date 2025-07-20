import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from utils.data_processing import get_link_prediction_data, get_idx_data_loader, get_link_prediction_data4KG_Data

from models.GAT_Standard import GAT_Simple
from models.GraphSage_Standard import GraphSage_Simple
from models.GCONV import GCONV_Simple
from models.modules import MergeLayer
from utils.utils import NegativeEdgeSampler, set_random_seed, get_parameter_sizes, convert_to_gpu, convert2pyg_batch_data


from utils.early_stopping import EarlyStopping
from utils.metrics import get_link_prediction_metrics
from evaluate_models_utils import evaluate_model_link_prediction 
from utils.data_clean import  get_link_prediction_args
from torch_geometric.loader import NeighborLoader


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args()
    SNAPSHOT = args.snapshot
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    weight_lambda = args.weight_lambda
    current_date = time.strftime("%m-%d-%H", time.localtime())
    suffix_date = time.strftime("%H-%M-%S", time.localtime())

    # get data for training, validation and testing
    if args.dataset_name == "entities":
        dataset_path = {
            "general": r"../Standard_Dataset/kg/",
            "node": r"extraction_results_{}_{}_nodes.csv".format(args.dataset_start, args.dataset_end),
            "edge": r"extraction_results_{}_{}_edges.csv".format(args.dataset_start, args.dataset_end),
            "node_feat": r"embedding_result/{}_{}_embeddings.pt".format(args.dataset_start, args.dataset_end),
            "edge_feat": r"embedding_result/{}_{}_edge_embeddings.pt".format(args.dataset_start, args.dataset_end)
            }
        node_raw_features, edge_raw_features, data_list = get_link_prediction_data4KG_Data(
            dataset_name=args.dataset_name, snapshot=SNAPSHOT, kg_path=dataset_path)
        data_list = [data_list]
    else:
        node_raw_features, edge_raw_features, data_list = \
            get_link_prediction_data(dataset_name=args.dataset_name, snapshot=SNAPSHOT, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    in_channels = node_raw_features.shape[1]
    out_channels = 64
    num_layers = args.num_layers
    
    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(1):
        full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = data_list[run]
        # initialize training neighbor sampler to retrieve temporal graph
        pyg_train, pyg_val, pyg_test, pyg_new_node_val, pyg_new_node_test = convert2pyg_batch_data([train_data, val_data, test_data, new_node_val_data, new_node_test_data], node_raw_features, edge_raw_features)
        dloader_share_kwargs = {"num_neighbors": [-1], "batch_size": args.batch_size, "shuffle": False}

        # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
        # in the inductive setting, negatives are sampled only amongst other new nodes
        # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
        train_idx_data_loader = NeighborLoader(pyg_train, **dloader_share_kwargs)
        val_idx_data_loader = NeighborLoader(pyg_val, **dloader_share_kwargs)
        new_node_val_idx_data_loader = NeighborLoader(pyg_new_node_val, **dloader_share_kwargs)
        test_idx_data_loader = NeighborLoader(pyg_test, **dloader_share_kwargs)
        new_node_test_idx_data_loader = NeighborLoader(pyg_new_node_test, **dloader_share_kwargs)
        
        set_random_seed(seed=2025)

        args.seed = 2025
        args.save_model_name = f'{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')
        
        shared_kwargs = {'in_channels': in_channels, 'out_channels': out_channels, 'device': device, 'dropout': args.dropout}
        match args.model_name:
            case "GCN":
                dynamic_backbone = GCONV_Simple(
                    hidden_dim1=args.hidden_channels, hidden_dim2=args.hidden_channels, time_dim=1, **shared_kwargs)
            case "GAT":
                dynamic_backbone = GAT_Simple(
                    num_layers=num_layers, heads=args.num_heads, negative_slope=args.negative_slope,
                    add_self_loops=True, time_dim=1, **shared_kwargs)
            case "GraphSAGE":
                dynamic_backbone = GraphSage_Simple(
                    hidden_channels=args.hidden_channels, time_dim=1, **shared_kwargs)  
            case "_":
                raise NotImplementedError(f"Model {args.model_name} is not implemented.")
        
        link_predictor=MergeLayer(input_dim1=out_channels, input_dim2=out_channels, hidden_dim=out_channels, output_dim=1).to(args.device)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        model = convert_to_gpu(model, device=args.device)
        
        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCELoss()
        mvgrl_loss, gca_loss = None, None
        
        for epoch in range(args.num_epochs):
            model.train()
            
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            
            for batch_idx, batch_data in enumerate(train_idx_data_loader_tqdm):
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    batch_data.edge_index[0].numpy(), batch_data.edge_index[1].numpy(), \
                    batch_data.time.numpy(), batch_data.edge_attr.numpy()
                
                batch_data = batch_data.to(device)
                        
                if args.model_name in ["GCN", "GAT", "GraphSAGE"]:
                    batch_z1 = model[0].compute_src_dst_node_temporal_embeddings(batch_data)
                else:
                    raise NotImplementedError(f"Model {args.model_name} is not implemented.")
                
                train_local_graph_neg_sampler = NegativeEdgeSampler(src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids, seed=0)
                _, batch_neg_dst_node_ids = train_local_graph_neg_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids, batch_neg_dst_node_ids = torch.from_numpy(batch_src_node_ids).to(device), torch.from_numpy(batch_neg_dst_node_ids).to(device)
                
                batch_src_node_embeddings, batch_dst_node_embeddings = batch_z1[batch_neg_src_node_ids], batch_z1[batch_data.edge_index[1]]
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = batch_z1[batch_neg_src_node_ids], batch_z1[batch_neg_dst_node_ids]
                
                positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                loss = loss_func(input=predicts, target=labels)
                train_losses.append(loss.item())

                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
            
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                    model=model,
                                                                    evaluate_idx_data_loader=val_idx_data_loader,
                                                                    device=device,
                                                                    loss_func=loss_func)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       device=device,
                                                                                       loss_func=loss_func)
            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(
                    model_name=args.model_name,
                    model=model,
                    evaluate_idx_data_loader=test_idx_data_loader,
                    device=device,
                    loss_func=loss_func
                )

                new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(
                    model_name=args.model_name,
                    model=model,
                    evaluate_idx_data_loader=new_node_test_idx_data_loader,
                    device=device,
                    loss_func=loss_func
                )

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                for metric_name in new_node_test_metrics[0].keys():
                    logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')


        test_losses, test_metrics = evaluate_model_link_prediction(
            model_name=args.model_name,
            model=model,
            evaluate_idx_data_loader=test_idx_data_loader,
            device=device,
            loss_func=loss_func
        )

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(
            model_name=args.model_name,
            model=model,
            evaluate_idx_data_loader=new_node_test_idx_data_loader,
            device=device,
            loss_func=loss_func
        )
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        logger.info(f"val loss: {np.mean(val_losses):.4f}')")
        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'val {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric
        
        logger.info(f'new node val loss: {np.mean(new_node_val_losses):.4f}')
        for metric_name in new_node_val_metrics[0].keys():
            average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
            logger.info(f'new node val {metric_name}, {average_new_node_val_metric:.4f}')
            new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.snapshot - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
            "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        args.save_model_name = f'{args.save_model_name}_{run}'
        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}/{current_date}/{suffix_date}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {SNAPSHOT-2} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
        logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()