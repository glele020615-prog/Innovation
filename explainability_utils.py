import numpy as np
import torch


def generate_per_subject_analysis_with_stats(subject_id, edge_imp, brain_region_names, true_label, pred_label,
                                             prob_class1, threshold=0.1, top_k=10, output_file=None):
    """为单个被试生成可解释性分析报告，并返回统计信息"""
    subject_stats = {
        'subject_id': subject_id,
        'true_label': true_label,
        'pred_label': pred_label,
        'prob_class1': prob_class1,
        'edge_imp_matrix': edge_imp.copy(),
        'node_importance': np.sum(edge_imp, axis=1) - np.diag(edge_imp),
        'top_edges': [],
        'top_nodes': []
    }

    V = edge_imp.shape[0]
    flat_indices = np.triu_indices(V, k=1)
    edge_values = edge_imp[flat_indices]
    top_k_indices = np.argsort(edge_values)[-top_k:][::-1]

    for k in top_k_indices:
        i, j = flat_indices[0][k], flat_indices[1][k]
        subject_stats['top_edges'].append({
            'i': i,
            'j': j,
            'importance': edge_values[k],
            'edge_name': f"{brain_region_names[i]}-{brain_region_names[j]}"
        })

    node_importance = subject_stats['node_importance']
    top_k_nodes = np.argsort(node_importance)[-top_k:][::-1]

    for node_idx in top_k_nodes:
        subject_stats['top_nodes'].append({
            'node_idx': node_idx,
            'importance': node_importance[node_idx],
            'node_name': brain_region_names[node_idx]
        })

    return subject_stats


def compute_gradient_based_edge_importance(model, data, adj, label, device):
    """使用梯度方法计算样本特定的边重要性"""
    model.eval()
    data = data.clone().detach().requires_grad_(True).to(device)
    adj = adj.clone().detach().requires_grad_(True).to(device)
    label = label.to(device)
    model.zero_grad()
    outputs = model(data, adj)
    pred_class = outputs.argmax(dim=1).item()
    loss = outputs[0, pred_class]
    loss.backward()
    edge_importance = None
    if adj.grad is not None:
        if len(adj.grad.shape) == 4:
            edge_importance = torch.abs(adj.grad[0, 0]).cpu().detach().numpy()
        elif len(adj.grad.shape) == 3:
            edge_importance = torch.abs(adj.grad[0]).cpu().detach().numpy()
        elif len(adj.grad.shape) == 2:
            edge_importance = torch.abs(adj.grad).cpu().detach().numpy()
        else:
            print(f"警告: adj.grad的形状异常: {adj.grad.shape}")

    if edge_importance is None or edge_importance.shape[0] != 116:
        with torch.no_grad():
            if hasattr(model, 'edge_importance') and model.edge_importance is not None:
                if isinstance(model.edge_importance, list) and len(model.edge_importance) > 0:
                    tmp_abs_edge = torch.abs(model.edge_importance[0])
                else:
                    tmp_abs_edge = torch.abs(model.edge_importance)
                if len(tmp_abs_edge.shape) == 4:
                    tmp_abs_edge = tmp_abs_edge.squeeze(0)
                if len(tmp_abs_edge.shape) == 3:
                    tmp_abs_edge = tmp_abs_edge.squeeze(0)
                edge_importances = (tmp_abs_edge / 2 + tmp_abs_edge.transpose(-1, -2) / 2)
                edge_importance = edge_importances.cpu().numpy()
            else:
                edge_importance = np.zeros((116, 116))

    if edge_importance.shape != (116, 116):
        if edge_importance.shape[0] == 116 and edge_importance.shape[1] == 116:
            pass
        elif len(edge_importance.shape) == 1 and edge_importance.shape[0] == 116 * 116:
            edge_importance = edge_importance.reshape((116, 116))
        else:
            edge_importance = np.zeros((116, 116))

    if edge_importance.max() > 0:
        edge_importance = edge_importance / edge_importance.max()

    return edge_importance