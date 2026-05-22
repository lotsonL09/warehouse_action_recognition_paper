import torch


def filtering_stage(video_batch:torch.Tensor,eps=1e-12,n_frames_selected:int=15):

    bs,n_frames,_,_,_=video_batch.shape

    device = video_batch.device

    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=video_batch.device).view(1,1,3,1,1)

    video_batch = (video_batch*weights).sum(dim=2,keepdim=True)

    diff =torch.abs(video_batch[:,1:]-video_batch[:,:-1]).mean(dim=(-1,-2))

    zero=torch.zeros_like(diff[:,:1])
    diff = torch.cat([zero,diff],dim=1)

    diff=diff**2

    motion_salience_distribution = diff/(diff.sum(dim=1,keepdim=True) +eps)

    #cumulative distribuction function
    cdf = torch.cumsum(motion_salience_distribution, dim=1)

    x_index=cdf.squeeze(-1)

    edges = torch.linspace(0,1,n_frames_selected+1,device=device)

    u = edges[:-1] + (edges[1:] - edges[:-1]) * torch.rand(bs,n_frames_selected,device=device)

    indices =  torch.searchsorted(x_index,u)

    indices = indices.clamp(max=n_frames-1)

    return indices,cdf

    # diff=torch.abs(video_batch[:,1:]-video_batch[:,:-1]).squeeze(dim=2) #corregir

    # diff=diff.view(bs,n_frames-1,-1) #(bs,n_frames-1,H*W)

    # p = diff/(diff.sum(dim=2,keepdim=True) + eps) #normalizacion para pseudo distribucion

    # E = -(p * torch.log2(p + eps)).sum(dim=2)  # (B, T-1)

    # zero_col = torch.zeros(bs, 1, device=device)
    # E=torch.cat([E,zero_col],dim=1) #bs,n_frames

    # E_hat = E[:, :-1].mean(dim=1, keepdim=True)  # (B, 1)

    # D = (E - E_hat) / (E + eps)  # (B, T)

    # selected_indices = []

    # for b in range(bs):
    #     frame_idxs=torch.where(D[b] > 0)[0]
    #     if len(frame_idxs) == 0:
    #         frame_idxs = torch.arange(n_frames, device=device)
    #     e = E[b,frame_idxs]
    #     e_norm = e / (e.sum() + eps)
    #     cdf = torch.cumsum(e_norm, dim=0)
    #     intervals = torch.linspace(0, 1, n_frames_selected + 1, device=device)
    #     chosen = []
    #     for k in range(n_frames_selected):
    #         a = intervals[k]
    #         b_int = intervals[k + 1]

    #         u = torch.rand(1, device=device) * (b_int - a) + a
    #         pos = torch.searchsorted(cdf, u)
    #         pos = torch.clamp(pos, 0, len(frame_idxs) - 1)

    #         chosen.append(frame_idxs[pos])
    #     chosen = torch.stack(chosen).squeeze()
        
    #     chosen, _ = torch.sort(chosen)

    #     selected_indices.append(chosen)

    # return torch.stack(selected_indices, dim=0)

def select_frames(video_batch, indices_list):
    B = video_batch.shape[0]
    selected_videos = []

    for b in range(B):
        selected_videos.append(video_batch[b, indices_list[b]])

    return torch.stack(selected_videos)  # (B, n_frames_selected, C, H, W)