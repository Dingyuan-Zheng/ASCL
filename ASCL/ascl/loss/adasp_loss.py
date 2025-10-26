from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations, permutations


# __all__ = ["AdaSPLoss"]

class AdaSPLoss(nn.Module):
    """
    Adaptive sparse pairwise (AdaSP) loss
    """

    def __init__(self, temp=0.04, loss_type='adasp'):
        super(AdaSPLoss, self).__init__()
        self.temp = temp  # discard this t value
        self.loss_type = loss_type

    def forward(self, feats, targets):

        feats_n = nn.functional.normalize(feats, dim=1)

        bs_size = feats_n.size(0)  # bs_size = 64
        N_id = len(torch.unique(targets))  # N_id = 16
        N_ins = bs_size // N_id  # N_ins = 4
        N_ins = 4  # 8 for veri, 4 for other 2 datasets
        t_value = 0.05  # we use this t value

        scale = 1. / self.temp

        sim_qq = torch.matmul(feats_n, feats_n.T)
        sf_sim_qq = sim_qq * scale  # discard this value

        ### zdy
        if N_id > (bs_size / N_ins):
            # print(sim_qq.size(), sim_pos_mask.size(), feats_n.size(), targets.size(), sim_self_mask1.size(), sim_self_mask2.size(), targets, bs_size)
            sum_tarid = targets.expand(bs_size, bs_size).eq(targets.expand(bs_size, bs_size).t()).sum(dim=1)
            sim_qq = sim_qq[sum_tarid == N_ins, :][:, sum_tarid == N_ins]
            targets = targets[sum_tarid == N_ins]
            N_id = len(torch.unique(targets))

        sim_pos_mask = torch.from_numpy(np.kron(np.eye(N_id), np.ones((N_ins, N_ins)))).cuda()  # 16 / 4X4

        sim_self_mask1 = (1.0 - torch.eye(N_id * N_ins)).cuda()  # mask self-instance similarity '1'
        sim_self_mask2 = -99999. * torch.eye(N_id * N_ins).cuda()  # enforce the self-instance similarity to extremely small value
        # sim_qq_pos = sim_qq.mul(sim_pos_mask).mul(sim_self_mask1) + sim_self_mask2

        sim_qq_pos = sim_qq.mul(sim_pos_mask).mul(sim_self_mask1) / t_value  # 1st-try: 0.07
        sim_qq_pos = sim_qq_pos + sim_self_mask2

        sim_neg_mask1 = 1.0 - sim_pos_mask
        sim_neg_mask2 = -99999. * sim_pos_mask
        # sim_qq_neg = sim_qq.mul(sim_neg_mask1) + sim_neg_mask2
        sim_qq_neg = sim_qq.mul(sim_neg_mask1) / t_value + sim_neg_mask2

        sim_pos_hh_lt = []
        sim_pos_he_lt = []
        sim_pos_hh_2nd_lt = []
        sim_pos_he_2nd_lt = []
        sim_neg_ls = []

        #zdy dense CSCL
        sim1_lt_all = []
        sim2_lt_all = []
        negs_lt = []
        pos_lt = []
        # zdy end dense CSCL

        for idx in range(N_id):
            # search hh, he
            pos_matrix = sim_qq_pos[idx * N_ins:(idx + 1) * N_ins, idx * N_ins:(idx + 1) * N_ins]
            pos_flat = pos_matrix.flatten()
            pos_id_ins, _ = torch.sort(pos_flat, descending=True)

            # zdy dense CSCL
            pos_seg = pos_id_ins[:N_ins * (N_ins - 1)][0:N_ins * (N_ins - 1):2]  # 4x3/2 or 8*7/2
            pos_seg_len = pos_seg.size(0)
            ss = np.arange(pos_seg_len)
            sim1_lt = []
            sim2_lt = []
            #negs_lt = []
            for element in combinations(ss, 2):
                # print(element)
                sim1_lt.append(pos_seg[element[0]])
                sim2_lt.append(pos_seg[element[1]])

            sim1_lt_all.append(torch.stack(sim1_lt))
            sim2_lt_all.append(torch.stack(sim2_lt))

            pos_lt.append(pos_seg)

            # search n_neg neg
            neg_matrix = sim_qq_neg[idx * N_ins:(idx + 1) * N_ins, :]
            neg_flat = neg_matrix.flatten()
            neg_id_ins, _ = torch.sort(neg_flat, descending=True)
            negs_lt.append(neg_id_ins[:N_ins])

            # end dense CSCL part1

            # pos_id_min_2nd, pos_id_min, pos_id_max = pos_id_ins[-(3 + N_ins)], pos_id_ins[-(1 + N_ins)], pos_id_ins[0] #mine least-hard/hardest
            pos_id_min_2nd, pos_id_min, pos_id_max_2nd, pos_id_max = pos_id_ins[-(3 + N_ins)], pos_id_ins[-(1 + N_ins)], pos_id_ins[4], pos_id_ins[0]  # ablation study 2,3->2nd; 4,5->3rd
            sim_pos_hh_lt.append(pos_id_min)  # hardest pair with min similarity
            sim_pos_he_lt.append(pos_id_max)  # easiest pair with max similarity
            sim_pos_hh_2nd_lt.append(pos_id_min_2nd)  # 2nd hardest pair with 2nd min similarity
            sim_pos_he_2nd_lt.append(pos_id_max_2nd)  # abl: 2nd easiest pair with 2nd max sim
            # search n_neg neg
            neg_matrix = sim_qq_neg[idx * N_ins:(idx + 1) * N_ins, :]
            neg_flat = neg_matrix.flatten()
            neg_id_ins, _ = torch.sort(neg_flat, descending=True)
            sim_neg_ls.append(neg_id_ins[:N_ins])  # fix neg selection scheme: N neg instances ; or use an instant value 4
            # print(len(sim_neg_ls)) ## others1: increase N; others2: modulate margin negs, and use non-margin negs; modulate with 1st and 2nd negs

        # zdy dense CSCL part2 start
        pos1_stack = torch.exp(torch.stack(sim1_lt_all))
        pos2_stack = torch.exp(torch.stack(sim2_lt_all))

        pos_stack = torch.stack(pos_lt) # dense only without cscl
        negs_stack = torch.exp(torch.stack(negs_lt))

        pos_neg_cat = torch.cat((pos_stack,torch.log(negs_stack)),dim=1)
        loss_dense = -1 * F.log_softmax(pos_neg_cat, dim=1)[:, :pos_stack.size(1)].mean()  # abl: loss dense


        sigma_pos = 2. * pos1_stack.mul(pos2_stack) / (pos1_stack + pos2_stack)
        pos_cscl_neg_cat = torch.cat((torch.log(sigma_pos), torch.log(negs_stack)), dim=1)

        # loss_dense_cscl = -1 * F.log_softmax(pos_cscl_neg_cat, dim=1)[:, :pos1_stack.size(1)].mean() #abl loss dense-cscl

        # zdy dense CSCL part2 end

        sim_neg_stack = torch.exp(torch.stack(sim_neg_ls))
        sim_pos_hh = torch.exp(torch.stack(sim_pos_hh_lt))

        sim_pos_he = torch.exp(torch.stack(sim_pos_he_lt))

        # abl
        sim_pos_he_2nd = torch.exp(torch.stack(sim_pos_he_2nd_lt))  # abl

        ### compute the dynamic weight for hh contrast
        sim_pos_cat = torch.cat((torch.stack(sim_pos_hh_lt).unsqueeze(1), torch.stack(sim_pos_he_lt).unsqueeze(1)),
                                dim=1)
        sim_hh_weight = torch.softmax(sim_pos_cat, dim=1)[:, 0]
        sim_pos_hh_neg = torch.cat((torch.log(sim_pos_hh).unsqueeze(1), torch.log(sim_neg_stack)), dim=1)
        ###
        sim_pos_hh_2nd = torch.exp(torch.stack(sim_pos_hh_2nd_lt))

        sim_pos_hm = 2. * sim_pos_he.mul(sim_pos_hh) / (sim_pos_he + sim_pos_hh)

        # abl
        sim_pos_hm_2nd = 2. * sim_pos_he_2nd.mul(sim_pos_hh) / (sim_pos_he_2nd + sim_pos_hh)  # abl hardest harmonic with 2nd easiest

        sim_pos_hm_neg = torch.cat((torch.log(sim_pos_hm).unsqueeze(1), torch.log(sim_neg_stack)), dim=1)


        sim_pos_hm_2nd_neg = torch.cat((torch.log(sim_pos_hm_2nd).unsqueeze(1), torch.log(sim_neg_stack)), dim=1)

        loss_our = -1 * F.log_softmax(sim_pos_hm_neg, dim=1)[:, 0].mean()  # CSCL loss
        loss_our_2 = -1 * F.log_softmax(sim_pos_hm_2nd_neg, dim=1)[:, 0].mean()  # discard -> abl
        loss_our_3 = (-1 * F.log_softmax(sim_pos_hh_neg, dim=1)[:, 0] * sim_hh_weight).mean()  # WSCL loss

        loss = (loss_our + loss_our_3) / 4.0


        return loss
