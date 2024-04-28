import torch
import torch.nn.functional as F

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p)
    en = -torch.sum(p * torch.log(p+1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en

def ova_loss_amlp(out_open, label, lambada=1.5):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    # out_open = F.softmax(out_open, 1)
    label_p = torch.zeros((out_open.size(0),
                           out_open.size(2))).long().cuda()
    label_range = torch.arange(0, out_open.size(0) ).long()
    label_p[label_range, label] = 1
    label_n = 1 - label_p
    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                    + 1e-8) * label_p, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                1e-8) * label_n, 1)[0])
    known_p_first=torch.max(out_open[:,1,:],dim=1)[0]
    unknown_p_first=torch.max(out_open[:,0,:],dim=1)[0]                                                
    # unknown_p_first=torch.mean(out_open[:,0,:],dim=1)[0]                                                
    loss_connection_first=torch.mean(-torch.log(torch.sigmoid(known_p_first-unknown_p_first)))*lambada
    return open_loss_pos, open_loss_neg, loss_connection_first


def ova_loss(out_open, label):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    label_p = torch.zeros((out_open.size(0),
                           out_open.size(2))).long().cuda()
    label_range = torch.arange(out_open.size(0)).long()
    label_p[label_range, label] = 1
    label_n = 1 - label_p
    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                    + 1e-8) * label_p, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                1e-8) * label_n, 1)[0])
    return open_loss_pos, open_loss_neg

def open_entropy(out_open):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2
    out_open = F.softmax(out_open, 1) # [36, 2, 10]
    ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
    return ent_open


def ova_loss_dc(out_open, label, kno_num, kno_unk_map, lambada=1.5):
    # e.g. 10/10/11
    # out_open [36, 31]
    # kno_unk_map [20, 31]
    assert len(out_open.size()) == 2
    label_p = torch.zeros((out_open.size(0), kno_num)).long().cuda() # [36, 20]
    item_range = torch.arange(0, out_open.size(0) ).long()
    label_p[item_range, label] = 1
    label_n = 1 - kno_unk_map[label][:, kno_num:].int() # [36, 11]

    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:,:kno_num] + 1e-8) * label_p, dim=1))

    open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, kno_num:] + 1e-8) * label_n, dim=1)[0])

    known_p_first=torch.max(out_open[:, :kno_num],dim=1)[0]
    unknown_p_first=torch.max(out_open[:, kno_num:],dim=1)[0]      
    loss_connection_first=torch.mean(-torch.log(torch.sigmoid(known_p_first-unknown_p_first)))*lambada

    return open_loss_pos, open_loss_neg, loss_connection_first


def open_entropy_dc(out_open, kno_num, kno_unk_map):
    # out_open [36, 31]
    bsz = out_open.size(0)
    kno_unk_map = (kno_unk_map==1) # [20, 31]
    assert len(out_open.size()) == 2
    ent_open_all = torch.zeros(bsz).cuda() # [36]
    for lbl in range(kno_num):
        mask = kno_unk_map[lbl] # [31]
        out_open_lbl = F.softmax(out_open[:, mask], dim=1) # [36, lbl_num]
        
        ent_open_all += torch.sum(-out_open_lbl*torch.log(out_open_lbl + 1e-8), dim=1) # [36]
    ent_open = torch.mean(ent_open_all/kno_num)
    return ent_open


