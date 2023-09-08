import torch
import torch.nn as nn
import torch.nn.functional as F

class Enrich_proto(nn.Module):
    def __init__(self, n_way, k_shot, queries, emb_dim=1024):
        super(Enrich_proto, self).__init__()
        self.emb_dim = emb_dim
        self.n_classes = n_way
        self.n_support = k_shot
        self.n_query = queries

        self.bn = nn.BatchNorm1d(32)

        self.fc = nn.Conv1d(5, 1, 1)
        self.fc2 = nn.Sequential(nn.Conv1d(2, 32, 1),
                                 self.bn,
                                 nn.ReLU())
        self.fc3 = nn.Linear(32, 1)

        self.q = nn.Sequential(nn.Linear(32, 32),
                               nn.ReLU())
        self.k = nn.Sequential(nn.Linear(32, 32),
                               nn.ReLU())
        self.v = nn.Sequential(nn.Linear(32, 32),
                               nn.ReLU())

        # self.transform = nn.Sequential(nn.Linear(1024, 256),
        #                         nn.ReLU())
        # self.q = nn.Sequential(nn.Linear(256, 256),
        #                         nn.ReLU())
        # self.k = nn.Sequential(nn.Linear(256, 256),
        #                         nn.ReLU())
        # self.v = nn.Sequential(nn.Linear(256, 256),
        #                         nn.ReLU())
        # self.transform2 = nn.Sequential(nn.Linear(256, 1024),
        #                         nn.ReLU())


    def meta_attention(self, proto_feat, support_feat):
        supplement_feat = self.fc(support_feat.view(self.n_classes, self.n_support, -1))  # (n_way, 1, 1024)
        p2s = torch.cat([proto_feat.unsqueeze(1), supplement_feat], dim=1)  # (n_way, 2, 1024)
        fusion_feat = self.fc2(p2s).transpose(1, 2)  # (n_way, 1024, 32)
        # fusion_feat = p2s.transpose(1, 2)
        q = self.q(fusion_feat)  # (n_way, 1024, 32)
        k = self.k(fusion_feat)  # (n_way, 1024, 32)
        v = self.v(fusion_feat)  # (n_way, 1024, 32)

        Rs = torch.matmul(q, k.transpose(1, 2))  # (n_way, 1024, 1024)
        Rs = F.softmax(Rs, 2)
        v_hat = torch.matmul(Rs, v)  # (n_way, 1024, 32)
        refine_feat = self.fc3(v_hat).squeeze()  # (n_way, 1024)
        upt_proto = proto_feat + refine_feat
        return upt_proto

    def self_attention(self, proto_feat):
        trans_proto = self.transform(proto_feat)
        q = self.q(trans_proto)  # (proto+query, emb_dim)
        k = self.k(trans_proto)  # (proto+query, emb_dim)
        v = self.v(trans_proto)  # (proto+query, emb_dim)
        R = torch.matmul(q.unsqueeze(-1), k.unsqueeze(1))  # (proto+query, emb_dim, emb_dim)
        R_hat = F.softmax(R, 1)  # (proto+query, emb_dim, emb_dim)
        att_feat = torch.matmul(v.unsqueeze(1), R_hat)  # (proto+query, 1, emb_dim)
        att_feat = self.transform2(att_feat.squeeze())
        upd_feat = att_feat + proto_feat  # (proto+query, emb_dim)
        return upd_feat


    def forward(self, proto_feat, support_feat):
        enhanced_proto = self.meta_attention(proto_feat, support_feat)
        # proto_feat = self.self_attention(proto_feat)
        return enhanced_proto



if __name__ == '__main__':
    proto = torch.rand((5, 1024)).cuda()
    support = torch.rand((25, 1024)).cuda()
    model = Enrich_proto(5, 5, 10).cuda()
    upt_proto = model(proto, support)
    print(upt_proto.shape)